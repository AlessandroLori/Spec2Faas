import asyncio
import hashlib
from dataclasses import dataclass

from app.redataset import ReDatasetRepository
from app.agents.EnvironmentPreparation import (
    GoSourceSyntaxError, EnvironmentPreparationExhausted,
)
from app.validation.DockerGoRunner import DockerGoRunner
from app.validation.ExecutionModels import  ExecutionResult
from app.validation.HumanevalCases import HumanEvalCaseExtractor, UnsupportedTestFormat, original_python_test_source
from app.validation.ReFaaSCases import ReFaaSCaseExtractor
from app.validation.SeBSCases import SeBSCaseExtractor
from app.validation.OutputComparator import ComparisonResult, OutputComparator
from app.validation.PythonReference import PythonReferenceRunner, ReferenceRunResult

from .Messages import TranslationRequest, TranslationResult
from .Translator import Translator


@dataclass(frozen=True, slots=True)
class TranslationEvaluation:
    """Esito complessivo: la validazione e la generazione restano distinguibili."""

    candidate_id: str
    status: str
    prepared_cases: int = 0
    reference: ReferenceRunResult | None = None
    translation: TranslationResult | None = None
    execution: ExecutionResult | None = None
    comparison: ComparisonResult | None = None
    reason: str | None = None
    environment_debuggable: bool = False
    # Ogni chiamata al Translator, anche vuota, resta identificabile e archiviabile.
    translation_attempts: tuple[TranslationResult, ...] = ()
    translation_attempt_think: tuple[bool | None, ...] = ()

    @property
    def passed(self) -> bool:
        return self.status == "passed"


class TranslationPipeline:
    """Coordina in modo deterministico il percorso esplicito Python → Go."""

    def __init__(
        self, *, translator: Translator, go_runner: DockerGoRunner,
        repository: ReDatasetRepository | None = None,
        extractor: HumanEvalCaseExtractor | None = None,
        reference_runner: PythonReferenceRunner | None = None,
        comparator: OutputComparator | None = None,
        environment_manager=None, environment_agent=None,
        translator_retries: int = 0, translator_retry_no_think: bool = False,
        translation_progress=None, provider_runtime=None,
    ) -> None:
        self.repository = repository or ReDatasetRepository()
        self.extractor = extractor or HumanEvalCaseExtractor()
        self.refaas_extractor = ReFaaSCaseExtractor()
        self.provider_runtime = provider_runtime
        self.sebs_extractor = provider_runtime if provider_runtime is not None else SeBSCaseExtractor()
        self.reference_runner = provider_runtime or reference_runner or PythonReferenceRunner()
        self.translator = translator
        self.go_runner = go_runner
        self.comparator = (provider_runtime.comparator if provider_runtime is not None
                           else comparator or OutputComparator())
        self.environment_manager = environment_manager
        self.environment_agent = environment_agent
        if type(translator_retries) is not int or not 0 <= translator_retries <= 5:
            raise ValueError("translator_retries deve essere fra 0 e 5")
        if type(translator_retry_no_think) is not bool:
            raise TypeError("translator_retry_no_think deve essere bool")
        self.translator_retries = translator_retries
        self.translator_retry_no_think = translator_retry_no_think
        self.translation_progress = translation_progress or (lambda message: None)

    async def run(self, candidate_id: str) -> TranslationEvaluation:
        """Non avvia il modello se la baseline non supera i test originali."""
        # Verifica il runtime PRIMA di interrogare il modello a pagamento.
        environment_problem = self.go_runner.preflight()
        if environment_problem is not None:
            return TranslationEvaluation(candidate_id, "environment_not_ready",
                                         reason=environment_problem)

        source = self.repository.load_source(candidate_id)
        if source.candidate.language != "python":
            raise ValueError("Il percorso iniziale accetta soltanto sorgenti Python")

        try:
            if source.candidate.source_id in SeBSCaseExtractor.SOURCE_IDS:
                extractor = self.sebs_extractor
            elif source.candidate.role == "refaas_original_function_source":
                extractor = self.refaas_extractor
            else:
                extractor = self.extractor
            suite = extractor.prepare(source)
        except UnsupportedTestFormat as exc:
            return TranslationEvaluation(candidate_id, "unsupported_tests", reason=exc.reason)

        python_environment = None
        validation_test_source = original_python_test_source(source, suite)
        if self.environment_manager is not None and self.provider_runtime is None:
            # Qualsiasi funzione Python, non solo SeBS, usa l'ambiente
            # preparato prima della baseline. Un errore interrompe la pipeline
            # prima di chiedere all'LLM una traduzione di sorgente non verificato.
            if self.environment_agent is None:
                raise ValueError("FaaSDeployer non configurato per l'ambiente Python")
            try:
                python_environment = await self.environment_agent.prepare_environment(
                    manager=self.environment_manager, language="python",
                    source=source.code,
                    entry=suite.entry_point or source.candidate.entry_point,
                    max_repairs=self.environment_agent.environment_max_repairs,
                    pip_requirements=source.python_requirements_text,
                    **({"test_source": validation_test_source}
                       if validation_test_source is not None else {}),
                )
            except RuntimeError as exc:
                return TranslationEvaluation(candidate_id, "environment_not_ready",
                                             len(suite.cases), reason=str(exc))
            if (python_environment.get("code_sha256") != suite.source_sha256
                    or python_environment.get("status") != "completed"
                    or (source.python_requirements_text is not None and
                        python_environment.get("declared_requirements_sha256") !=
                        source.python_requirements_sha256)):
                return TranslationEvaluation(candidate_id, "environment_not_ready",
                                             len(suite.cases), reason="Manifest Python incoerente")
            verified = await asyncio.to_thread(
                self.reference_runner.verify, source, suite,
                python_environment=python_environment)
        else:
            verified = await asyncio.to_thread(self.reference_runner.verify, source, suite)
        if not verified.passed:
            return TranslationEvaluation(
                candidate_id, "baseline_not_passed", len(suite.cases),
                reference=verified, reason=verified.status,
            )

        # La reference passa prima della traduzione. Con ambiente opzionale
        # attivo, anche la baseline gira nel Docker che verra' profilato.

        request = TranslationRequest(
            candidate_id=candidate_id, source_code=source.code,
            source_language="python", target_language="go",
            entry_point=suite.entry_point or source.candidate.entry_point,
            source_sha256=source.candidate.source_sha256,
        )
        self.translation_progress("Tentativo 1: inferenza LLM in corso")
        translation = await self.translator.translate(request)
        self.translation_progress(
            f"Tentativo 1: {translation.status} | "
            f"motivo={translation.error_reason or 'nessuno'} | "
            f"token={translation.total_tokens if translation.total_tokens is not None else 'n/d'}"
        )
        translation_attempts = [translation]
        translation_modes = [getattr(self.translator, "_ollama_think", None)]
        # Solo le risposte senza codice E senza testo utile sono ritentate.
        # Nessuna trasformazione Go: ogni nuova risposta arriva dal modello.
        # La riduzione a no-think avviene SOLTANTO se richiesta esplicitamente
        # dalla CLI; la prima generazione usa sempre la configurazione originale.
        for number in range(2, self.translator_retries + 2):
            if (translation.generated or translation.status != "invalid_output"
                    or translation.error_reason not in ("empty_response", "non_text_response")):
                break
            override = False if self.translator_retry_no_think else None
            self.translation_progress(
                f"Tentativo {number}: risposta precedente vuota, "
                f"retry con {'no-think' if override is False else 'configurazione originale'}"
            )
            translation = await self.translator.translate(request, think_override=override)
            self.translation_progress(
                f"Tentativo {number}: {translation.status} | "
                f"motivo={translation.error_reason or 'nessuno'} | "
                f"token={translation.total_tokens if translation.total_tokens is not None else 'n/d'}"
            )
            translation_attempts.append(translation)
            translation_modes.append(False if override is False else getattr(self.translator, "_ollama_think", None))
        attempt_metadata = {
            "translation_attempts": tuple(translation_attempts),
            "translation_attempt_think": tuple(translation_modes),
        }
        if (
            translation.candidate_id != candidate_id
            or translation.source_sha256 != suite.source_sha256
            or translation.source_language != "python"
            or translation.target_language != "go"
            or translation.entry_point != request.entry_point
        ):
            return TranslationEvaluation(candidate_id, "invalid_translation_result",
                                         len(suite.cases), verified, translation,
                                         **attempt_metadata)
        if not translation.generated:
            return TranslationEvaluation(
                candidate_id, "translation_not_generated", len(suite.cases),
                reference=verified, translation=translation,
                reason=translation.error_reason or translation.status,
                **attempt_metadata,
            )

        if self.environment_manager is not None:
            try:
                await self.environment_agent.prepare_environment(
                    manager=self.environment_manager, language="go",
                    source=translation.code, entry=request.entry_point,
                    max_repairs=self.environment_agent.environment_max_repairs)
            except GoSourceSyntaxError as exc:
                execution = ExecutionResult(
                    candidate_id=suite.candidate_id, test_sha256=suite.test_sha256,
                    language="go", code_sha256=hashlib.sha256(
                        translation.code.encode("utf-8")).hexdigest(),
                    status="compile_error", cases=(), compile_seconds=0,
                    execution_seconds=0, error_message=str(exc))
                comparison = self.comparator.compare(suite, execution)
                return TranslationEvaluation(candidate_id, "go_execution_not_completed",
                                             len(suite.cases), verified, translation,
                                             execution, comparison, str(exc), **attempt_metadata)
            except EnvironmentPreparationExhausted as exc:
                return TranslationEvaluation(candidate_id, "environment_not_ready",
                                             len(suite.cases), verified, translation,
                                             reason=str(exc),
                                             environment_debuggable=exc.debugger_can_revise_code,
                                             **attempt_metadata)
            except RuntimeError as exc:
                return TranslationEvaluation(candidate_id, "environment_not_ready",
                                             len(suite.cases), verified, translation,
                                             reason=str(exc), **attempt_metadata)
            info = self.environment_manager.go_result
            self.go_runner.build_image = info["build_image"]
            self.go_runner.runtime_image = info["go_image"]
            self.go_runner.go_modules_enabled = info["go_modules_enabled"]
            self.go_runner.go_cgo_enabled = info["go_cgo_enabled"]

        execution = await asyncio.to_thread(
            self.go_runner.run, translation.code, entry_point=request.entry_point, suite=suite,
        )
        # Verifica anche che l'esecutore abbia valutato proprio quel codice.
        if (execution.language != "go" or execution.candidate_id != candidate_id
                or execution.test_sha256 != suite.test_sha256
                or execution.code_sha256 != hashlib.sha256(translation.code.encode("utf-8")).hexdigest()):
            return TranslationEvaluation(candidate_id, "invalid_execution_result",
                                         len(suite.cases), verified, translation, execution,
                                         **attempt_metadata)
        comparison = self.comparator.compare(suite, execution)
        if not execution.completed:
            status = "go_execution_not_completed"
        elif comparison.passed:
            status = "passed"
        else:
            status = "validation_failed"
        return TranslationEvaluation(
            candidate_id, status, len(suite.cases), verified, translation,
            execution, comparison,
            None if status == "passed" else execution.error_message or comparison.error_message,
            **attempt_metadata,
        )
