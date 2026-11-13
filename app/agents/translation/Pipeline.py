import asyncio
import hashlib
from dataclasses import dataclass

from app.redataset import ReDatasetRepository
from app.validation.DockerGoRunner import DockerGoRunner
from app.validation.ExecutionModels import  ExecutionResult
from app.validation.HumanevalCases import HumanEvalCaseExtractor, UnsupportedTestFormat
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
    ) -> None:
        self.repository = repository or ReDatasetRepository()
        self.extractor = extractor or HumanEvalCaseExtractor()
        self.refaas_extractor = ReFaaSCaseExtractor()
        self.sebs_extractor = SeBSCaseExtractor()
        self.reference_runner = reference_runner or PythonReferenceRunner()
        self.translator = translator
        self.go_runner = go_runner
        self.comparator = comparator or OutputComparator()

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

        verified = await asyncio.to_thread(self.reference_runner.verify, source, suite)
        if not verified.passed:
            return TranslationEvaluation(
                candidate_id, "baseline_not_passed", len(suite.cases),
                reference=verified, reason=verified.status,
            )

        request = TranslationRequest(
            candidate_id=candidate_id, source_code=source.code,
            source_language="python", target_language="go",
            entry_point=suite.entry_point or source.candidate.entry_point,
            source_sha256=source.candidate.source_sha256,
        )
        translation = await self.translator.translate(request)
        if (
            translation.candidate_id != candidate_id
            or translation.source_sha256 != suite.source_sha256
            or translation.source_language != "python"
            or translation.target_language != "go"
            or translation.entry_point != request.entry_point
        ):
            return TranslationEvaluation(candidate_id, "invalid_translation_result",
                                         len(suite.cases), verified, translation)
        if not translation.generated:
            return TranslationEvaluation(
                candidate_id, "translation_not_generated", len(suite.cases),
                reference=verified, translation=translation,
                reason=translation.error_reason or translation.status,
            )

        execution = await asyncio.to_thread(
            self.go_runner.run, translation.code, entry_point=request.entry_point, suite=suite,
        )
        # Verifica anche che l'esecutore abbia valutato proprio quel codice.
        if (execution.language != "go" or execution.candidate_id != candidate_id
                or execution.test_sha256 != suite.test_sha256
                or execution.code_sha256 != hashlib.sha256(translation.code.encode("utf-8")).hexdigest()):
            return TranslationEvaluation(candidate_id, "invalid_execution_result",
                                         len(suite.cases), verified, translation, execution)
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
        )
