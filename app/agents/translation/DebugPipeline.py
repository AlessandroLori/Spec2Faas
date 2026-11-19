"""Coordina le correzioni LLM dopo una traduzione già valutata."""

import asyncio
import hashlib
import json
from itertools import count
from dataclasses import dataclass, replace
from typing import Any

from app.agents.EnvironmentPreparation import (
    GoSourceSyntaxError, EnvironmentPreparationExhausted,
)
from app.validation.ExecutionModels import ExecutionResult
from app.validation.OutputComparator import ComparisonResult
from app.validation.SeBSCases import SeBSCaseExtractor
from app.validation.ReFaaSCases import ReFaaSCaseExtractor

from .DebugMessages import DebugRequest, DebugResult
from .Debugger import TranslationDebugger
from .Pipeline import TranslationEvaluation, TranslationPipeline


@dataclass(frozen=True, slots=True)
class DebugAttemptEvaluation:
    """Un tentativo LLM e la sua eventuale validazione Go."""

    request: DebugRequest
    result: DebugResult
    status: str
    execution: ExecutionResult | None = None
    comparison: ComparisonResult | None = None
    reason: str | None = None
    # True solo se una revisione LLM diventa la nuova base per il debug.
    selected_as_checkpoint: bool = False
    # Il confronto ufficiale conta il prefisso; questo numero conta TUTTI
    # i casi originali passati, valutati singolarmente con lo stesso oracolo.
    verified_case_passes: int | None = None
    # ID dei casi verificati individualmente; non modificano l'oracolo.
    verified_passed_case_ids: tuple[str, ...] | None = None
    verified_failed_case_ids: tuple[str, ...] | None = None
    # Revisioni con pari numero di successi ma copertura diversa: esplorazione
    # temporanea senza sostituire il miglior checkpoint protetto.
    selected_for_exploration: bool = False

    @property
    def passed(self) -> bool:
        return self.status == "passed"


@dataclass(frozen=True, slots=True)
class DebugPipelineEvaluation:
    """Conserva separatamente la baseline e ogni successiva correzione."""

    initial: TranslationEvaluation
    attempts: tuple[DebugAttemptEvaluation, ...]
    status: str
    checkpoint_source_attempt: int = 0

    @property
    def passed(self) -> bool:
        return self.initial.passed or any(item.passed for item in self.attempts)


class DebugPipeline:
    """Aggiunge debugging solo via LLM, senza modificare TranslationPipeline."""

    # L'esito di un container o di un servizio non deve essere attribuito al codice.
    _DEBUGGABLE_EXECUTIONS = frozenset({
        "compile_error", "runtime_error", "protocol_error", "execution_timeout",
    })
    # Limite per i tentativi finiti; -1 indica una sequenza senza tetto numerico.
    MAX_DEBUG_ATTEMPTS = 30

    def __init__(
        self, *, pipeline: TranslationPipeline, debugger: TranslationDebugger,
        max_debug_attempts: int = 1, strategy: str = "checkpoint",
    ) -> None:
        if (type(max_debug_attempts) is not int
                or (max_debug_attempts != -1
                    and not 0 <= max_debug_attempts <= self.MAX_DEBUG_ATTEMPTS)):
            raise ValueError(
                f"I tentativi LLM devono essere -1 (senza limite) oppure fra 0 e "
                f"{self.MAX_DEBUG_ATTEMPTS}"
            )
        if strategy not in ("linear", "checkpoint"):
            raise ValueError("Strategia Debugger non valida")
        self.strategy = strategy
        self.pipeline = pipeline
        self.debugger = debugger
        self.max_debug_attempts = max_debug_attempts

    @staticmethod
    def _verified_case_ids(suite, execution: ExecutionResult | None, comparator) -> tuple[str, ...] | None:
        """Elenca i casi superati con l'oracolo originale, senza fermarsi al primo FAIL."""
        if execution is None or not execution.completed:
            return None
        if len(execution.cases) != len(suite.cases):
            raise ValueError("Esiti Go non allineati alla suite originale")
        passed = []
        for expected, observed in zip(suite.cases, execution.cases):
            if expected.case_id != observed.case_id:
                raise ValueError("Ordine dei casi Go non coerente")
            one_suite = replace(suite, cases=(expected,))
            one_execution = replace(execution, cases=(observed,))
            if comparator.compare(one_suite, one_execution).passed:
                passed.append(expected.case_id)
        return tuple(passed)

    @staticmethod
    def _verified_case_passes(suite, execution: ExecutionResult | None, comparator) -> int | None:
        """Compatibilita': il vecchio valore numerico resta disponibile."""
        ids = DebugPipeline._verified_case_ids(suite, execution, comparator)
        return len(ids) if ids is not None else None

    @staticmethod
    def _failed_case_ids(suite, passed_ids: tuple[str, ...] | None) -> tuple[str, ...] | None:
        """Tutti gli ID non superati, non solo il primo errore del comparatore."""
        if passed_ids is None:
            return None
        passed = set(passed_ids)
        return tuple(case.case_id for case in suite.cases if case.case_id not in passed)

    @staticmethod
    def _can_debug(initial: TranslationEvaluation) -> bool:
        """Evita nuove inferenze in caso di errore infrastrutturale o della baseline."""
        translation = initial.translation
        if not initial.reference or not initial.reference.passed or translation is None:
            return False
        if initial.status == "translation_not_generated":
            return bool(translation.raw_response and translation.raw_response.strip())
        if initial.status == "validation_failed":
            return bool(translation.code)
        if initial.status == "go_execution_not_completed":
            return bool(translation.code and initial.execution
                        and initial.execution.status in DebugPipeline._DEBUGGABLE_EXECUTIONS)
        if initial.status == "environment_not_ready":
            return bool(translation.code and initial.environment_debuggable)
        return False

    @staticmethod
    def _json_short(value: Any, max_chars: int = 2400) -> str:
        """Limita la dimensione della diagnostica senza modificare i casi di test."""
        serialized = json.dumps(value, ensure_ascii=False, default=str, allow_nan=False)
        return serialized if len(serialized) <= max_chars else serialized[:max_chars] + "... [troncato]"

    @staticmethod
    def _refaas_normalize_bodies(expected: Any, actual: Any) -> tuple[Any, Any]:
        """Decodifica body solo quando lo fa anche il comparatore ReFaaS."""
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return expected, actual
        if not isinstance(expected.get("body"), str) or not isinstance(actual.get("body"), str):
            return expected, actual
        try:
            ebody = json.loads(expected["body"])
            abody = json.loads(actual["body"])
        except (TypeError, ValueError):
            return expected, actual
        return {**expected, "body": ebody}, {**actual, "body": abody}

    @classmethod
    def _first_structural_difference(
        cls, expected: Any, actual: Any, path: str = "$", depth: int = 0,
    ) -> str | None:
        """Descrive la prima differenza, senza sostituire l'oracolo originale."""
        if expected == actual:
            return None
        if depth >= 24:
            return f"At {path}: nested values differ (depth limit)."
        if isinstance(expected, dict) and isinstance(actual, dict):
            missing = sorted(expected.keys() - actual.keys(), key=str)
            if missing:
                key = missing[0]
                return (f"At {path}.{key}: required key is MISSING. "
                        f"Actual keys at {path}: {cls._json_short(sorted(actual.keys(), key=str), 250)}")
            extra = sorted(actual.keys() - expected.keys(), key=str)
            if extra:
                key = extra[0]
                return (f"At {path}.{key}: UNEXPECTED key in actual output. "
                        f"Expected keys: {cls._json_short(sorted(expected.keys(), key=str), 250)}")
            for key in sorted(expected.keys(), key=str):
                difference = cls._first_structural_difference(
                    expected[key], actual[key], f"{path}.{key}", depth + 1)
                if difference is not None:
                    return difference
        elif isinstance(expected, list) and isinstance(actual, list):
            if len(expected) != len(actual):
                return f"At {path}: expected list length {len(expected)}, got {len(actual)}."
            for index, (e, a) in enumerate(zip(expected, actual)):
                difference = cls._first_structural_difference(e, a, f"{path}[{index}]", depth + 1)
                if difference is not None:
                    return difference
        return (f"At {path}: expected {cls._json_short(expected, 400)}; "
                f"actual {cls._json_short(actual, 400)}. "
                f"Expected type {type(expected).__name__}, actual type {type(actual).__name__}.")

    @classmethod
    def _refaas_feedback(cls, *, expected_case, observed_case) -> list[str]:
        """Feedback sintetico per la stringa JSON annidata delle risposte Lambda."""
        expected = expected_case.expected
        actual = observed_case.value if observed_case is not None else None
        normalized_expected, normalized_actual = cls._refaas_normalize_bodies(expected, actual)
        lines = [
            "ReFaaS Lambda contract: response.body must be a JSON-encoded string "
            "when the Python reference returns a string. Keep the exact response "
            "envelope (including nested keys) and original error behavior.",
        ]
        if observed_case is not None and observed_case.status != "returned":
            lines.append(f"Go execution status: {observed_case.status}")
            if observed_case.error_message:
                lines.append("Runtime detail: " + observed_case.error_message[:500])
            return lines
        difference = cls._first_structural_difference(normalized_expected, normalized_actual)
        if difference is not None:
            lines.append("First structural mismatch (read-only diagnostic): " + difference)
        else:
            lines.append("Outputs differ under the original comparator; check raw JSON types.")
        # Per casi brevi manteniamo input e output completi. Per casi grandi,
        # il percorso strutturale evita migliaia di caratteri di JSON ripetuti.
        encoded_input = cls._json_short(expected_case.args)
        if len(encoded_input) <= 500:
            lines.append("Input arguments: " + encoded_input)
        else:
            lines.append("Input arguments (excerpt): " + encoded_input[:350] + "... [truncated]")
        return lines

    @classmethod
    def _feedback(
        cls, *, status: str, reason: str | None,
        execution: ExecutionResult | None,
        comparison: ComparisonResult | None, suite,
    ) -> tuple[str, str | None, str]:
        """Descrive un problema usando esclusivamente gli esiti già osservati."""
        if status == "translation_not_generated":
            return "invalid_output", None, f"Invalid Go translation: {reason or 'invalid_output'}"
        if execution is not None and not execution.completed:
            details = execution.error_message or execution.status
            return execution.status, None, details
        if comparison is not None and not comparison.passed:
            case_id = comparison.failed_case_id
            lines = [f"Validation: {comparison.status}.",
                     f"Failed case: {case_id or 'not available'}.",
                     comparison.error_message or "Output mismatch."]
            if case_id:
                expected = next((c for c in suite.cases if c.case_id == case_id), None)
                observed = next((c for c in execution.cases if c.case_id == case_id), None) if execution else None
                if expected is not None:
                    if suite.comparator == ReFaaSCaseExtractor.COMPARATOR:
                        lines.extend(cls._refaas_feedback(
                            expected_case=expected, observed_case=observed))
                        return "validation_failed", case_id, "\n".join(lines)
                    lines.append("Input arguments: " + cls._json_short(expected.args))
                    if suite.comparator == SeBSCaseExtractor.COMPARATOR:
                        # SeBS verifica con validate_output() e NON dispone
                        # necessariamente di un expected JSON. Non inventarlo.
                        lines.append("Expected behavior: use the original SeBS "
                                     "validate_output() oracle, not a literal JSON expected value.")
                        if suite.validator_code:
                            validator = suite.validator_code
                            lines.append("Original SeBS generator and validator (read-only):\n" +
                                         (validator if len(validator) <= 9500 else
                                          validator[:9500] + "\n[validator truncated]"))
                    else:
                        lines.append("Expected output: " + cls._json_short(expected.expected))
                if observed is not None:
                    lines += ["Actual output: " + cls._json_short(observed.value),
                              "Execution status: " + observed.status]
                    if observed.error_message:
                        lines.append("Runtime detail: " + observed.error_message[:1200])
            return "validation_failed", case_id, "\n".join(lines)
        return status, None, reason or status

    async def run(self, candidate_id: str) -> DebugPipelineEvaluation:
        """La prima traduzione rimane intatta; ogni correzione richiede un LLM."""
        if self.strategy == "linear":
            return await self._run_linear(candidate_id)
        initial = await self.pipeline.run(candidate_id)
        if initial.passed:
            return DebugPipelineEvaluation(initial, (), "passed_initially")
        if self.max_debug_attempts == 0 or not self._can_debug(initial):
            return DebugPipelineEvaluation(initial, (), initial.status)

        source = self.pipeline.repository.load_source(candidate_id)
        # Lo stesso criterio di selezione di TranslationPipeline.run: la
        # fase di debug deve riutilizzare ESATTAMENTE la suite già verificata.
        if source.candidate.source_id in SeBSCaseExtractor.SOURCE_IDS:
            extractor = self.pipeline.sebs_extractor
        elif source.candidate.role == "refaas_original_function_source":
            extractor = self.pipeline.refaas_extractor
        else:
            extractor = self.pipeline.extractor
        suite = extractor.prepare(source)
        translation = initial.translation
        assert translation is not None
        if (source.candidate.source_sha256 != translation.source_sha256
                or suite.source_sha256 != translation.source_sha256
                or suite.test_sha256 != initial.reference.test_sha256):
            return DebugPipelineEvaluation(initial, (), "invalid_debug_inputs")

        current_code = translation.code or translation.raw_response
        assert current_code is not None
        status = initial.status
        reason = initial.reason
        execution = initial.execution
        comparison = initial.comparison
        checkpoint_ids = self._verified_case_ids(suite, execution, self.pipeline.comparator)
        checkpoint_score = len(checkpoint_ids) if checkpoint_ids is not None else None
        # Il miglior checkpoint positivo è distinto dalla base attiva. Una
        # revisione a pari punteggio con casi diversi può essere ESPLORATA
        # senza cancellare il miglior Go già verificato.
        best_checkpoint = ((current_code, status, reason, execution, comparison,
                            checkpoint_ids, 0)
                           if checkpoint_score is not None and checkpoint_score > 0 else None)
        exploring = False
        attempts: list[DebugAttemptEvaluation] = []
        # Il checkpoint mantiene la migliore versione che abbia completato i test.
        # Se non esiste ancora una versione compilabile, si continua con la
        # correzione progressiva dell'ultima proposta LLM come nel flusso originale.
        checkpoint_source_attempt = 0
        rejected_hint: str | None = None
        active_branch_hint: str | None = None

        # Il codice cambia esclusivamente tramite proposta LLM, anche con -1.
        # Senza limite numerico si esce per successo, errore non recuperabile o stop manuale.
        attempts_range = (count(1) if self.max_debug_attempts == -1
                          else range(1, self.max_debug_attempts + 1))
        for attempt in attempts_range:
            failure_kind, failed_case_id, feedback = self._feedback(
                status=status, reason=reason, execution=execution,
                comparison=comparison, suite=suite,
            )
            checkpoint_passed = checkpoint_score
            # I dettagli dei casi non entrano nel prompt se la suite e' grande.
            # Solo gli ID di un piccolo gruppo sono utili al modello.
            if active_branch_hint:
                feedback += "\n\n" + active_branch_hint
            if rejected_hint:
                feedback += ("\n\nRECENT REJECTED LLM REVISION (not the current Go "
                             "checkpoint):\n" + rejected_hint)
            request = DebugRequest(
                candidate_id=candidate_id,
                source_code=source.code,
                current_code=current_code,
                entry_point=translation.entry_point,
                failure_kind=failure_kind,
                failure_details=feedback,
                attempt=attempt,
                source_sha256=translation.source_sha256,
                current_code_sha256=hashlib.sha256(current_code.encode("utf-8")).hexdigest(),
                failed_case_id=failed_case_id,
                checkpoint_passed_cases=checkpoint_passed,
                checkpoint_total_cases=(comparison.total_cases if checkpoint_passed is not None else None),
                checkpoint_passed_case_ids=(checkpoint_ids if checkpoint_ids is not None else None),
            )
            proposed = await self.debugger.debug(request)
            if (
                proposed.candidate_id != candidate_id
                or proposed.source_sha256 != translation.source_sha256
                or proposed.previous_code_sha256 != request.current_code_sha256
                or proposed.entry_point != request.entry_point
                or proposed.attempt != attempt
            ):
                attempts.append(DebugAttemptEvaluation(request, proposed, "invalid_debug_result"))
                break
            if not proposed.proposed:
                attempts.append(DebugAttemptEvaluation(
                    request, proposed, proposed.status, reason=proposed.error_reason,
                ))
                if proposed.status != "invalid_output":
                    # Timeout/provider error: non far finta che sia un errore del codice.
                    break
                # Una risposta malformata/assente NON sostituisce il codice
                # corrente, neanche se contiene un frammento Go. Il frammento
                # resta integralmente salvato come artefatto della revisione.
                rejected_hint = (
                    f"Attempt {attempt}: invalid model output ({proposed.error_reason or proposed.status}). "
                    "Keep the current Go checkpoint; return a complete valid Go file."
                    + (f" Raw reply length: {len(proposed.raw_response)} chars."
                       if proposed.raw_response else "")
                )
                if exploring and best_checkpoint is not None:
                    (current_code, status, reason, execution, comparison,
                     checkpoint_ids, _) = best_checkpoint
                    checkpoint_score = len(checkpoint_ids)
                    exploring = False
                    active_branch_hint = None
                continue

            # Le dipendenze possono cambiare con ogni revisione Go LLM.
            syntax_failure = None
            if getattr(self.pipeline, "environment_manager", None) is not None:
                try:
                    await self.pipeline.environment_agent.prepare_environment(
                        manager=self.pipeline.environment_manager, language="go",
                        source=proposed.code, entry=translation.entry_point,
                        max_repairs=self.pipeline.environment_agent.environment_max_repairs)
                except GoSourceSyntaxError as exc:
                    syntax_failure = str(exc)
                except EnvironmentPreparationExhausted as exc:
                    if not exc.debugger_can_revise_code:
                        attempts.append(DebugAttemptEvaluation(
                            request, proposed, "environment_not_ready", reason=str(exc)))
                        break
                    # La build e' fallita a causa di import o moduli del Go
                    # proposto. Solo un'altra revisione LLM puo' cambiare il
                    # codice. Il Docker non viene riparato deterministicamente.
                    has_validated_checkpoint = checkpoint_score is not None
                    promote = not has_validated_checkpoint and proposed.code != current_code
                    attempts.append(DebugAttemptEvaluation(
                        request, proposed, "environment_not_ready", reason=str(exc),
                        selected_as_checkpoint=promote))
                    if promote:
                        current_code, status, reason = (
                            proposed.code, "environment_not_ready", str(exc))
                        execution = comparison = None
                        checkpoint_score = None
                        checkpoint_source_attempt = attempt
                        rejected_hint = None
                    else:
                        rejected_hint = (
                            f"Attempt {attempt}: environment dependency error: {str(exc)[:1700]}. "
                            "Preserve the existing checkpoint while repairing the Go source.")
                    if exploring and best_checkpoint is not None:
                        (current_code, status, reason, execution, comparison,
                         checkpoint_ids, _) = best_checkpoint
                        checkpoint_score = len(checkpoint_ids)
                        exploring = False
                        active_branch_hint = None
                    continue
                except RuntimeError as exc:
                    attempts.append(DebugAttemptEvaluation(
                        request, proposed, "environment_not_ready", reason=str(exc)))
                    break
                if syntax_failure is None:
                    info = self.pipeline.environment_manager.go_result
                    runner = self.pipeline.go_runner
                    runner.build_image = info["build_image"]
                    runner.runtime_image = info["go_image"]
                    runner.go_modules_enabled = info["go_modules_enabled"]
                    runner.go_cgo_enabled = info["go_cgo_enabled"]

            # Solo il Debugger LLM puo' correggere un errore di sintassi Go.
            if syntax_failure is not None:
                executed = ExecutionResult(
                    candidate_id=suite.candidate_id, test_sha256=suite.test_sha256,
                    language="go", code_sha256=hashlib.sha256(proposed.code.encode("utf-8")).hexdigest(),
                    status="compile_error", cases=(), compile_seconds=0,
                    execution_seconds=0, error_message=syntax_failure)
            else:
                executed = await asyncio.to_thread(
                    self.pipeline.go_runner.run, proposed.code,
                    entry_point=translation.entry_point, suite=suite,
                )
            expected_hash = hashlib.sha256(proposed.code.encode("utf-8")).hexdigest()
            if (executed.language != "go" or executed.candidate_id != candidate_id
                    or executed.test_sha256 != suite.test_sha256
                    or executed.code_sha256 != expected_hash):
                attempts.append(DebugAttemptEvaluation(request, proposed, "invalid_execution_result", executed))
                break
            compared = self.pipeline.comparator.compare(suite, executed)
            candidate_ids = self._verified_case_ids(suite, executed, self.pipeline.comparator)
            candidate_score = len(candidate_ids) if candidate_ids is not None else None
            candidate_failed_ids = self._failed_case_ids(suite, candidate_ids)
            if executed.completed and compared.passed:
                attempts.append(DebugAttemptEvaluation(
                    request, proposed, "passed", executed, compared,
                    selected_as_checkpoint=True,
                    verified_case_passes=candidate_score,
                    verified_passed_case_ids=candidate_ids,
                    verified_failed_case_ids=candidate_failed_ids))
                return DebugPipelineEvaluation(initial, tuple(attempts),
                                               "passed_after_debug", attempt)
            outcome = "validation_failed" if executed.completed else "go_execution_not_completed"
            diagnostic = executed.error_message or compared.error_message
            best_score = (len(best_checkpoint[5]) if best_checkpoint is not None else -1)
            # I miglioramenti numerici proteggono un nuovo best checkpoint.
            # A pari punteggio >0, set di test DIVERSI consentono un ramo
            # temporaneo. Al successivo peggioramento si riparte dal best.
            improve_best = (candidate_score is not None and candidate_score > best_score
                            and proposed.code != current_code)
            explore_zero = (candidate_score == 0 and checkpoint_score == 0
                            and proposed.code != current_code)
            explore_tie = (
                candidate_score is not None and candidate_score > 0
                and candidate_score == best_score
                and best_checkpoint is not None
                and candidate_ids != best_checkpoint[5]
                and candidate_ids != checkpoint_ids
                and proposed.code != current_code
            )
            # Quando manca ancora una revisione eseguibile, le proposte LLM
            # recuperabili devono diventare la base del prossimo debug.
            # La v3 scartava erroneamente ogni nuova proposta con errore di
            # compilazione, ripetendo sempre gli errori del Go iniziale.
            advance_unvalidated = (
                checkpoint_score is None
                and candidate_score is None
                and proposed.code != current_code
            )
            promote = improve_best or explore_zero or advance_unvalidated
            attempts.append(DebugAttemptEvaluation(
                request, proposed, outcome, executed, compared, diagnostic,
                selected_as_checkpoint=promote,
                selected_for_exploration=explore_tie,
                verified_case_passes=candidate_score,
                verified_passed_case_ids=candidate_ids,
                verified_failed_case_ids=candidate_failed_ids,
            ))
            if not executed.completed and executed.status not in self._DEBUGGABLE_EXECUTIONS:
                break
            if promote or explore_tie:
                current_code, status, reason = proposed.code, outcome, diagnostic
                execution, comparison = executed, compared
                checkpoint_ids = candidate_ids
                checkpoint_score = candidate_score
                rejected_hint = None
                exploring = bool(explore_tie)
                active_branch_hint = None
                if improve_best and candidate_score is not None and candidate_score > 0:
                    best_checkpoint = (current_code, status, reason, execution,
                                       comparison, checkpoint_ids, attempt)
                    checkpoint_source_attempt = attempt
                elif explore_zero or (promote and candidate_score == 0) or advance_unvalidated:
                    # In assenza di un successo positivo, avanziamo anche su
                    # codice non ancora compilabile: ogni correzione e' LLM.
                    checkpoint_source_attempt = attempt
                if explore_tie:
                    # Questa nota e' breve e riferita al codice attivo, NON
                    # contiene un secondo file Go o output JSON voluminosi.
                    active_branch_hint = (
                        "EXPLORATION BRANCH: this Go version passes a DIFFERENT "
                        "set of original cases at the same score. A separate "
                        "best-tested checkpoint is preserved; improve THIS "
                        "code and retain its passing cases."
                    )
            else:
                _, _, alternative_feedback = self._feedback(
                    status=outcome, reason=diagnostic, execution=executed,
                    comparison=compared, suite=suite)
                rejected_hint = (
                    f"Attempt {attempt} was not selected as the next Go base. "
                    + ("It returned identical Go code. " if proposed.code == current_code else "")
                    + alternative_feedback[:550]
                    + "\nContinue from the ACTIVE Go base, not this rejected alternative. "
                      "Preserve the already passing cases."
                )
                if exploring and best_checkpoint is not None:
                    # Il ramo esplorativo e' peggiorato o non ha migliorato
                    # la copertura: recuperiamo il miglior checkpoint reale.
                    (current_code, status, reason, execution, comparison,
                     checkpoint_ids, _) = best_checkpoint
                    checkpoint_score = len(checkpoint_ids)
                    exploring = False
                    active_branch_hint = None

        return DebugPipelineEvaluation(initial, tuple(attempts),
                                       "debug_not_resolved", checkpoint_source_attempt)

    @classmethod
    def _feedback_legacy(
        cls, *, status: str, reason: str | None,
        execution: ExecutionResult | None,
        comparison: ComparisonResult | None, suite,
    ) -> tuple[str, str | None, str]:
        """Descrive un problema usando esclusivamente gli esiti già osservati."""
        if status == "translation_not_generated":
            return "invalid_output", None, f"Invalid Go translation: {reason or 'invalid_output'}"
        if execution is not None and not execution.completed:
            details = execution.error_message or execution.status
            return execution.status, None, details
        if comparison is not None and not comparison.passed:
            case_id = comparison.failed_case_id
            lines = [f"Validation: {comparison.status}.",
                     f"Failed case: {case_id or 'not available'}.",
                     comparison.error_message or "Output mismatch."]
            if case_id:
                expected = next((c for c in suite.cases if c.case_id == case_id), None)
                observed = next((c for c in execution.cases if c.case_id == case_id), None) if execution else None
                if expected is not None:
                    lines += ["Input arguments: " + cls._json_short(expected.args),
                              "Expected output: " + cls._json_short(expected.expected)]
                if observed is not None:
                    lines += ["Actual output: " + cls._json_short(observed.value),
                              "Execution status: " + observed.status]
                    if observed.error_message:
                        lines.append("Runtime detail: " + observed.error_message[:1200])
            return "validation_failed", case_id, "\n".join(lines)
        return status, None, reason or status

    async def _run_linear(self, candidate_id: str) -> DebugPipelineEvaluation:
        """La prima traduzione rimane intatta; ogni correzione richiede un LLM."""
        initial = await self.pipeline.run(candidate_id)
        if initial.passed:
            return DebugPipelineEvaluation(initial, (), "passed_initially")
        if self.max_debug_attempts == 0 or not self._can_debug(initial):
            return DebugPipelineEvaluation(initial, (), initial.status)

        source = self.pipeline.repository.load_source(candidate_id)
        # Lo stesso criterio di selezione di TranslationPipeline.run: la
        # fase di debug deve riutilizzare ESATTAMENTE la suite già verificata.
        if source.candidate.source_id in SeBSCaseExtractor.SOURCE_IDS:
            extractor = self.pipeline.sebs_extractor
        elif source.candidate.role == "refaas_original_function_source":
            extractor = self.pipeline.refaas_extractor
        else:
            extractor = self.pipeline.extractor
        suite = extractor.prepare(source)
        translation = initial.translation
        assert translation is not None
        if (source.candidate.source_sha256 != translation.source_sha256
                or suite.source_sha256 != translation.source_sha256
                or suite.test_sha256 != initial.reference.test_sha256):
            return DebugPipelineEvaluation(initial, (), "invalid_debug_inputs")

        current_code = translation.code or translation.raw_response
        assert current_code is not None
        status = initial.status
        reason = initial.reason
        execution = initial.execution
        comparison = initial.comparison
        attempts: list[DebugAttemptEvaluation] = []

        for attempt in (count(1) if self.max_debug_attempts == -1
                        else range(1, self.max_debug_attempts + 1)):
            failure_kind, failed_case_id, feedback = (
                self._feedback if suite.comparator == SeBSCaseExtractor.COMPARATOR
                else self._feedback_legacy
            )(
                status=status, reason=reason, execution=execution,
                comparison=comparison, suite=suite,
            )
            request = DebugRequest(
                candidate_id=candidate_id,
                source_code=source.code,
                current_code=current_code,
                entry_point=translation.entry_point,
                failure_kind=failure_kind,
                failure_details=feedback,
                attempt=attempt,
                source_sha256=translation.source_sha256,
                current_code_sha256=hashlib.sha256(current_code.encode("utf-8")).hexdigest(),
                failed_case_id=failed_case_id,
            )
            proposed = await self.debugger.debug(request)
            if (
                proposed.candidate_id != candidate_id
                or proposed.source_sha256 != translation.source_sha256
                or proposed.previous_code_sha256 != request.current_code_sha256
                or proposed.entry_point != request.entry_point
                or proposed.attempt != attempt
            ):
                attempts.append(DebugAttemptEvaluation(request, proposed, "invalid_debug_result"))
                break
            if not proposed.proposed:
                attempts.append(DebugAttemptEvaluation(
                    request, proposed, proposed.status, reason=proposed.error_reason,
                ))
                if proposed.status != "invalid_output":
                    # Timeout/provider error: non far finta che sia un errore del codice.
                    break
                if proposed.error_reason == "empty_response" and proposed.raw_response == "":
                    # Una risposta vuota non e' una nuova versione Go. Manteniamo
                    # il codice e il feedback gia' verificati per il tentativo seguente.
                    continue
                if not proposed.raw_response:
                    break
                # Se esiste una risposta Go non valida, puo' essere riutilizzata
                # dall'LLM come dato diagnostico; nessuna modifica automatica.
                current_code = proposed.raw_response
                status = "translation_not_generated"
                reason = proposed.error_reason
                execution = comparison = None
                continue

            # Le dipendenze possono cambiare con ogni revisione Go LLM.
            syntax_failure = None
            if getattr(self.pipeline, "environment_manager", None) is not None:
                try:
                    await self.pipeline.environment_agent.prepare_environment(
                        manager=self.pipeline.environment_manager, language="go",
                        source=proposed.code, entry=translation.entry_point,
                        max_repairs=self.pipeline.environment_agent.environment_max_repairs)
                except GoSourceSyntaxError as exc:
                    syntax_failure = str(exc)
                except EnvironmentPreparationExhausted as exc:
                    attempts.append(DebugAttemptEvaluation(
                        request, proposed, "environment_not_ready", reason=str(exc)))
                    if not exc.debugger_can_revise_code:
                        break
                    current_code, status, reason = proposed.code, "environment_not_ready", str(exc)
                    execution = comparison = None
                    continue
                except RuntimeError as exc:
                    attempts.append(DebugAttemptEvaluation(
                        request, proposed, "environment_not_ready", reason=str(exc)))
                    break
                if syntax_failure is None:
                    info = self.pipeline.environment_manager.go_result
                    runner = self.pipeline.go_runner
                    runner.build_image = info["build_image"]
                    runner.runtime_image = info["go_image"]
                    runner.go_modules_enabled = info["go_modules_enabled"]
                    runner.go_cgo_enabled = info["go_cgo_enabled"]

            # Solo il Debugger LLM puo' correggere un errore di sintassi Go.
            if syntax_failure is not None:
                executed = ExecutionResult(
                    candidate_id=suite.candidate_id, test_sha256=suite.test_sha256,
                    language="go", code_sha256=hashlib.sha256(proposed.code.encode("utf-8")).hexdigest(),
                    status="compile_error", cases=(), compile_seconds=0,
                    execution_seconds=0, error_message=syntax_failure)
            else:
                executed = await asyncio.to_thread(
                    self.pipeline.go_runner.run, proposed.code,
                    entry_point=translation.entry_point, suite=suite,
                )
            expected_hash = hashlib.sha256(proposed.code.encode("utf-8")).hexdigest()
            if (executed.language != "go" or executed.candidate_id != candidate_id
                    or executed.test_sha256 != suite.test_sha256
                    or executed.code_sha256 != expected_hash):
                attempts.append(DebugAttemptEvaluation(request, proposed, "invalid_execution_result", executed))
                break
            compared = self.pipeline.comparator.compare(suite, executed)
            verified_ids = self._verified_case_ids(suite, executed, self.pipeline.comparator)
            verified_failed = self._failed_case_ids(suite, verified_ids)
            if executed.completed and compared.passed:
                attempts.append(DebugAttemptEvaluation(request, proposed, "passed", executed, compared,
                    selected_as_checkpoint=True,
                    verified_case_passes=len(verified_ids) if verified_ids is not None else None,
                    verified_passed_case_ids=verified_ids,
                    verified_failed_case_ids=verified_failed))
                return DebugPipelineEvaluation(initial, tuple(attempts), "passed_after_debug", attempt)
            outcome = "validation_failed" if executed.completed else "go_execution_not_completed"
            diagnostic = executed.error_message or compared.error_message
            attempts.append(DebugAttemptEvaluation(
                request, proposed, outcome, executed, compared, diagnostic,
                selected_as_checkpoint=True,
                verified_case_passes=len(verified_ids) if verified_ids is not None else None,
                verified_passed_case_ids=verified_ids,
                verified_failed_case_ids=verified_failed,
            ))
            if not executed.completed and executed.status not in self._DEBUGGABLE_EXECUTIONS:
                break
            current_code, status, reason = proposed.code, outcome, diagnostic
            execution, comparison = executed, compared

        return DebugPipelineEvaluation(initial, tuple(attempts), "debug_not_resolved",
                                       attempts[-1].request.attempt if attempts else 0)
