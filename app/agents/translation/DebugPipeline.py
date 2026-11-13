"""Coordina le correzioni LLM dopo una traduzione già valutata."""

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from app.validation.ExecutionModels import ExecutionResult
from app.validation.OutputComparator import ComparisonResult
from app.validation.SeBSCases import SeBSCaseExtractor

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

    @property
    def passed(self) -> bool:
        return self.status == "passed"


@dataclass(frozen=True, slots=True)
class DebugPipelineEvaluation:
    """Conserva separatamente la baseline e ogni successiva correzione."""

    initial: TranslationEvaluation
    attempts: tuple[DebugAttemptEvaluation, ...]
    status: str

    @property
    def passed(self) -> bool:
        return self.initial.passed or any(item.passed for item in self.attempts)


class DebugPipeline:
    """Aggiunge debugging solo via LLM, senza modificare TranslationPipeline."""

    # L'esito di un container o di un servizio non deve essere attribuito al codice.
    _DEBUGGABLE_EXECUTIONS = frozenset({
        "compile_error", "runtime_error", "protocol_error", "execution_timeout",
    })
    # Unica fonte del limite massimo, condivisa con l'interfaccia da terminale.
    MAX_DEBUG_ATTEMPTS = 30

    def __init__(
        self, *, pipeline: TranslationPipeline, debugger: TranslationDebugger,
        max_debug_attempts: int = 1,
    ) -> None:
        if (type(max_debug_attempts) is not int
                or not 0 <= max_debug_attempts <= self.MAX_DEBUG_ATTEMPTS):
            raise ValueError(
                f"I tentativi LLM devono essere compresi tra 0 e {self.MAX_DEBUG_ATTEMPTS}"
            )
        self.pipeline = pipeline
        self.debugger = debugger
        self.max_debug_attempts = max_debug_attempts

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
        return False

    @staticmethod
    def _json_short(value: Any, max_chars: int = 2400) -> str:
        """Limita la dimensione della diagnostica senza modificare i casi di test."""
        serialized = json.dumps(value, ensure_ascii=False, default=str, allow_nan=False)
        return serialized if len(serialized) <= max_chars else serialized[:max_chars] + "... [troncato]"

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
                    lines += ["Input arguments: " + cls._json_short(expected.args),
                              "Expected output: " + cls._json_short(expected.expected)]
                if observed is not None:
                    lines += ["Actual output: " + cls._json_short(observed.value),
                              "Execution status: " + observed.status]
                    if observed.error_message:
                        lines.append("Runtime detail: " + observed.error_message[:1200])
            return "validation_failed", case_id, "\n".join(lines)
        return status, None, reason or status

    async def run(self, candidate_id: str) -> DebugPipelineEvaluation:
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

        for attempt in range(1, self.max_debug_attempts + 1):
            failure_kind, failed_case_id, feedback = self._feedback(
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

            # Non applichiamo fix sintattici, di import o di entry point.
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
            if executed.completed and compared.passed:
                attempts.append(DebugAttemptEvaluation(request, proposed, "passed", executed, compared))
                return DebugPipelineEvaluation(initial, tuple(attempts), "passed_after_debug")
            outcome = "validation_failed" if executed.completed else "go_execution_not_completed"
            diagnostic = executed.error_message or compared.error_message
            attempts.append(DebugAttemptEvaluation(
                request, proposed, outcome, executed, compared, diagnostic,
            ))
            if not executed.completed and executed.status not in self._DEBUGGABLE_EXECUTIONS:
                break
            current_code, status, reason = proposed.code, outcome, diagnostic
            execution, comparison = executed, compared

        return DebugPipelineEvaluation(initial, tuple(attempts), "debug_not_resolved")
