"""Feedback sperimentale: dati REALI di test, mai correzioni di codice Go.

Il feedback base arriva sempre dalla pipeline originale. Questa implementazione
puo' soltanto aggiungere un caso verificato; non effettua rollback/checkpoint.
"""

from dataclasses import replace
from typing import Any


class FeedbackProfiles:
    """Selezione deterministica e limitata di evidenze per il Debugger linear."""

    MAX_ADDITION_CHARS = 1400

    @staticmethod
    def verified_results(suite, execution, comparator) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
        """Confronta con l'oracolo originale; None se non c'e' esecuzione completa."""
        if execution is None or not execution.completed:
            return None
        if len(execution.cases) != len(suite.cases):
            return None
        passed, failed = [], []
        for expected, observed in zip(suite.cases, execution.cases):
            if expected.case_id != observed.case_id:
                return None
            one_suite = replace(suite, cases=(expected,))
            one_execution = replace(execution, cases=(observed,))
            (passed if comparator.compare(one_suite, one_execution).passed else failed).append(expected.case_id)
        return tuple(passed), tuple(failed)

    @staticmethod
    def _case(suite, execution, case_id: str):
        for expected, observed in zip(suite.cases, execution.cases):
            if expected.case_id == case_id and observed.case_id == case_id:
                return expected, observed
        return None

    @staticmethod
    def _signature(expected, observed) -> tuple[Any, ...]:
        """Categoria semplice e riproducibile, non un classificatore LLM."""
        if observed.status != "returned":
            return (observed.status, (observed.error_message or "")[:180])
        return ("returned", type(expected.expected).__name__, type(observed.value).__name__)

    @classmethod
    def _excerpt(cls, suite, execution, case_id: str, json_short) -> str:
        item = cls._case(suite, execution, case_id)
        if item is None:
            return ""
        expected, observed = item
        detail = (
            f"Case ID: {case_id}\n"
            f"Input arguments: {json_short(expected.args, 320)}\n"
            f"Expected output: {json_short(expected.expected, 320)}\n"
            f"Actual output: {json_short(observed.value, 320)}\n"
            f"Execution status: {observed.status}"
        )
        if observed.error_message:
            detail += "\nRuntime detail: " + observed.error_message[:300]
        return detail[:cls.MAX_ADDITION_CHARS]

    @classmethod
    def enhance(cls, *, variant: str, baseline: str, suite, execution, comparison,
                comparator, json_short, historical_passed: set[str],
                historical_failed: set[str]) -> str:
        """Aggiunge una sola osservazione diversa, solo se verificabile."""
        if variant == "baseline":
            return baseline
        if variant not in ("feedback-v1", "feedback-history-v1"):
            raise ValueError("Profilo Feedback non valido")
        results = cls.verified_results(suite, execution, comparator)
        if results is None or comparison is None or comparison.passed:
            return baseline
        passed_ids, failed_ids = results
        if variant == "feedback-v1":
            first = comparison.failed_case_id
            first_item = cls._case(suite, execution, first) if first else None
            first_signature = cls._signature(*first_item) if first_item else None
            candidate = None
            for case_id in failed_ids:
                if case_id == first:
                    continue
                item = cls._case(suite, execution, case_id)
                if item is not None and cls._signature(*item) != first_signature:
                    candidate = case_id
                    break
            if candidate is None:
                return baseline
            addition = cls._excerpt(suite, execution, candidate, json_short)
            return baseline + "\n\nADDITIONAL DISTINCT FAILED CASE (current Go revision; verified):\n" + addition

        # Nessuna supposizione di successo se il codice non ha completato i test.
        passed, failed = set(passed_ids), set(failed_ids)
        regressions = failed.intersection(historical_passed)
        if regressions:
            case_id = next(c.case_id for c in suite.cases if c.case_id in regressions)
            addition = cls._excerpt(suite, execution, case_id, json_short)
            return baseline + "\n\nREGRESSION (passed in an earlier tested revision, FAILED now):\n" + addition
        repaired = passed.intersection(historical_failed)
        if repaired:
            case_id = next(c.case_id for c in suite.cases if c.case_id in repaired)
            item = cls._case(suite, execution, case_id)
            expected, _ = item
            addition = (
                f"Case ID: {case_id}\n"
                f"Input arguments: {json_short(expected.args, 320)}\n"
                f"Expected output: {json_short(expected.expected, 320)}\n"
                "Status: PASSED in the CURRENT tested Go revision."
            )
            return baseline + "\n\nPREVIOUSLY FIXED CASE (do not regress this behavior):\n" + addition[:cls.MAX_ADDITION_CHARS]
        return baseline
