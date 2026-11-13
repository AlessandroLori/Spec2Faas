from dataclasses import dataclass
from typing import Any
import json
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .ExecutionModels import ExecutionResult
from .HumanevalCases import PreparedCase, PreparedSuite
from .ReFaaSCases import ReFaaSCaseExtractor
from .SeBSCases import SeBSCaseExtractor


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Risultato del confronto tra gli output eseguiti e quelli attesi."""

    candidate_id: str
    status: str
    total_cases: int
    passed_cases: int
    failed_case_id: str | None = None
    error_message: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "passed" and self.passed_cases == self.total_cases


class OutputComparator:
    """Applica le regole originali HumanEval+ agli output serializzati."""

    COMPARATOR = "humaneval_plus_assertion"

    def compare(self, suite: PreparedSuite, result: ExecutionResult) -> ComparisonResult:
        """Confronta ogni output senza eseguire il codice della funzione."""
        is_refaas = suite.comparator == ReFaaSCaseExtractor.COMPARATOR
        is_sebs = suite.comparator == SeBSCaseExtractor.COMPARATOR
        is_sebs_derived = suite.comparator == SeBSCaseExtractor.DERIVED_COMPARATOR
        if not is_refaas and not is_sebs and not is_sebs_derived and suite.comparator != self.COMPARATOR:
            raise ValueError("Comparatore non supportato")
        if suite.candidate_id != result.candidate_id or suite.test_sha256 != result.test_sha256:
            raise ValueError("I risultati non appartengono alla suite selezionata")
        if not result.completed:
            return ComparisonResult(
                candidate_id=suite.candidate_id, status="not_executed",
                total_cases=len(suite.cases), passed_cases=0,
                error_message=f"Esecuzione non completata: {result.status}",
            )
        if len(result.cases) != len(suite.cases):
            raise ValueError("Numero di risultati diverso dal numero di casi")

        for index, (case, actual) in enumerate(zip(suite.cases, result.cases)):
            if case.case_id != actual.case_id:
                raise ValueError("Ordine o identificativo dei casi non coerente")
            if actual.status != "returned":
                return ComparisonResult(
                    candidate_id=suite.candidate_id,
                    status="execution_error", total_cases=len(suite.cases),
                    passed_cases=index, failed_case_id=case.case_id,
                    error_message=actual.error_message or "Errore nella funzione eseguita",
                )
            if not (self._equivalent_refaas(actual.value, case.expected) if is_refaas
                    else self._validate_sebs(suite, case, actual.value, result.language) if is_sebs
                    else self._equivalent_derived(actual.value, case.expected) if is_sebs_derived
                    else self._equivalent(actual.value, case)):
                return ComparisonResult(
                    candidate_id=suite.candidate_id,
                    status="failed", total_cases=len(suite.cases),
                    passed_cases=index, failed_case_id=case.case_id,
                    error_message="Output diverso dall'atteso",
                )
        return ComparisonResult(suite.candidate_id, "passed", len(suite.cases),
                                len(suite.cases))

    @staticmethod
    def _equivalent_derived(actual: Any, expected: Any) -> bool:
        """Uguaglianza strutturale JSON esatta, senza coercizioni bool/int Python."""
        try:
            return json.dumps(actual, sort_keys=True, ensure_ascii=False, allow_nan=False) == (
                json.dumps(expected, sort_keys=True, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _validate_sebs(suite: PreparedSuite, case: PreparedCase, actual: Any,
                       language: str) -> bool:
        """Chiama validate_output ORIGINALE in processo separato, non una sua replica."""
        if (suite.validator_code is None or
                hashlib.sha256(suite.validator_code.encode("utf-8")).hexdigest() != suite.test_sha256):
            raise ValueError("Hash del validatore SeBS originale non coerente")
        payload = {"test_code": suite.validator_code, "event": case.args[0],
                   "output": actual, "language": language}
        worker = Path(__file__).with_name("SeBSValidatorWorker.py")
        if not worker.is_file():
            raise RuntimeError("Validatore SeBS non disponibile: SeBSValidatorWorker.py mancante")
        with tempfile.TemporaryDirectory(prefix="sebs_validation_") as workdir:
            try:
                finished = subprocess.run(
                    [sys.executable, "-I", str(worker)],
                    input=json.dumps(payload, allow_nan=False), capture_output=True,
                    text=True, cwd=workdir, timeout=5, check=False)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError("Impossibile eseguire il validatore originale SeBS") from exc
        if finished.returncode != 0:
            raise RuntimeError(f"Processo validatore SeBS terminato con codice {finished.returncode}")
        try:
            reply = json.loads(finished.stdout)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Risposta non JSON del validatore originale SeBS") from exc
        if not isinstance(reply, dict) or type(reply.get("valid")) is not bool:
            description = reply.get("error_message", "risposta non valida") if isinstance(reply, dict) else "risposta non valida"
            raise RuntimeError(f"Validatore SeBS non eseguibile: {description}")
        return reply["valid"]

    @staticmethod
    def _equivalent_refaas(actual: Any, expected: Any) -> bool:
        """Il campo body di Lambda è una stringa JSON: confrontiamone la struttura."""
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            return actual == expected
        actual, expected = actual.copy(), expected.copy()
        if isinstance(actual.get("body"), str) and isinstance(expected.get("body"), str):
            try:
                actual["body"] = json.loads(actual["body"])
                expected["body"] = json.loads(expected["body"])
            except ValueError:
                pass
        return actual == expected

    @staticmethod
    def _is_floats(value: Any) -> bool:
        """Riproduce la regola is_floats dei test originali per valori JSON."""
        if type(value) is float:
            return True
        if isinstance(value, list):
            return all(type(item) is float for item in value)
        return False

    @classmethod
    def _equivalent(cls, actual: Any, case: PreparedCase) -> bool:
        """Usa uguaglianza Python e, quando previsto, np.allclose."""
        expected = case.expected
        exact = actual == expected
        if exact:
            return True
        tolerance = case.atol
        if tolerance == 0 and cls._is_floats(expected):
            tolerance = 1e-6
        if tolerance == 0:
            return False
        try:
            return bool(np.allclose(actual, expected, rtol=1e-7, atol=tolerance))
        except (ValueError, TypeError):
            return False
