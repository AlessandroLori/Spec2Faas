from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CaseExecution:
    """Output restituito da un singolo caso nel linguaggio eseguito."""

    case_id: str
    status: str
    value: Any = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Risultati di esecuzione, separati dagli output attesi dei test."""

    candidate_id: str
    test_sha256: str
    language: str
    code_sha256: str
    status: str
    cases: tuple[CaseExecution, ...]
    compile_seconds: float
    execution_seconds: float
    error_message: str | None = None

    @property
    def completed(self) -> bool:
        """Indica che l'esecutore ha restituito un esito per ogni caso."""
        return self.status == "completed"
