"""Contratti tipizzati per un singolo tentativo di correzione LLM."""

from dataclasses import dataclass
import hashlib


@dataclass(frozen=True, slots=True)
class DebugRequest:
    """Sorgente, traduzione corrente e diagnostica, senza test generati."""

    candidate_id: str
    source_code: str
    current_code: str
    entry_point: str
    failure_kind: str
    failure_details: str
    attempt: int = 1
    source_language: str = "python"
    target_language: str = "go"
    source_sha256: str | None = None
    current_code_sha256: str | None = None
    failed_case_id: str | None = None


@dataclass(frozen=True, slots=True)
class DebugResult:
    """Output di un tentativo LLM; non implica compilazione o test superati."""

    candidate_id: str
    source_sha256: str
    previous_code_sha256: str
    entry_point: str
    model: str
    attempt: int
    failure_kind: str
    status: str
    code: str | None
    raw_response: str | None
    generation_seconds: float
    total_tokens: int | None
    error_reason: str | None = None

    @property
    def proposed(self) -> bool:
        """Indica solo che la risposta soddisfa il formato Go richiesto."""
        return self.status == "proposed" and self.code is not None

    @property
    def code_sha256(self) -> str | None:
        """Impronta del codice proposto, distinta dalla versione precedente."""
        return (hashlib.sha256(self.code.encode("utf-8")).hexdigest()
                if self.code is not None else None)
