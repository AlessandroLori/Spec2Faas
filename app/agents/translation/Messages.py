from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TranslationRequest:
    """Codice da tradurre e linguaggi scelti prima dell'avvio del workflow."""

    candidate_id: str
    source_code: str
    source_language: str
    target_language: str
    entry_point: str
    source_sha256: str | None = None
    provider_context: str | None = None


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """Risposta del Translator, senza eseguire o validare il codice prodotto."""

    candidate_id: str
    source_sha256: str
    source_language: str
    target_language: str
    entry_point: str
    model: str
    status: str
    code: str | None
    raw_response: str | None
    generation_seconds: float
    total_tokens: int | None
    error_reason: str | None = None
    diagnostics: dict | None = None

    @property
    def generated(self) -> bool:
        """Indica che è stato estratto codice dalla risposta del modello."""
        return self.status == "generated" and self.code is not None
