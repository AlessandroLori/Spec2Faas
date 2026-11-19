from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FunctionCandidate:
    """Metadati di una funzione candidata presenti nel catalogo ReDataset."""

    candidate_id: str
    source_id: str
    source_path: str
    corpus_path: str
    language: str
    role: str
    benchmark_name: str
    workflow_name: str
    source_sha256: str
    status: str
    semantic_group: str
    entry_point: str
    source_task_id: str
    test_path: str


@dataclass(frozen=True, slots=True)
class TestSource:
    """Contenuto originale di un file di test, senza eseguirlo."""

    path: str
    content: str


@dataclass(frozen=True, slots=True)
class FunctionSource:
    """Codice originale e test disponibili per una funzione del corpus."""

    candidate: FunctionCandidate
    code: str
    tests: tuple[TestSource, ...]
    # Il contenuto originale di requirements rimane nel dataset, mai alterato.
    python_requirements_text: str | None = None
    python_requirements_path: str | None = None
    python_requirements_sha256: str | None = None
