import csv
import hashlib
from pathlib import Path, PurePosixPath

from .Models import FunctionCandidate, FunctionSource, TestSource


class ReDatasetRepository:
    """Legge il catalogo e i sorgenti di ReDataset senza modificare i file."""

    _CATALOG_PATH = "catalog/function_candidates.csv"
    _REQUIRED_COLUMNS = {
        "function_candidate_id", "source_id", "source_path", "corpus_path",
        "language", "role", "workflow_name", "benchmark_name",
        "source_sha256", "status", "semantic_group", "entry_point",
        "source_task_id", "test_path",
    }

    def __init__(self, root: str | Path | None = None) -> None:
        # Per impostazione predefinita il dataset si trova nella root di Spec2FaaS.
        if root is None:
            root = Path(__file__).resolve().parents[2] / "ReDataset"

        self.root = Path(root).resolve()
        catalog_path = self._resolve_file(self._CATALOG_PATH)
        self._candidates = self._read_catalog(catalog_path)

    def _resolve_file(self, relative_path: str) -> Path:
        """Rifiuta percorsi assoluti o che escono dalla directory del dataset."""
        path = PurePosixPath(relative_path)
        if path.is_absolute() or not relative_path or ".." in path.parts:
            raise ValueError(f"Percorso ReDataset non valido: {relative_path!r}")

        resolved = (self.root / Path(*path.parts)).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"Percorso esterno a ReDataset: {relative_path!r}")
        if not resolved.is_file():
            raise FileNotFoundError(f"File ReDataset mancante: {relative_path}")
        return resolved

    @classmethod
    def _read_catalog(cls, path: Path) -> dict[str, FunctionCandidate]:
        """Carica una sola volta le righe del catalogo e verifica gli ID."""
        candidates: dict[str, FunctionCandidate] = {}
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            missing = cls._REQUIRED_COLUMNS - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Colonne mancanti nel catalogo: {sorted(missing)}")

            for row in reader:
                candidate = FunctionCandidate(
                    candidate_id=row["function_candidate_id"],
                    source_id=row["source_id"],
                    source_path=row["source_path"],
                    corpus_path=row["corpus_path"],
                    language=row["language"],
                    role=row["role"],
                    benchmark_name=row["benchmark_name"],
                    workflow_name=row["workflow_name"],
                    source_sha256=row["source_sha256"],
                    status=row["status"],
                    semantic_group=row["semantic_group"],
                    entry_point=row["entry_point"],
                    source_task_id=row["source_task_id"],
                    test_path=row["test_path"],
                )
                if not candidate.candidate_id:
                    raise ValueError("Il catalogo contiene un identificativo vuoto")
                if candidate.candidate_id in candidates:
                    raise ValueError(f"Identificativo duplicato: {candidate.candidate_id}")
                candidates[candidate.candidate_id] = candidate

        return candidates

    def list_candidates(
        self,
        *,
        source_id: str | None = None,
        language: str | None = None,
        role: str | None = None,
    ) -> tuple[FunctionCandidate, ...]:
        """Elenca i candidati senza scartare le varianti dello stesso workload."""
        return tuple(
            candidate
            for candidate in self._candidates.values()
            if (source_id is None or candidate.source_id == source_id)
            and (language is None or candidate.language == language)
            and (role is None or candidate.role == role)
        )

    def get_candidate(self, candidate_id: str) -> FunctionCandidate:
        """Restituisce i metadati di un candidato tramite identificativo."""
        return self._candidates[candidate_id]

    def load_source(self, candidate_id: str) -> FunctionSource:
        """Legge il codice, ne verifica l'hash e recupera i test collegati."""
        candidate = self.get_candidate(candidate_id)
        source_path = self._resolve_file(candidate.corpus_path)
        content = source_path.read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != candidate.source_sha256:
            raise ValueError(
                f"Hash del sorgente non valido per {candidate_id}: "
                f"atteso {candidate.source_sha256}, trovato {actual_hash}"
            )

        requirements = self._load_python_requirements(candidate)
        return FunctionSource(
            candidate=candidate,
            code=content.decode("utf-8"),
            tests=self._load_tests(candidate),
            python_requirements_text=requirements[0],
            python_requirements_path=requirements[1],
            python_requirements_sha256=requirements[2],
        )

    def _load_python_requirements(self, candidate: FunctionCandidate):
        """Legge il manifest originale SeBS per Python 3.11, se disponibile.

        Non lo interpreta, non installa pacchetti e non modifica il dataset.
        La selezione riflette il runtime Python 3.11 adottato dalla pipeline.
        """
        if (candidate.source_id not in ("sebs-master", "sebs-feature-workflows")
                or candidate.language != "python"):
            return None, None, None
        original = PurePosixPath(candidate.source_path)
        if original.is_absolute() or ".." in original.parts:
            raise ValueError("Percorso sorgente originale non valido")
        directory = PurePosixPath("sources") / candidate.source_id / original.parent
        for name in ("requirements.txt.3.11", "requirements.txt"):
            path = (directory / name).as_posix()
            try:
                file = self._resolve_file(path)
            except FileNotFoundError:
                continue
            data = file.read_bytes()
            return data.decode("utf-8"), path, hashlib.sha256(data).hexdigest()
        return None, None, None

    def _load_tests(self, candidate: FunctionCandidate) -> tuple[TestSource, ...]:
        """Recupera soltanto i test collegati esplicitamente o per convenzione ReFaaS."""
        paths: list[str] = []

        if candidate.test_path:
            paths.append(candidate.test_path)
        elif (candidate.source_id in ("sebs-master", "sebs-feature-workflows")
              and candidate.role == "benchmark_handler_source"
              and candidate.language == "python"):
            # Convenzione del corpus SeBS: input.py e' fratello di python/.
            # Nessun catalogo viene aggiornato; se il file non esiste,
            # l'estrattore rifiuta il task invece di generare test.
            source = PurePosixPath(candidate.corpus_path)
            if source.parent.name == "python" and source.name == "function.py":
                relative = (source.parent.parent / "input.py").as_posix()
                try:
                    self._resolve_file(relative)
                except FileNotFoundError:
                    pass
                else:
                    paths.append(relative)
        elif candidate.role == "refaas_original_function_source":
            # Nei pacchetti ReFaaS i test JSON si trovano accanto a main.py.
            parent = PurePosixPath(candidate.corpus_path).parent
            test_dir = self.root.joinpath(*parent.parts, "test").resolve()
            if not test_dir.is_relative_to(self.root):
                raise ValueError("Directory dei test esterna a ReDataset")
            if test_dir.is_dir():
                for file in sorted(test_dir.glob("*.json")):
                    paths.append(file.relative_to(self.root).as_posix())

        return tuple(
            TestSource(path=path, content=self._resolve_file(path).read_text("utf-8"))
            for path in paths
        )
