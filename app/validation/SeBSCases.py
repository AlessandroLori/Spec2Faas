"""Adatta generatori e validatori ORIGINALI SeBS senza alterare ReDataset.

Interpreta un sottoinsieme dichiarativo dei generatori originali Python
che producono la configurazione ``test`` serializzabile in JSON. I generatori
che richiedono provider, file, casualita' o codice arbitrario vengono rifiutati. La fixture
storica few_people rimane separata ed esplicitamente marcata come derivata.
"""

import ast
import hashlib
import json
from .SeBSDeclarativeGenerator import SeBSDeclarativeGenerator
from pathlib import PurePosixPath

from app.redataset.Models import FunctionSource, auxiliary_files_sha256
from .HumanevalCases import PreparedCase, PreparedSuite, UnsupportedTestFormat


class SeBSCaseExtractor:
    """Recupera la configurazione test originale e conserva validate_output()."""

    SOURCE_IDS = frozenset({"sebs-master", "sebs-feature-workflows"})
    COMPARATOR = "sebs_original_validate_output"
    PROVIDER_COMPARATOR = "sebs_provider_original_validate_output"
    DERIVED_COMPARATOR = "sebs_derived_explicit_json"
    DERIVED_PROVENANCE = "derived_by_spec2faas_not_original_sebs"

    # Compatibilita' con la suite derivata few_people gia' presente nel corpus.
    # Non costituisce un modello per creare nuove fixture nel ReDataset.
    SUPPORTED_TASK = "SeBS/master/010.sleep"
    DERIVED_TASK = "SeBS/feature-workflows/610.gen/few_people"
    DERIVED_PATH = (
        "functions/sebs-feature-workflows/600.workflows/610.gen/"
        "tests/few_people_derived_cases.json"
    )
    DERIVED_SHA256 = "0e5000ad2f69908f66158ea929664c2b96999abdd1708c75ed2c995eab0d3df6"
    ENABLED_TASKS = frozenset({SUPPORTED_TASK, DERIVED_TASK})  # Solo compatibilita' API.

    def __init__(self, *, generator_bindings: dict[str, object] | None = None):
        self.generator_bindings = generator_bindings

    def _original_test_event(self, text: str) -> dict:
        """Prepara il test originale con l'interprete dati-only condiviso."""
        return SeBSDeclarativeGenerator.extract(text, bindings=self.generator_bindings)

    def prepare(self, source: FunctionSource) -> PreparedSuite:
        candidate = source.candidate
        if candidate.source_id not in self.SOURCE_IDS or candidate.language != "python":
            raise UnsupportedTestFormat("not_sebs_python")
        if candidate.source_task_id == self.DERIVED_TASK:
            return self._prepare_derived(source)
        if candidate.role != "benchmark_handler_source":
            raise UnsupportedTestFormat("sebs_original_validator_not_available")
        path = PurePosixPath(candidate.corpus_path)
        expected_test_path = (path.parent.parent / "input.py").as_posix()
        if (path.name != "function.py" or path.parent.name != "python"
                or len(source.tests) != 1 or source.tests[0].path != expected_test_path):
            raise UnsupportedTestFormat("sebs_original_test_not_available")
        try:
            tree = ast.parse(source.code)
        except SyntaxError as exc:
            raise UnsupportedTestFormat("sebs_python_source_unparseable") from exc
        handlers = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "handler"]
        if (len(handlers) != 1 or len(handlers[0].args.args) != 1
                or handlers[0].args.vararg is not None
                or handlers[0].args.kwarg is not None):
            raise UnsupportedTestFormat("sebs_entry_point_not_supported")
        test = source.tests[0]
        event = self._original_test_event(test.content)
        return PreparedSuite(
            candidate_id=candidate.candidate_id,
            task_id=candidate.source_task_id,
            source_sha256=candidate.source_sha256,
            test_path=test.path,
            test_sha256=hashlib.sha256(test.content.encode("utf-8")).hexdigest(),
            comparator=self.COMPARATOR,
            cases=(PreparedCase("original_test_size", (event,), None, 0.0),),
            test_paths=(test.path,),
            entry_point="handler",
            validator_code=test.content,
            # Il Go tradotto viene confrontato con i risultati attesi per la
            # versione Python originale, non con una tabella Go inesistente.
            validator_reference_language="python",
            auxiliary_sha256=(auxiliary_files_sha256(source.auxiliary_files)
                              if source.auxiliary_files else None),
        )

    def _prepare_derived(self, source: FunctionSource) -> PreparedSuite:
        """Mantiene solo la fixture storica few_people, attestata da SHA-256."""
        candidate = source.candidate
        if (candidate.role != "workflow_handler_source" or
                not candidate.corpus_path.endswith("/600.workflows/610.gen/python/few_people.py") or
                candidate.test_path != self.DERIVED_PATH or
                len(source.tests) != 1 or source.tests[0].path != self.DERIVED_PATH):
            raise UnsupportedTestFormat("sebs_derived_suite_not_available")
        test = source.tests[0]
        if hashlib.sha256(test.content.encode("utf-8")).hexdigest() != self.DERIVED_SHA256:
            raise UnsupportedTestFormat("sebs_derived_suite_hash_changed")
        try:
            document = json.loads(test.content)
        except (ValueError, TypeError) as exc:
            raise UnsupportedTestFormat("sebs_derived_suite_malformed") from exc
        if (not isinstance(document, dict) or document.get("schema_version") != 1 or
                document.get("provenance") != self.DERIVED_PROVENANCE or
                document.get("source_task_id") != candidate.source_task_id or
                not isinstance(document.get("cases"), list) or not document["cases"]):
            raise UnsupportedTestFormat("sebs_derived_suite_malformed")
        seen = set()
        cases = []
        for item in document["cases"]:
            if not isinstance(item, dict) or set(item) != {"case_id", "event", "expected"}:
                raise UnsupportedTestFormat("sebs_derived_case_invalid")
            case_id, event, expected = item["case_id"], item["event"], item["expected"]
            if (not isinstance(case_id, str) or not case_id or case_id in seen or
                    type(event) is not dict or type(expected) is not dict or
                    not all(type(key) is str for key in event) or
                    not all(type(key) is str for key in expected)):
                raise UnsupportedTestFormat("sebs_derived_case_invalid")
            seen.add(case_id)
            cases.append(PreparedCase(case_id, (event,), expected, 0.0))
        return PreparedSuite(
            candidate_id=candidate.candidate_id,
            task_id=candidate.source_task_id,
            source_sha256=candidate.source_sha256,
            test_path=self.DERIVED_PATH,
            test_sha256=hashlib.sha256(test.content.encode("utf-8")).hexdigest(),
            comparator=self.DERIVED_COMPARATOR,
            cases=tuple(cases),
            test_paths=(self.DERIVED_PATH,),
            entry_point="handler",
        )
