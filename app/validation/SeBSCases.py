"""SeBS: interpretazione conservativa dei test originali verificabili.

Il ramo master e i workflow restano catalogati anche quando non hanno una suite
individuale riproducibile. Nessun risultato atteso viene attribuito agli autori
senza un loro generatore e validatore originali.
"""

import ast
import hashlib
import json
from pathlib import PurePosixPath

from app.redataset.Models import FunctionSource
from .HumanevalCases import PreparedCase, PreparedSuite, UnsupportedTestFormat


class SeBSCaseExtractor:
    SOURCE_IDS = frozenset({"sebs-master", "sebs-feature-workflows"})
    COMPARATOR = "sebs_original_validate_output"
    SUPPORTED_TASK = "SeBS/master/010.sleep"
    DERIVED_TASK = "SeBS/feature-workflows/610.gen/few_people"
    DERIVED_COMPARATOR = "sebs_derived_explicit_json"
    DERIVED_PATH = (
        "functions/sebs-feature-workflows/600.workflows/610.gen/"
        "tests/few_people_derived_cases.json"
    )
    DERIVED_PROVENANCE = "derived_by_spec2faas_not_original_sebs"
    # Solo questi task dispongono attualmente di suite collegate e verificate.
    ENABLED_TASKS = frozenset({SUPPORTED_TASK, DERIVED_TASK})
    DERIVED_SHA256 = "0e5000ad2f69908f66158ea929664c2b96999abdd1708c75ed2c995eab0d3df6"

    def prepare(self, source: FunctionSource) -> PreparedSuite:
        candidate = source.candidate
        if candidate.source_id not in self.SOURCE_IDS or candidate.language != "python":
            raise UnsupportedTestFormat("not_sebs_python")
        if candidate.source_task_id == self.DERIVED_TASK:
            return self._prepare_derived_few_people(source)
        if candidate.source_task_id != self.SUPPORTED_TASK:
            raise UnsupportedTestFormat("sebs_original_validator_not_integrated")
        if candidate.role != "benchmark_handler_source" or not candidate.corpus_path.endswith(
            "/000.microbenchmarks/010.sleep/python/function.py"
        ):
            raise UnsupportedTestFormat("sebs_unexpected_source_or_role")
        if len(source.tests) != 1:
            raise UnsupportedTestFormat("sebs_original_test_not_available")
        test = source.tests[0]
        if PurePosixPath(test.path).name != "input.py" or not test.path.startswith(
            "functions/sebs-master/000.microbenchmarks/010.sleep/"
        ):
            raise UnsupportedTestFormat("sebs_original_test_not_available")
        try:
            fn_tree = ast.parse(source.code)
            source_handlers = [n for n in fn_tree.body if isinstance(n, ast.FunctionDef) and n.name == "handler"]
            if len(source_handlers) != 1 or len(source_handlers[0].args.args) != 1:
                raise UnsupportedTestFormat("sebs_entry_point_not_supported")
            tree = ast.parse(test.content)
            sizes = [n.value for n in tree.body if isinstance(n, ast.Assign)
                     and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                     and n.targets[0].id == "size_generators"]
            if len(sizes) != 1:
                raise UnsupportedTestFormat("sebs_missing_size_generators")
            mapping = ast.literal_eval(sizes[0])
            duration = mapping["test"]
            if type(duration) not in (int, float) or duration <= 0 or duration > 10:
                raise UnsupportedTestFormat("sebs_invalid_test_duration")
            generators = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "generate_input"]
            validators = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validate_output"]
            if len(generators) != 1 or len(validators) != 1:
                raise UnsupportedTestFormat("sebs_missing_original_generator_or_validator")
            # Questo adattatore gestisce solo il formato originale con
            # return {'sleep': size_generators[size]}, senza eseguire input.py.
            returns = [n for n in generators[0].body if isinstance(n, ast.Return)]
            if len(generators[0].body) != 1 or len(returns) != 1:
                raise UnsupportedTestFormat("sebs_generator_requires_adapter")
            ret = returns[0].value
            if not (isinstance(ret, ast.Dict) and len(ret.keys) == 1
                    and isinstance(ret.keys[0], ast.Constant) and ret.keys[0].value == "sleep"
                    and isinstance(ret.values[0], ast.Subscript)
                    and isinstance(ret.values[0].value, ast.Name)
                    and ret.values[0].value.id == "size_generators"
                    and isinstance(ret.values[0].slice, ast.Name)
                    and ret.values[0].slice.id == "size"):
                raise UnsupportedTestFormat("sebs_generator_requires_adapter")
        except (SyntaxError, ValueError, TypeError, KeyError, IndexError) as exc:
            raise UnsupportedTestFormat("sebs_original_test_cannot_be_prepared") from exc
        event = {"sleep": duration}
        return PreparedSuite(
            candidate_id=candidate.candidate_id,
            task_id=self.SUPPORTED_TASK,
            source_sha256=candidate.source_sha256,
            test_path=test.path,
            test_sha256=hashlib.sha256(test.content.encode("utf-8")).hexdigest(),
            comparator=self.COMPARATOR,
            cases=(PreparedCase("original_test_size", (event,), event, 0.0),),
            test_paths=(test.path,),
            entry_point="handler",
            validator_code=test.content,
        )

    def _prepare_derived_few_people(self, source: FunctionSource) -> PreparedSuite:
        """Prepara casi deterministici DERIVATI, mai attribuiti a SeBS.

        La scelta del task resta esplicita: non interpretiamo file JSON
        arbitrari come test. L'integrita' del sorgente e dei test e' verificata
        rispettivamente dal repository e da PythonReferenceRunner.
        """
        candidate = source.candidate
        if (candidate.role != "workflow_handler_source" or
                not candidate.corpus_path.endswith(
                    "/600.workflows/610.gen/python/few_people.py") or
                candidate.test_path != self.DERIVED_PATH or
                len(source.tests) != 1 or
                source.tests[0].path != self.DERIVED_PATH):
            raise UnsupportedTestFormat("sebs_derived_suite_not_available")
        if hashlib.sha256(source.tests[0].content.encode("utf-8")).hexdigest() != self.DERIVED_SHA256:
            raise UnsupportedTestFormat("sebs_derived_suite_hash_changed")
        try:
            document = json.loads(source.tests[0].content)
        except (ValueError, TypeError) as exc:
            raise UnsupportedTestFormat("sebs_derived_suite_malformed") from exc
        if (not isinstance(document, dict) or document.get("schema_version") != 1 or
                document.get("provenance") != self.DERIVED_PROVENANCE or
                document.get("source_task_id") != self.DERIVED_TASK or
                not isinstance(document.get("cases"), list) or
                not document["cases"]):
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
            task_id=self.DERIVED_TASK,
            source_sha256=candidate.source_sha256,
            test_path=self.DERIVED_PATH,
            test_sha256=hashlib.sha256(source.tests[0].content.encode("utf-8")).hexdigest(),
            comparator=self.DERIVED_COMPARATOR,
            cases=tuple(cases),
            test_paths=(self.DERIVED_PATH,),
            entry_point="handler",
        )
