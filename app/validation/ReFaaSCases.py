import ast
import hashlib
import json
import re

from app.redataset.Models import FunctionSource
from .HumanevalCases import PreparedCase, PreparedSuite, UnsupportedTestFormat


"""Prepara i test JSON originali ReFaaS senza modificare i sorgenti."""


class ReFaaSCaseExtractor:
    """Traduce le coppie input/output JSON in invocazioni lambda_handler(event, None)."""

    SOURCE_ID = "predictive-refaas-main"
    COMPARATOR = "refaas_expected_json"
    ENTRY = "lambda_handler"

    @staticmethod
    def tests_sha256(tests) -> str:
        """Hash di *tutti* i test, con percorsi e separatori non ambigui."""
        payload = [[test.path, test.content] for test in sorted(tests, key=lambda x: x.path)]
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()

    def prepare(self, source: FunctionSource) -> PreparedSuite:
        candidate = source.candidate
        matched = re.fullmatch(r"functions/refaas-original-14/(f\d{2})/main\.py",
                               candidate.corpus_path)
        if candidate.source_id != self.SOURCE_ID or candidate.role != "refaas_original_function_source" or not matched:
            raise UnsupportedTestFormat("not_refaas_original_function")
        try:
            tree = ast.parse(source.code)
        except SyntaxError as exc:
            raise UnsupportedTestFormat("unparseable_python_source") from exc
        if not any(isinstance(node, ast.FunctionDef) and node.name == self.ENTRY
                   and len(node.args.args) == 2 for node in tree.body):
            raise UnsupportedTestFormat("unsupported_refaas_entry_point")
        if not source.tests or any(not test.path.endswith(".json") for test in source.tests):
            raise UnsupportedTestFormat("missing_refaas_json_tests")
        cases = []
        for test in sorted(source.tests, key=lambda item: item.path):
            try:
                record = json.loads(test.content)
                event = json.loads(record["input"])
                expected = json.loads(record["output"])
            except (KeyError, TypeError, ValueError) as exc:
                raise UnsupportedTestFormat("invalid_refaas_json_test") from exc
            if not isinstance(event, dict) or not isinstance(expected, dict):
                raise UnsupportedTestFormat("unsupported_refaas_test_shape")
            cases.append(PreparedCase(
                case_id=test.path.rsplit("/", 1)[-1], args=(event, None),
                expected=expected, atol=0.0,
            ))
        task_id = "ReFaaS/" + matched.group(1)
        return PreparedSuite(
            candidate_id=candidate.candidate_id,
            task_id=task_id,
            source_sha256=candidate.source_sha256,
            test_path=source.tests[0].path,
            test_sha256=self.tests_sha256(source.tests),
            comparator=self.COMPARATOR,
            cases=tuple(cases),
            test_paths=tuple(t.path for t in sorted(source.tests, key=lambda item: item.path)),
            entry_point=self.ENTRY,
        )
