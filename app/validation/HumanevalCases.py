import ast
import hashlib
import math
from dataclasses import dataclass
from typing import Any

from app.redataset.Models import FunctionSource


class UnsupportedTestFormat(ValueError):
    """Indica che i test richiedono una preparazione non ancora supportata."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class PreparedCase:
    """Input posizionali, output atteso e tolleranza del test originale."""

    case_id: str
    args: tuple[Any, ...]
    expected: Any
    atol: float


@dataclass(frozen=True, slots=True)
class PreparedSuite:
    """Casi estratti, con gli hash necessari per risalire agli originali."""

    candidate_id: str
    task_id: str
    source_sha256: str
    test_path: str
    test_sha256: str
    comparator: str
    cases: tuple[PreparedCase, ...]
    # HumanEval usa un file test; ReFaaS usa più coppie JSON per funzione.
    test_paths: tuple[str, ...] = ()
    entry_point: str | None = None
    # Solo per i test originali SeBS: codice del validatore attestato dall'hash.
    validator_code: str | None = None
    # Per un porting Python->Go si usano le aspettative dell'origine Python.
    validator_reference_language: str | None = None
    auxiliary_sha256: str | None = None


def original_python_test_source(source: FunctionSource, suite: PreparedSuite) -> str | None:
    """Seleziona codice dei test Python originali usati dalla suite, senza eseguirlo.

    Serve solo per scoprire dipendenze statiche della baseline Docker; i test
    rimangono invariati in ReDataset e continuano a essere verificati per hash.
    """
    paths = set(getattr(suite, "test_paths", ()) or (getattr(suite, "test_path", None),))
    blocks = [test.content for test in getattr(source, "tests", ())
              if test.path in paths and test.path.endswith(".py")]
    return "\n\n".join(blocks) if blocks else None


class HumanEvalCaseExtractor:
    """Estrae test HumanEval+ con risultati letterali senza eseguire Python."""

    SOURCE_ID = "spec2faas-humanevalplus-164"
    COMPARATOR = "humaneval_plus_assertion"

    def prepare(self, source: FunctionSource) -> PreparedSuite:
        """Accetta solo il formato input/output esplicito verificato nel corpus."""
        candidate = source.candidate
        if candidate.source_id != self.SOURCE_ID:
            raise ValueError("L'estrattore accetta soltanto sorgenti HumanEval+")
        if len(source.tests) != 1 or not source.tests[0].path.endswith("/tests.py"):
            raise UnsupportedTestFormat("missing_test_file")

        test_file = source.tests[0]
        tree = ast.parse(test_file.content, filename=test_file.path)
        checks = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "check"
        ]
        if len(checks) != 1:
            raise UnsupportedTestFormat("unsupported_check_function")

        check = checks[0]
        assignments: dict[str, Any] = {}
        for statement in check.body:
            if not isinstance(statement, ast.Assign):
                continue
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id in ("inputs", "results"):
                    try:
                        assignments[target.id] = ast.literal_eval(statement.value)
                    except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
                        raise UnsupportedTestFormat("nonliteral_test_data") from exc

        if "results" not in assignments:
            raise UnsupportedTestFormat("reference_generated_expected")
        if "inputs" not in assignments:
            raise UnsupportedTestFormat("missing_inputs")

        inputs, results = assignments["inputs"], assignments["results"]
        if not isinstance(inputs, list) or not isinstance(results, list):
            raise UnsupportedTestFormat("unsupported_test_data_type")
        if len(inputs) != len(results) or not inputs:
            raise UnsupportedTestFormat("invalid_test_data_length")

        atol = self._read_comparator(check)
        if not self._json_value(inputs) or not self._json_value(results):
            raise UnsupportedTestFormat("non_json_value")
        if any(not isinstance(args, list) for args in inputs):
            raise UnsupportedTestFormat("unsupported_argument_structure")

        cases = tuple(
            PreparedCase(
                case_id=f"case_{index:06d}",
                args=tuple(args),
                expected=expected,
                atol=atol,
            )
            for index, (args, expected) in enumerate(zip(inputs, results), start=1)
        )
        return PreparedSuite(
            candidate_id=candidate.candidate_id,
            task_id=candidate.source_task_id,
            source_sha256=candidate.source_sha256,
            test_path=test_file.path,
            test_sha256=hashlib.sha256(test_file.content.encode("utf-8")).hexdigest(),
            comparator=self.COMPARATOR,
            cases=cases,
        )

    @staticmethod
    def _read_comparator(check: ast.FunctionDef) -> float:
        """Verifica che il ciclo usi esattamente assertion(candidate(*inp), exp, atol)."""
        loops = [node for node in check.body if isinstance(node, ast.For)]
        if len(loops) != 1:
            raise UnsupportedTestFormat("unsupported_assertion")

        loop = loops[0]
        if len(loop.body) != 1 or not isinstance(loop.body[0], ast.Expr):
            raise UnsupportedTestFormat("unsupported_assertion")
        expression = loop.body[0].value
        if not isinstance(expression, ast.Call) or len(expression.args) != 3:
            raise UnsupportedTestFormat("unsupported_assertion")
        if not isinstance(expression.args[2], ast.Constant):
            raise UnsupportedTestFormat("unsupported_assertion")
        atol = expression.args[2].value
        if type(atol) not in (int, float) or atol < 0 or not math.isfinite(atol):
            raise UnsupportedTestFormat("unsupported_assertion")

        # Il confronto strutturale evita di interpretare cicli di test arbitrari.
        template = ast.parse(
            "for i, (inp, exp) in enumerate(zip(inputs, results)):\n"
            "    assertion(candidate(*inp), exp, 0)\n"
        ).body[0]
        template.body[0].value.args[2].value = atol
        if ast.dump(loop) != ast.dump(template):
            raise UnsupportedTestFormat("unsupported_assertion")
        return float(atol)

    @classmethod
    def _json_value(cls, value: Any) -> bool:
        """Rifiuta strutture Python che JSON trasformerebbe o perderebbe."""
        if value is None or type(value) in (str, int, bool):
            return True
        if type(value) is float:
            return math.isfinite(value)
        if type(value) is list:
            return all(cls._json_value(item) for item in value)
        if type(value) is dict:
            return all(
                type(key) is str and cls._json_value(item)
                for key, item in value.items()
            )
        return False
