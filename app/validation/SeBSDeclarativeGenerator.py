"""Interpreta il sottoinsieme dichiarativo dei generatori originali SeBS.

Non usa ``eval``/``exec`` e non invoca funzioni di SeBS: i callback verso
provider, le letture di asset e le chiamate Python arbitrarie vengono rifiutati.
Serve ad estrarre l'evento originale ``size='test'`` senza inventare fixture.
"""

import ast
import copy
import json
import operator

from .HumanevalCases import UnsupportedTestFormat


class SeBSDeclarativeGenerator:
    """Valuta assegnazioni, contenitori e controllo di flusso dati-only da AST.

    Ogni nodo/costrutto non esplicitamente ammesso viene rifiutato. I limiti
    proteggono da generatori ciclici o da un'esplosione di dati accidentale.
    """

    MAX_STEPS = 2000
    MAX_ITERATIONS = 200
    MAX_JSON_BYTES = 100_000

    _BINARY = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
    }
    _COMPARE = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b,
    }

    def __init__(self) -> None:
        self._steps = 0
        self._returned = False
        self._value = None
        self._env: dict[str, object] = {}

    def _tick(self) -> None:
        self._steps += 1
        if self._steps > self.MAX_STEPS:
            raise UnsupportedTestFormat("sebs_generator_exceeds_declarative_limits")

    @staticmethod
    def _reject() -> None:
        raise UnsupportedTestFormat("sebs_generator_requires_adapter")

    def _expr(self, node: ast.AST):
        self._tick()
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in self._env:
                self._reject()
            return self._env[node.id]
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                self._reject()
            return {self._expr(k): self._expr(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [self._expr(item) for item in node.elts]
            return tuple(values) if isinstance(node, ast.Tuple) else values
        if isinstance(node, ast.Subscript):
            container, key = self._expr(node.value), self._expr(node.slice)
            if type(container) not in (dict, list, tuple) or type(key) not in (str, int):
                self._reject()
            try:
                return container[key]
            except (KeyError, IndexError, TypeError):
                self._reject()
        if isinstance(node, ast.UnaryOp):
            value = self._expr(node.operand)
            if isinstance(node.op, ast.Not):
                return not value
            if type(value) in (int, float) and isinstance(node.op, (ast.UAdd, ast.USub)):
                return +value if isinstance(node.op, ast.UAdd) else -value
            self._reject()
        if isinstance(node, ast.BinOp) and type(node.op) in self._BINARY:
            left, right = self._expr(node.left), self._expr(node.right)
            if (type(left) not in (int, float, str, list, tuple)
                    or type(right) not in (int, float, str, list, tuple)):
                self._reject()
            if type(left) in (str, list, tuple) or type(right) in (str, list, tuple):
                if not isinstance(node.op, ast.Add) or type(left) is not type(right):
                    self._reject()
                if len(left) + len(right) > self.MAX_ITERATIONS:
                    raise UnsupportedTestFormat("sebs_generator_exceeds_declarative_limits")
            try:
                result = self._BINARY[type(node.op)](left, right)
            except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                self._reject()
            if type(result) not in (int, float, str, list, tuple) or (isinstance(result, str) and len(result) > self.MAX_JSON_BYTES):
                self._reject()
            return result
        if isinstance(node, ast.Compare):
            left = self._expr(node.left)
            for op, cmp_node in zip(node.ops, node.comparators):
                if type(op) not in self._COMPARE:
                    self._reject()
                right = self._expr(cmp_node)
                try:
                    ok = self._COMPARE[type(op)](left, right)
                except (TypeError, ValueError):
                    self._reject()
                if not ok:
                    return False
                left = right
            return True
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            value = self._expr(node.values[0])
            for element in node.values[1:]:
                if (isinstance(node.op, ast.And) and not value) or (isinstance(node.op, ast.Or) and value):
                    break
                value = self._expr(element)
            return value
        if isinstance(node, ast.IfExp):
            return self._expr(node.body if self._expr(node.test) else node.orelse)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            # Solo costruttori inerti; mai callback o funzioni definite nel file SeBS.
            name = node.func.id
            if name in ("dict", "list", "tuple") and len(node.args) == 0:
                return {"dict": dict, "list": list, "tuple": tuple}[name]()
            if name == "range" and 1 <= len(node.args) <= 3:
                arguments = [self._expr(arg) for arg in node.args]
                if any(type(arg) is not int for arg in arguments):
                    self._reject()
                try:
                    indices = range(*arguments)
                except ValueError:
                    self._reject()
                if len(indices) > self.MAX_ITERATIONS:
                    raise UnsupportedTestFormat("sebs_generator_exceeds_declarative_limits")
                return indices
        self._reject()

    def _assign(self, target: ast.AST, value: object) -> None:
        self._tick()
        if isinstance(target, ast.Name):
            self._env[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)) and type(value) in (tuple, list):
            if len(target.elts) != len(value):
                self._reject()
            for elt, item in zip(target.elts, value):
                self._assign(elt, item)
            return
        if isinstance(target, ast.Subscript):
            container, key = self._expr(target.value), self._expr(target.slice)
            if type(container) is dict and type(key) in (str, int):
                container[key] = value
                return
            if type(container) is list and type(key) is int and -len(container) <= key < len(container):
                container[key] = value
                return
        self._reject()

    def _statements(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            if self._returned:
                return
            self._tick()
            if isinstance(statement, ast.Return):
                self._value = self._expr(statement.value) if statement.value else None
                self._returned = True
            elif isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                self._assign(statement.targets[0], self._expr(statement.value))
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                self._assign(statement.target, self._expr(statement.value))
            elif isinstance(statement, ast.If):
                self._statements(statement.body if self._expr(statement.test) else statement.orelse)
            elif isinstance(statement, ast.For) and not statement.orelse:
                iterable = self._expr(statement.iter)
                if type(iterable) not in (list, tuple, range) or len(iterable) > self.MAX_ITERATIONS:
                    self._reject()
                for value in iterable:
                    self._assign(statement.target, value)
                    self._statements(statement.body)
                    if self._returned:
                        break
            elif (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                  and isinstance(statement.value.func, ast.Attribute)
                  and not statement.value.keywords):
                # Soltanto metodi mutatori di contenitori in memoria, su dati
                # gia' ricavati dal programma dichiarativo.
                call = statement.value
                receiver = self._expr(call.func.value)
                if (type(receiver) is list and call.func.attr == "append"
                        and len(call.args) == 1 and len(receiver) < self.MAX_ITERATIONS):
                    receiver.append(self._expr(call.args[0]))
                elif (type(receiver) is dict and call.func.attr == "update"
                      and len(call.args) == 1):
                    value = self._expr(call.args[0])
                    if type(value) is not dict or len(receiver) + len(value) > self.MAX_ITERATIONS:
                        self._reject()
                    receiver.update(value)
                else:
                    self._reject()
            elif isinstance(statement, ast.Pass):
                continue
            else:
                # Qualsiasi chiamata a upload_func/nosql_func, open, os.*, ecc.
                # non viene eseguita, neppure per caso.
                self._reject()

    @classmethod
    def extract(cls, text: str) -> dict:
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            raise UnsupportedTestFormat("sebs_original_test_cannot_be_prepared") from exc
        generators = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and node.name == "generate_input"]
        validators = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "validate_output"]
        if len(generators) != 1 or len(validators) != 1:
            raise UnsupportedTestFormat("sebs_missing_original_generator_or_validator")
        generator = generators[0]
        if not isinstance(generator, ast.FunctionDef):
            raise UnsupportedTestFormat("sebs_generator_requires_adapter")

        evaluator = cls()
        evaluator._env["size"] = "test"
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    evaluator._env[node.targets[0].id] = copy.deepcopy(ast.literal_eval(node.value))
                except (TypeError, ValueError, SyntaxError, RecursionError):
                    continue
        evaluator._statements(generator.body)
        event = evaluator._value
        if not evaluator._returned or type(event) is not dict or any(type(k) is not str for k in event):
            raise UnsupportedTestFormat("sebs_original_generator_non_json_event")
        try:
            encoded = json.dumps(event, allow_nan=False)
            if len(encoded.encode("utf-8")) > cls.MAX_JSON_BYTES or json.loads(encoded) != event:
                raise UnsupportedTestFormat("sebs_original_generator_non_json_event")
        except (ValueError, TypeError, OverflowError, RecursionError) as exc:
            raise UnsupportedTestFormat("sebs_original_generator_non_json_event") from exc
        return event
