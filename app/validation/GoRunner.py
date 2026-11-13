import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from .ExecutionModels import CaseExecution, ExecutionResult
from .HumanevalCases import PreparedSuite


class GoRunner:
    """Compila una funzione Go e la esegue sui casi già preparati."""

    _IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z", re.ASCII)
    _ENTRY_MARKER = "__REDATASET_ENTRY_POINT__"

    def __init__(
        self,
        *,
        allow_host_execution: bool = False,
        compile_timeout_seconds: float = 90.0,
        execution_timeout_seconds: float = 60.0,
        go_executable: str = "go",
    ) -> None:
        if any(
            not math.isfinite(value) or value <= 0
            for value in (compile_timeout_seconds, execution_timeout_seconds)
        ):
            raise ValueError("I timeout devono essere positivi e finiti")
        self.allow_host_execution = allow_host_execution
        self.compile_timeout_seconds = compile_timeout_seconds
        self.execution_timeout_seconds = execution_timeout_seconds
        self.go_executable = go_executable
        self._harness_path = Path(__file__).with_name("GoHarness.go")

    def run(self, code: str, *, entry_point: str, suite: PreparedSuite) -> ExecutionResult:
        """Restituisce gli output Go"""
        if not self.allow_host_execution:
            raise RuntimeError(
                "Esecuzione Go sull'host disabilitata: richiede consenso esplicito "
                "e codice controllato. Le traduzioni LLM richiederanno un sandbox."
            )
        if not self._IDENTIFIER.fullmatch(entry_point) or entry_point == "main":
            raise ValueError("Entry point Go non valido o riservato")
        if not suite.cases:
            raise ValueError("La suite deve contenere almeno un caso")
        ids = [case.case_id for case in suite.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("La suite contiene identificativi di caso duplicati")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Il codice Go non può essere vuoto")

        code_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()

        def outcome(
            status: str, *, cases: tuple[CaseExecution, ...] = (),
            compile_seconds: float = 0.0, execution_seconds: float = 0.0,
            error_message: str | None = None,
        ) -> ExecutionResult:
            return ExecutionResult(
                candidate_id=suite.candidate_id,
                test_sha256=suite.test_sha256,
                language="go",
                code_sha256=code_sha256,
                status=status,
                cases=cases,
                compile_seconds=compile_seconds,
                execution_seconds=execution_seconds,
                error_message=error_message,
            )

        harness = self._harness_path.read_text(encoding="utf-8")
        harness = harness.replace(self._ENTRY_MARKER, entry_point)
        # Le dipendenze esterne non sono ancora supportate: Go resta offline.
        env = os.environ.copy()
        env.update({
            "GOTOOLCHAIN": "local", "GOWORK": "off", "GO111MODULE": "off",
            "GOPROXY": "off", "GOSUMDB": "off", "CGO_ENABLED": "0",
            "GOFLAGS": "",
        })

        with tempfile.TemporaryDirectory(prefix="redataset_go_") as workdir:
            directory = Path(workdir)
            (directory / "function.go").write_text(code, encoding="utf-8")
            (directory / "harness.go").write_text(harness, encoding="utf-8")
            binary = directory / "function_runner"
            start = time.perf_counter()
            try:
                build = subprocess.run(
                    [self.go_executable, "build", "-trimpath", "-o", str(binary),
                     "function.go", "harness.go"],
                    cwd=directory, env=env, capture_output=True, text=True,
                    timeout=self.compile_timeout_seconds, check=False,
                )
            except FileNotFoundError:
                return outcome("tool_unavailable", error_message="Compilatore Go non trovato")
            except subprocess.TimeoutExpired:
                return outcome("compile_timeout", compile_seconds=time.perf_counter() - start)
            compile_seconds = time.perf_counter() - start
            if build.returncode != 0:
                return outcome(
                    "compile_error", compile_seconds=compile_seconds,
                    error_message=(build.stderr or build.stdout)[:1600],
                )

            # Ogni richiesta contiene solo gli input, mai gli output attesi.
            try:
                requests = "".join(
                    json.dumps({"index": index, "args": case.args},
                               ensure_ascii=False, allow_nan=False) + "\n"
                    for index, case in enumerate(suite.cases)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("La suite contiene input non rappresentabili in JSON") from exc

            start = time.perf_counter()
            try:
                executed = subprocess.run(
                    [str(binary)], input=requests,
                    cwd=directory, capture_output=True, text=True,
                    timeout=self.execution_timeout_seconds, check=False,
                )
            except subprocess.TimeoutExpired:
                return outcome(
                    "execution_timeout", compile_seconds=compile_seconds,
                    execution_seconds=time.perf_counter() - start,
                )
            execution_seconds = time.perf_counter() - start
            if executed.returncode != 0:
                return outcome(
                    "runtime_error", compile_seconds=compile_seconds,
                    execution_seconds=execution_seconds,
                    error_message=f"Codice di uscita {executed.returncode}: "
                                  f"{executed.stderr[:600]}",
                )
            try:
                rows = [json.loads(line) for line in executed.stdout.splitlines()]
            except (TypeError, ValueError):
                return outcome("protocol_error", compile_seconds=compile_seconds,
                               execution_seconds=execution_seconds,
                               error_message="Il programma Go non ha restituito JSON valido")
            if len(rows) != len(suite.cases) or any(
                not isinstance(row, dict)
                or type(row.get("index")) is not int
                or row["index"] != index
                or "value" not in row
                or not isinstance(row.get("error", ""), str)
                for index, row in enumerate(rows)
            ):
                return outcome("protocol_error", compile_seconds=compile_seconds,
                               execution_seconds=execution_seconds,
                               error_message="Numero, ordine o formato degli output non valido")

            cases = tuple(
                CaseExecution(
                    case_id=case.case_id,
                    status="error" if row.get("error") else "returned",
                    value=row["value"],
                    error_message=row.get("error") or None,
                )
                for case, row in zip(suite.cases, rows)
            )
            return outcome("completed", cases=cases, compile_seconds=compile_seconds,
                           execution_seconds=execution_seconds)
