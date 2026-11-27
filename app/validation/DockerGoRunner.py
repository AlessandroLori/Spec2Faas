import hashlib
import json
import math
import os
import re
import selectors
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from app.providers.ProviderGoSdk import ProviderGoSdk
from .ExecutionModels import CaseExecution, ExecutionResult
from .HumanevalCases import PreparedSuite


class DockerGoRunner:
    """Compila ed esegue Go in container senza accesso alla rete o all'host."""

    _IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z", re.ASCII)
    _ENTRY_MARKER = "__REDATASET_ENTRY_POINT__"

    def __init__(
        self,
        *,
        enabled: bool = False,
        build_image: str = "golang:1.25-alpine",
        runtime_image: str = "alpine:3.22",
        docker_executable: str = "docker",
        compile_timeout_seconds: float = 180.0,
        execution_timeout_seconds: float = 60.0,
        max_output_bytes: int = 32 * 1024 * 1024,
        go_modules_enabled: bool = False, go_cgo_enabled: bool = False,
        provider_runtime=None,
    ) -> None:
        if any(not math.isfinite(t) or t <= 0 for t in (
            compile_timeout_seconds, execution_timeout_seconds,
        )):
            raise ValueError("I timeout devono essere positivi e finiti")
        if type(max_output_bytes) is not int or max_output_bytes <= 0:
            raise ValueError("Il limite di output deve essere positivo")
        if not build_image or not runtime_image or not docker_executable:
            raise ValueError("Immagini e comando Docker devono essere specificati")
        self.enabled = enabled
        self.build_image = build_image
        self.runtime_image = runtime_image
        self.docker_executable = docker_executable
        self.compile_timeout_seconds = compile_timeout_seconds
        self.execution_timeout_seconds = execution_timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.go_modules_enabled = go_modules_enabled
        self.go_cgo_enabled = go_cgo_enabled
        self.provider_runtime = provider_runtime
        self._harness_path = Path(__file__).with_name("GoHarness.go")

    def _container_command(
        self, *, name: str, directory: Path, image: str,
        writable: bool, program: list[str],
    ) -> list[str]:
        """Applica gli stessi limiti alla compilazione e all'esecuzione."""
        mount = f"type=bind,src={directory},dst=/work"
        if not writable:
            mount += ",readonly"
        command = [
            self.docker_executable, "run", "--rm", "--pull=never",
            "--name", name,
            *(["--network=none"] if writable or self.provider_runtime is None
              else self.provider_runtime.runtime_docker_args()),
            "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--pids-limit=64", "--memory=1g", "--memory-swap=1g",
            "--cpus=1", "--user", f"{os.getuid()}:{os.getgid()}",
            "--mount", mount, "--workdir", "/work", "--interactive",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=512m" if writable
            else "/tmp:rw,nosuid,nodev,size=64m",
        ]
        if writable:
            command += [
                "--env", "GOCACHE=/tmp/gocache",
                "--env", "GOPATH=/tmp/gopath", "--env", "HOME=/tmp",
                "--env", "GOTOOLCHAIN=local", "--env", "GOWORK=off",
                "--env", "GO111MODULE=on" if self.go_modules_enabled else "GO111MODULE=off",
                "--env", "GOPROXY=off", "--env", "GOSUMDB=off",
                "--env", "CGO_ENABLED=1" if self.go_cgo_enabled else "CGO_ENABLED=0",
            ]
        return command + [image, *program]

    def _cleanup(self, name: str, process: subprocess.Popen) -> None:
        """Ferma anche il container quando scade il timeout del client Docker."""
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
        try:
            subprocess.run(
                [self.docker_executable, "rm", "-f", name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _call_container(
        self, command: list[str], *, name: str, timeout: float,
        payload: bytes = b"",
    ) -> tuple[str, int | None, bytes, bytes]:
        """Legge output limitato; non permette stdout illimitato sul processo host."""
        with tempfile.TemporaryFile() as input_file:
            input_file.write(payload)
            input_file.seek(0)
            try:
                process = subprocess.Popen(
                    command, stdin=input_file, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except OSError:
                return "tool_unavailable", None, b"", b""

            out, err = bytearray(), bytearray()
            deadline = time.monotonic() + timeout
            with selectors.DefaultSelector() as selector:
                assert process.stdout is not None and process.stderr is not None
                selector.register(process.stdout, selectors.EVENT_READ, out)
                selector.register(process.stderr, selectors.EVENT_READ, err)
                status = "ok"
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        status = "timeout"
                        break
                    for key, _ in selector.select(timeout=min(remaining, 0.2)):
                        data = os.read(key.fileobj.fileno(), 65536)
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        target = key.data
                        target.extend(data)
                        if len(out) + len(err) > self.max_output_bytes:
                            status = "output_limit"
                            break
                    if status != "ok":
                        break

            if status != "ok":
                self._cleanup(name, process)
                process.stdout.close()
                process.stderr.close()
                return status, None, bytes(out[:1600]), bytes(err[:1600])
            try:
                return "ok", process.wait(timeout=max(0.1, deadline - time.monotonic())), bytes(out), bytes(err)
            except subprocess.TimeoutExpired:
                self._cleanup(name, process)
                return "timeout", None, bytes(out[:1600]), bytes(err[:1600])
            finally:
                if process.stdout:
                    process.stdout.close()
                if process.stderr:
                    process.stderr.close()

    def preflight(self) -> str | None:
        """Controlla Docker e le immagini locali prima di spendere token LLM."""
        if not self.enabled:
            return "DockerGoRunner non è stato abilitato"
        if os.getuid() == 0:
            return "È richiesto un utente host non root"
        if shutil.which(self.docker_executable) is None:
            return "Docker non è installato o non è nel PATH"
        sdk = getattr(self.provider_runtime, "go_sdk", None)
        if not isinstance(sdk, ProviderGoSdk):
            sdk = None
        if sdk is not None:
            try:
                sdk.ensure_image()
            except Exception as exc:
                return f"Preparazione SDK provider Go fallita: {exc}"
            self.build_image = sdk.image
            self.go_modules_enabled = True
        for image in (self.build_image, self.runtime_image):
            try:
                checked = subprocess.run(
                    [self.docker_executable, "image", "inspect", image],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=10, check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                return "Impossibile contattare Docker"
            if checked.returncode != 0:
                return f"Immagine Docker non disponibile localmente: {image}"
        return None

    def run(self, code: str, *, entry_point: str, suite: PreparedSuite) -> ExecutionResult:
        """Compila una volta, poi verifica tutti i casi nel container runtime."""
        if not self.enabled:
            raise RuntimeError("DockerGoRunner disabilitato: abilitarlo esplicitamente")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Il codice Go non può essere vuoto")
        if not isinstance(entry_point, str) or not self._IDENTIFIER.fullmatch(entry_point) or entry_point == "main":
            raise ValueError("Entry point Go non valido")
        if not suite.cases or len({c.case_id for c in suite.cases}) != len(suite.cases):
            raise ValueError("La suite è vuota o contiene ID duplicati")

        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()

        def outcome(status, *, cases=(), compile_seconds=0.0, execution_seconds=0.0, error_message=None):
            return ExecutionResult(
                candidate_id=suite.candidate_id, test_sha256=suite.test_sha256,
                language="go", code_sha256=code_hash, status=status,
                cases=cases, compile_seconds=compile_seconds,
                execution_seconds=execution_seconds, error_message=error_message,
            )

        if os.getuid() == 0:
            raise RuntimeError("Eseguire DockerGoRunner da un utente host non root")
        if shutil.which(self.docker_executable) is None:
            return outcome("tool_unavailable", error_message="Docker non trovato")
        try:
            requests = "".join(
                json.dumps({"index": i, "args": c.args}, ensure_ascii=False, allow_nan=False) + "\n"
                for i, c in enumerate(suite.cases)
            ).encode("utf-8")
        except (ValueError, TypeError) as exc:
            raise ValueError("Input dei test non rappresentabili in JSON") from exc

        sdk = getattr(self.provider_runtime, "go_sdk", None)
        if not isinstance(sdk, ProviderGoSdk):
            sdk = None
        if sdk is not None and (not self.go_modules_enabled or self.build_image != sdk.image):
            problem = self.preflight()
            if problem is not None:
                return outcome("tool_unavailable", error_message=problem)
        harness = self._harness_path.read_text(encoding="utf-8")
        harness = harness.replace(self._ENTRY_MARKER, entry_point)
        with tempfile.TemporaryDirectory(prefix="redataset_go_docker_") as workdir:
            directory = Path(workdir)
            (directory / "function.go").write_text(code, encoding="utf-8")
            (directory / "harness.go").write_text(harness, encoding="utf-8")
            if sdk is not None:
                (directory / "provider_s3.go").write_bytes(sdk.helper.read_bytes())

            build_name = "redataset-build-" + uuid.uuid4().hex
            compile_cmd = self._container_command(
                name=build_name, directory=directory, image=self.build_image,
                writable=True, program=(
                    ["go", "-C", "/opt/spec2faas", "build", "-mod=vendor", "-trimpath",
                     "-o", "/work/function_runner", "/work/function.go", "/work/harness.go",
                     *(["/work/provider_s3.go"] if sdk is not None else [])]
                    if self.go_modules_enabled else
                    ["go", "build", "-trimpath", "-o", "/work/function_runner",
                     "function.go", "harness.go"]),
            )
            start = time.perf_counter()
            status, exit_code, stdout, stderr = self._call_container(
                compile_cmd, name=build_name, timeout=self.compile_timeout_seconds,
            )
            compile_seconds = time.perf_counter() - start
            if status != "ok":
                mapping = {"timeout": "compile_timeout", "output_limit": "protocol_error"}
                return outcome(mapping.get(status, status), compile_seconds=compile_seconds,
                               error_message="Limite output del container superato" if status == "output_limit" else None)
            if exit_code != 0:
                return outcome(
                    "sandbox_error" if exit_code == 125 else "compile_error",
                    compile_seconds=compile_seconds,
                    error_message=(stderr or stdout).decode("utf-8", "replace")[:1600],
                )
            if not (directory / "function_runner").is_file():
                return outcome("sandbox_error", compile_seconds=compile_seconds,
                               error_message="Il binario non è stato prodotto")

            # Ogni nuova proposta del Debugger deve vedere identico stato S3,
            # non quello lasciato da Python o dal precedente candidato Go.
            if self.provider_runtime is not None:
                self.provider_runtime.before_go_execution()
            run_name = "redataset-run-" + uuid.uuid4().hex
            execute_cmd = self._container_command(
                name=run_name, directory=directory, image=self.runtime_image,
                writable=False, program=["/work/function_runner"],
            )
            start = time.perf_counter()
            status, exit_code, stdout, stderr = self._call_container(
                execute_cmd, name=run_name, timeout=self.execution_timeout_seconds,
                payload=requests,
            )
            execution_seconds = time.perf_counter() - start
            if status != "ok":
                mapping = {"timeout": "execution_timeout", "output_limit": "protocol_error"}
                return outcome(mapping.get(status, status), compile_seconds=compile_seconds,
                               execution_seconds=execution_seconds,
                               error_message="Limite output del container superato" if status == "output_limit" else None)
            if exit_code != 0:
                return outcome(
                    "sandbox_error" if exit_code == 125 else "runtime_error",
                    compile_seconds=compile_seconds, execution_seconds=execution_seconds,
                    error_message=f"Exit code {exit_code}: {stderr.decode('utf-8', 'replace')[:600]}",
                )
            try:
                rows = [json.loads(line) for line in stdout.decode("utf-8").splitlines()]
            except (UnicodeError, ValueError):
                return outcome("protocol_error", compile_seconds=compile_seconds,
                               execution_seconds=execution_seconds,
                               error_message="Output non in formato JSON lines")
            if len(rows) != len(suite.cases) or any(
                not isinstance(row, dict) or type(row.get("index")) is not int
                or row["index"] != idx or "value" not in row
                or not isinstance(row.get("error", ""), str)
                for idx, row in enumerate(rows)
            ):
                return outcome("protocol_error", compile_seconds=compile_seconds,
                               execution_seconds=execution_seconds,
                               error_message="Numero, ordine o struttura degli output non valido")
            cases = tuple(
                CaseExecution(case_id=case.case_id,
                              status="error" if row.get("error") else "returned",
                              value=row["value"], error_message=row.get("error") or None)
                for case, row in zip(suite.cases, rows)
            )
            return outcome("completed", cases=cases, compile_seconds=compile_seconds,
                           execution_seconds=execution_seconds)
