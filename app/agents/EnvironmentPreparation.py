"""Preparazione Docker riproducibile, senza comandi host proposti da un LLM."""

import ast
import hashlib
import json
import re
import os
import sys
import subprocess
from pathlib import Path


class EnvironmentBuildError(RuntimeError):
    """Un ambiente non e' stato costruito; il sorgente non viene modificato."""

    def __init__(self, message: str, *, log_path: Path,
                 failure_kind: str = "unclassified") -> None:
        super().__init__(message)
        self.log_path = log_path
        self.failure_kind = failure_kind


class EnvironmentPreparationExhausted(RuntimeError):
    """Preparazione esaurita: distingue errori nel Go proposto e infrastruttura."""

    def __init__(self, message: str, *, log_path: Path, failure_kind: str,
                 debugger_can_revise_code: bool) -> None:
        super().__init__(message)
        self.log_path = log_path
        self.failure_kind = failure_kind
        self.debugger_can_revise_code = debugger_can_revise_code


class GoSourceSyntaxError(RuntimeError):
    """Il Go non e' sintatticamente valido: compete al Debugger LLM."""


class EnvironmentPreparation:
    """Costruisce soltanto Dockerfile da template con pacchetti validati."""

    _PKG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:==[A-Za-z0-9_.+-]+)?\Z")
    _APK = re.compile(r"[a-z0-9][a-z0-9+_.-]*\Z")
    _MODULE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._~+/-]*@[A-Za-z0-9._+!/-]+\Z")
    _GO_IMPORT = re.compile(r'^\s*(?:[A-Za-z_]\w*\s+)?"([^"\n]+)"', re.M)
    _PYTHON_MAP = {
        "PIL": "Pillow", "yaml": "PyYAML", "cv2": "opencv-python-headless",
        "sklearn": "scikit-learn", "Crypto": "pycryptodome",
        "numpy": "numpy", "pandas": "pandas", "requests": "requests",
        "jinja2": "Jinja2", "pyaes": "pyaes", "igraph": "igraph",
        "torch": "torch", "torchvision": "torchvision", "cryptography": "cryptography",
    }

    _INFRASTRUCTURE_SIGNATURES = (
        "cannot connect to the docker daemon", "error during connect",
        "permission denied while trying to connect to the docker daemon",
        "no space left on device", "network is unreachable", "connection refused",
        "temporary failure in name resolution", "i/o timeout", "tls handshake timeout",
        "context deadline exceeded", "unable to resolve host address",
    )
    _GO_DEPENDENCY_SIGNATURES = (
        "toolchain upgrade needed", "requires go >=", "requires go >",
        "no required module provides package", "cannot find module providing package",
        "module found, but does not contain package", "does not contain package",
        "cannot find package", "unknown revision", "invalid version:",
        "malformed module path", "go: errors parsing go.mod",
        "go: finding module for package", "go: downloading ",
        "unrecognized import path", "module lookup disabled by goproxy=off",
    )

    @classmethod
    def classify_failure(cls, *, language: str, log_text: str) -> str:
        """Classifica l'errore osservato, senza modificare codice o dipendenze."""
        diagnostic = log_text.lower()
        # Un download fallito per rete non e' un errore correggibile dal Debugger.
        if any(x in diagnostic for x in cls._INFRASTRUCTURE_SIGNATURES):
            return "infrastructure"
        if language == "go" and any(x in diagnostic for x in cls._GO_DEPENDENCY_SIGNATURES):
            return "go_dependency_resolution"
        return "unclassified"

    @staticmethod
    def build_log_excerpt(path: Path, *, max_chars: int = 12000) -> str:
        """Mantiene le righe d'errore e la coda della build, inclusi stdout e stderr.

        Una build lunga puo' includere migliaia di righe di installazione: nel
        prompt sono conservati il contesto finale e i diagnostici pertinenti.
        Il file build.log completo resta comunque immutato sul disco.
        """
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) <= max_chars:
            return text
        markers = (
            "error", "failed", "requires go", "toolchain", "cannot find",
            "no required module", "does not contain", "go: ", "traceback",
            "modulenotfounderror", "permission denied", "connection refused",
        )
        relevant = [line for line in text.splitlines()
                    if any(marker in line.lower() for marker in markers)]
        # Le righe diagnostiche NON sostituiscono il log completo; sono una
        # selezione strutturata, con la coda originale a fornire contesto.
        selected = "\n".join(relevant[-40:])[-max_chars // 2:]
        tail = text[-(max_chars - len(selected) - 100):]
        return ("[righe diagnostiche]\n" + selected + "\n[coda completa]\n" + tail)

    def __init__(self, *, directory: Path, docker: str = "docker",
                 python_image: str = "python:3.11-alpine",
                 go_image: str = "golang:1.25-alpine",
                 runtime_image: str = "alpine:3.22",
                 timeout_seconds: float = 600.0) -> None:
        self.directory = directory.expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.docker = docker
        self.python_image = python_image
        self.go_image = go_image
        self.runtime_image = runtime_image
        self.timeout_seconds = timeout_seconds
        self.python_result: dict | None = None
        self.go_result: dict | None = None

    @staticmethod
    def python_imports(source: str) -> list[str]:
        """Elenca gli import top-level senza eseguire il codice della funzione."""
        tree = ast.parse(source)
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(item.name.split(".", 1)[0] for item in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".", 1)[0])
        return sorted(modules)

    @classmethod
    def python_packages(cls, source: str) -> list[str]:
        """Rileva import noti; gli altri restano visibili nel manifest diagnostico."""
        return sorted(set(cls._PYTHON_MAP[mod] for mod in cls.python_imports(source)
                          if mod in cls._PYTHON_MAP))

    @classmethod
    def python_declared_packages(cls, requirements_text: str | None) -> list[str]:
        """Estrae requirements Python originali con sintassi pip semplice.

        URL, flag pip, inclusioni di file e constraint complessi richiederanno
        in futuro un adattatore esplicito, senza eseguire comandi arbitrari.
        """
        if requirements_text is None:
            return []
        if type(requirements_text) is not str:
            raise ValueError("Requirements Python non validi")
        packages: list[str] = []
        for line in requirements_text.splitlines():
            spec = line.split("#", 1)[0].strip()
            if not spec:
                continue
            if not cls._PKG.fullmatch(spec):
                raise ValueError("Requirement Python originale non supportato: " + spec[:160])
            packages.append(spec)
        cls.validate_packages(pip=packages, apk=[], modules=[])
        return packages

    @classmethod
    def resolved_python_packages(cls, source: str, requirements_text: str | None,
                                 *, test_source: str | None = None) -> list[str]:
        """Usa requirements originali e import statici di funzione e test Python.

        I test sono solo analizzati con AST, mai eseguiti durante la selezione;
        il sorgente originale resta immutato e mantiene il proprio SHA-256.
        """
        declared = cls.python_declared_packages(requirements_text)
        pinned = {name.split("==", 1)[0].replace("_", "-").lower()
                  for name in declared}
        imports = cls.python_packages(source)
        if test_source is not None:
            if type(test_source) is not str:
                raise ValueError("Sorgente dei test Python non valido")
            imports += cls.python_packages(test_source)
        mapped = sorted({name for name in imports
                         if name.replace("_", "-").lower() not in pinned})
        packages = declared + mapped
        cls.validate_packages(pip=packages, apk=[], modules=[])
        return packages

    @classmethod
    def go_external_imports(cls, source: str) -> list[str]:
        """Identifica import di moduli con dominio; non interpreta il codice Go."""
        found = []
        block = re.search(r'(?m)^\s*import\s*\((.*?)\)', source, re.S)
        if block:
            found += cls._GO_IMPORT.findall(block.group(1))
        found += re.findall(r'(?m)^\s*import\s+(?:[A-Za-z_]\w*\s+)?"([^"\n]+)"', source)
        return sorted({x for x in found if "." in x.split("/", 1)[0]})

    @classmethod
    def validate_packages(cls, *, pip: list[str], apk: list[str], modules: list[str]) -> None:
        """Blocca shell expansion, RUN arbitrari, URL e pacchetti in eccesso."""
        for values, matcher, label in ((pip, cls._PKG, "pip"), (apk, cls._APK, "apk"),
                                       (modules, cls._MODULE, "Go")):
            if type(values) is not list or len(values) > 24 or any(
                type(x) is not str or len(x) > 120 or not matcher.fullmatch(x) for x in values
            ):
                raise ValueError("Specifiche dipendenze non autorizzate: " + label)

    @staticmethod
    def _record(path: Path, data: dict) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _docker(self, args: list[str], *, timeout: float | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess:
        try:
            return subprocess.run([self.docker, *args], capture_output=True, check=False,
                                  timeout=timeout or self.timeout_seconds, cwd=cwd)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("Docker non disponibile o operazione scaduta: " + str(exc)) from exc

    def _image_digest(self, image: str) -> str:
        result = self._docker(["image", "inspect", "--format", "{{.Id}}", image], timeout=20)
        if result.returncode:
            raise RuntimeError("Impossibile ottenere il digest dell'immagine: " + image)
        return result.stdout.decode("utf-8", "replace").strip()

    def build(self, *, language: str, source: str, entry: str,
              pip: list[str] | None = None, apk: list[str] | None = None,
              modules: list[str] | None = None, attempt: int = 0,
              enable_cgo: bool = False) -> dict:
        """Prepara solo dipendenze; nessun container di benchmark viene avviato."""
        if (language not in ("python", "go") or type(attempt) is not int
                or attempt < 0 or type(enable_cgo) is not bool
                or type(entry) is not str or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", entry)
                or entry == "main"):
            raise ValueError("Lingua o tentativo non valido")
        pip, apk, modules = list(pip or []), list(apk or []), list(modules or [])
        self.validate_packages(pip=pip, apk=apk, modules=modules)
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        # Uno stesso sorgente può essere proposto piu' volte dall'LLM. Ogni
        # build deve avere i propri log, anche se hash e configurazione sono
        # identici. Non sovrascrivere evidenze di tentativi precedenti.
        folder_base = self.directory / language / f"{digest[:12]}_attempt_{attempt:02d}"
        folder = folder_base
        suffix = 0
        while True:
            try:
                folder.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                suffix += 1
                if suffix > 1000:
                    raise RuntimeError("Troppi tentativi Docker con lo stesso hash")
                folder = folder_base.with_name(folder_base.name + f"_repeat_{suffix:03d}")
        manifest = {"language": language, "code_sha256": digest, "entry": entry,
                    "attempt": attempt, "pip": pip, "apk": apk, "modules": modules,
                    "enable_cgo": enable_cgo,
                    "status": "preparing", "folder": str(folder)}
        self._record(folder / "plan.json", manifest)
        if language == "python":
            has_packages = bool(pip or apk)
            image = f"reliq-env-py:{digest[:16]}-{attempt}"
            steps = [f"FROM {self.python_image}"]
            if apk:
                steps.append("RUN apk add --no-cache " + " ".join(apk))
            if pip:
                steps.append("RUN python -m pip install --no-cache-dir " + " ".join(pip))
            runtime_image = image if has_packages else self.python_image
            build_image = None
            file_name = "Dockerfile.python"
        else:
            external_imports = self.go_external_imports(source)
            cgo = enable_cgo or 'import "C"' in source
            has_packages = bool(external_imports or apk or modules or cgo)
            image = f"reliq-env-go:{digest[:16]}-{attempt}"
            steps = [f"FROM {self.go_image}", "WORKDIR /opt/reliq"]
            if apk or cgo:
                deps = sorted(set(apk + (["build-base"] if cgo else [])))
                steps.append("RUN apk add --no-cache " + " ".join(deps))
            if has_packages:
                steps += ["COPY function.go harness.go ./", "RUN go mod init reliq.local/workload"]
                for mod in modules:
                    steps.append("RUN go mod edit -require=" + mod)
                steps.append("RUN go mod tidy && go mod vendor")
            runtime_image = (f"reliq-env-go-runtime:{digest[:16]}-{attempt}"
                             if apk else self.runtime_image)
            build_image = image if has_packages else self.go_image
            file_name = "Dockerfile.go"
            (folder / "function.go").write_text(source, encoding="utf-8")
            harness_path = Path(__file__).resolve().parents[1] / "validation" / "GoHarness.go"
            harness = harness_path.read_text(encoding="utf-8")
            if harness.count("__REDATASET_ENTRY_POINT__") != 1:
                raise RuntimeError("Marker GoHarness non valido")
            (folder / "harness.go").write_text(
                harness.replace("__REDATASET_ENTRY_POINT__", entry), encoding="utf-8")
            manifest["cgo"] = cgo
            manifest["external_imports"] = external_imports
        (folder / file_name).write_text("\n".join(steps) + "\n", encoding="utf-8")
        log_path = folder / "build.log"
        if language == "go" and has_packages:
            # gofmt -e controlla solo la sintassi del sorgente; non lo corregge.
            syntax = self._docker([
                "run", "--rm", "--pull=never", "--network=none", "--read-only",
                "--cap-drop=ALL", "--security-opt=no-new-privileges", "--cpus=1",
                "--memory=1g", "--user", f"{os.getuid()}:{os.getgid()}",
                "--mount", f"type=bind,src={folder},dst=/work,readonly",
                self.go_image, "gofmt", "-e", "/work/function.go"], timeout=30)
            if syntax.returncode != 0:
                diagnostic = (syntax.stderr or syntax.stdout).decode("utf-8", "replace")[-1200:]
                (folder / "syntax.log").write_text(diagnostic + "\n", encoding="utf-8")
                if syntax.returncode != 125:
                    raise GoSourceSyntaxError(diagnostic)
                raise EnvironmentBuildError("Docker gofmt non eseguibile: " + diagnostic,
                                            log_path=folder / "syntax.log")
        if has_packages:
            # --progress appartiene a BuildKit/buildx: il builder legacy non lo supporta.
            # stdout/stderr sono comunque catturati e conservati in build.log.
            outcome = self._docker(["build", "--pull=false", "-f", file_name,
                                    "-t", image, "."], timeout=self.timeout_seconds, cwd=folder)
            log_path.write_bytes((outcome.stdout + b"\n" + outcome.stderr))
            if outcome.returncode:
                manifest["status"] = "failed"
                diagnostic = (outcome.stdout + b"\n" + outcome.stderr).decode(
                    "utf-8", "replace")
                kind = self.classify_failure(language=language, log_text=diagnostic)
                manifest["failure_kind"] = kind
                self._record(folder / "plan.json", manifest)
                raise EnvironmentBuildError(
                    f"Docker build {language} fallita ({kind}): " +
                    self.build_log_excerpt(log_path, max_chars=4500),
                    log_path=log_path, failure_kind=kind)
        else:
            log_path.write_text("Nessuna dipendenza esterna: immagini originali riutilizzate.\n", encoding="utf-8")
        if language == "go" and apk:
            runtime_file = folder / "Dockerfile.go.runtime"
            runtime_file.write_text(
                f"FROM {self.runtime_image}\nRUN apk add --no-cache " + " ".join(apk) + "\n",
                encoding="utf-8")
            runtime_build = self._docker([
                "build", "--pull=false", "-f", runtime_file.name,
                "-t", runtime_image, "."], timeout=self.timeout_seconds, cwd=folder)
            (folder / "runtime-build.log").write_bytes((runtime_build.stdout + b"\n" +
                                                         runtime_build.stderr))
            if runtime_build.returncode:
                manifest["status"] = "failed_runtime"
                self._record(folder / "plan.json", manifest)
                raise EnvironmentBuildError("Docker runtime Go fallito: " +
                                            (runtime_build.stderr or runtime_build.stdout)
                                            .decode("utf-8", "replace")[-2400:],
                                            log_path=folder / "runtime-build.log")
        if language == "python":
            external = [mod for mod in self.python_imports(source)
                        if mod not in sys.stdlib_module_names and mod not in sys.builtin_module_names
                        and mod != "__future__"]
            manifest["imports_checked"] = external
            if external:
                statement = "; ".join("import " + mod for mod in external)
                checked = self._docker([
                    "run", "--rm", "--pull=never", "--network=none", "--read-only",
                    "--cap-drop=ALL", "--security-opt=no-new-privileges",
                    "--memory=1g", "--cpus=1", runtime_image,
                    "python", "-I", "-c", statement], timeout=120)
                (folder / "imports.log").write_bytes(checked.stderr + checked.stdout)
                if checked.returncode != 0:
                    manifest["status"] = "failed_import_check"
                    self._record(folder / "plan.json", manifest)
                    raise EnvironmentBuildError("Import Python non disponibile: " +
                        (checked.stderr or checked.stdout).decode("utf-8", "replace")[-1600:],
                        log_path=folder / "imports.log")
            if has_packages:
                frozen = self._docker(["run", "--rm", "--pull=never", "--network=none",
                                       runtime_image, "python", "-m", "pip", "freeze", "--all"],
                                      timeout=45)
                (folder / "pip-freeze.txt").write_bytes(frozen.stdout + frozen.stderr)
                if frozen.returncode:
                    raise EnvironmentBuildError("Impossibile rilevare versioni pip",
                                                log_path=folder / "pip-freeze.txt")
        if language == "go" and has_packages:
            created = self._docker(["create", image], timeout=20)
            if created.returncode:
                raise EnvironmentBuildError("Impossibile ispezionare i moduli Go",
                                            log_path=log_path)
            container = created.stdout.decode("ascii", "replace").strip()
            try:
                for filename in ("go.mod", "go.sum"):
                    copied = self._docker(["cp", f"{container}:/opt/reliq/{filename}",
                                           str(folder / filename)], timeout=45)
                    if copied.returncode and filename == "go.mod":
                        raise EnvironmentBuildError("go.mod non presente nell'immagine",
                                                    log_path=log_path)
            finally:
                self._docker(["rm", "-f", container], timeout=20)
        manifest.update({"status": "completed", "python_image": runtime_image if language == "python" else None,
                         "build_image": build_image, "go_image": runtime_image if language == "go" else None,
                         "go_modules_enabled": bool(language == "go" and has_packages),
                         "go_cgo_enabled": bool(language == "go" and manifest.get("cgo")),
                         "image_digest": self._image_digest(image if has_packages else
                                                            (self.python_image if language == "python" else self.go_image)),
                         "runtime_image_digest": self._image_digest(runtime_image)})
        self._record(folder / "plan.json", manifest)
        if language == "python":
            self.python_result = manifest
        else:
            self.go_result = manifest
        self._record(self.directory / f"current_{language}.json", manifest)
        return manifest
