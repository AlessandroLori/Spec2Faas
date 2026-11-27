"""Confronta l'energia di invocazione Python/Go in container Docker persistenti."""

import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable

from app.profiling.EnergyProfiler import RaplMeter, ScaphandreSampler, median_or_none, two_point
from app.profiling.EnergyVariability import sample_statistics, summarize_energy_variability


class FunctionEnergyRunner:
    """Prepara i runtime e coordina N/1 senza avviare Docker nella finestra RAPL."""

    _entry = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z", re.ASCII)
    _marker = "__REDATASET_ENTRY_POINT__"

    def __init__(self, *, rapl: RaplMeter, docker: str = "docker",
                 python_image: str = "python:3.11-alpine",
                 build_image: str = "golang:1.25-alpine",
                 go_image: str = "alpine:3.22",
                 scaphandre_url: str | None = None,
                 timeout_seconds: float = 90.0,
                 progress: Callable[[str], None] | None = None,
                 go_gomaxprocs: int | None = 2,
                 go_modules_enabled: bool = False, go_cgo_enabled: bool = False) -> None:
        self.rapl = rapl
        self.docker = docker
        self.python_image = python_image
        self.build_image = build_image
        self.go_image = go_image
        self.go_modules_enabled = go_modules_enabled
        self.go_cgo_enabled = go_cgo_enabled
        self.scaphandre_url = scaphandre_url
        # -1 e' il sentinel CLI per disattivare il timeout delle invocazioni.
        self.timeout_seconds = None if timeout_seconds == -1 else timeout_seconds
        self._progress = progress
        if go_gomaxprocs is not None and (
                type(go_gomaxprocs) is not int or not 1 <= go_gomaxprocs <= 256):
            raise ValueError("GOMAXPROCS Go deve essere un intero fra 1 e 256")
        self.go_gomaxprocs = go_gomaxprocs
        self._validation_harness = Path(__file__).resolve().parents[1] / "validation" / "GoHarness.go"
        self._python_harness = Path(__file__).with_name("PythonEnergyHarness.py")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or (
            not math.isfinite(timeout_seconds) or (timeout_seconds <= 0 and timeout_seconds != -1)
        ):
            raise ValueError("Timeout Docker non valido: usare secondi positivi o -1")

    def _notify(self, message: str) -> None:
        """Mostra la fase corrente senza influenzare la finestra di misura RAPL."""
        if self._progress is not None:
            self._progress(message)

    @staticmethod
    def _execute(command: list[str], *, timeout: float = 15.0, payload: bytes | None = None) -> subprocess.CompletedProcess:
        """Esegue comandi Docker senza passare da una shell."""
        try:
            return subprocess.run(command, input=payload, capture_output=True,
                                  timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Timeout comando: {command[:3]}") from exc
        except OSError as exc:
            raise RuntimeError("Docker non disponibile") from exc

    def preflight(self) -> dict:
        """Controlla sensore e immagini già presenti; non scarica nulla."""
        if os.getuid() == 0:
            raise RuntimeError("Avviare il profiler come utente host non root")
        if shutil.which(self.docker) is None:
            raise RuntimeError("Docker non disponibile nel PATH")
        for image in (self.python_image, self.build_image, self.go_image):
            result = self._execute([self.docker, "image", "inspect", image])
            if result.returncode:
                raise RuntimeError(f"Immagine Docker mancante: {image}; scaricarla prima del benchmark")
        info = self._execute([self.docker, "info", "--format", "{{.ServerVersion}}"])
        if info.returncode:
            raise RuntimeError("Docker daemon non accessibile dall'utente corrente")
        return {"docker_version": info.stdout.decode("utf-8", "replace").strip(),
                "rapl": self.rapl.preflight(),
                "images": [self.python_image, self.build_image, self.go_image]}

    def _container_flags(self, name: str, directory: Path, *, writable: bool) -> list[str]:
        """Imposta limiti Docker coerenti con il validatore Go."""
        mount = f"type=bind,src={directory},dst=/work" + ("" if writable else ",readonly")
        flags = [
            "--pull=never", "--name", name, "--network=none", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=64",
            "--memory=1g", "--memory-swap=1g", "--cpus=1",
            "--user", f"{os.getuid()}:{os.getgid()}", "--mount", mount,
            "--workdir", "/work", "--tmpfs", "/tmp:rw,nosuid,nodev,size=128m",
        ]
        # Solo runtime Go energetico, non Python né compilatore. Docker exec
        # eredita l'ambiente del container anche per preflight e calibrazione.
        if (self.go_gomaxprocs is not None and not writable
                and name.startswith("function-energy-go-")):
            flags.extend(["--env", f"GOMAXPROCS={self.go_gomaxprocs}"])
        return flags

    def _compile_go(self, directory: Path) -> None:
        """Compila Go prima di qualunque misurazione energetica."""
        name = "function-build-" + uuid.uuid4().hex[:12]
        flags = self._container_flags(name, directory, writable=True)
        command = [self.docker, "run", "--rm", *flags, "--env", "HOME=/tmp",
                   "--env", "GOCACHE=/tmp/gocache",
                   "--env", "GO111MODULE=on" if self.go_modules_enabled else "GO111MODULE=off",
                   "--env", "GOWORK=off", "--env", "GOTOOLCHAIN=local",
                   "--env", "GOPROXY=off", "--env", "GOSUMDB=off",
                   "--env", "CGO_ENABLED=1" if self.go_cgo_enabled else "CGO_ENABLED=0",
                   self.build_image]
        if self.go_modules_enabled:
            command += ["go", "-C", "/opt/spec2faas", "build", "-mod=vendor", "-trimpath",
                        "-o", "/work/runner", "/work/function.go", "/work/harness.go"]
        else:
            command += ["go", "build", "-trimpath", "-o", "/work/runner",
                        "function.go", "harness.go"]
        outcome = self._execute(command, timeout=180)
        if outcome.returncode:
            raise RuntimeError("Compilazione Go fallita: " + outcome.stderr.decode("utf-8", "replace")[:1000])
        if not (directory / "runner").is_file():
            raise RuntimeError("Il compilatore Go non ha prodotto il binario")

    def _launch(self, name: str, directory: Path, image: str) -> str:
        """Avvia il container prima del benchmark, mantenendolo inattivo."""
        flags = self._container_flags(name, directory, writable=False)
        result = self._execute([self.docker, "run", "--detach", *flags,
                                image, "sleep", "2147483647" if self.timeout_seconds is None else "3600"])
        if result.returncode:
            raise RuntimeError("Avvio container fallito: " + result.stderr.decode("utf-8", "replace")[:500])
        container_id = result.stdout.decode("utf-8", "replace").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", container_id):
            raise RuntimeError("Docker non ha restituito un ID valido")
        return container_id

    def _docker_host_pids(self, container_name: str) -> set[int]:
        """Recupera i PID host effettivi del container, anche con cgroups v2.

        Il metodo viene invocato soltanto durante la prova Scaphandre,
        separata dalla finestra RAPL a due punti.
        """
        result = self._execute([self.docker, "top", container_name, "-eo", "pid"], timeout=5.0)
        if result.returncode:
            raise RuntimeError("Docker top non disponibile durante la misura Scaphandre: "
                               + result.stderr.decode("utf-8", "replace")[:200])
        lines = result.stdout.decode("utf-8", "replace").splitlines()
        if not lines or lines[0].strip().upper() != "PID":
            raise RuntimeError("Formato PID inatteso da docker top")
        pids = set()
        for line in lines[1:]:
            if not line.strip().isdigit():
                raise RuntimeError("Docker top ha restituito un PID non numerico")
            pids.add(int(line.strip()))
        return pids

    def _call(self, name: str, language: str, input_path: Path,
              *, entry: str, capture: bool = False) -> tuple[float, dict | None]:
        """Esegue uno stream nel container già avviato; evita di accumulare output in RAM."""
        program = (["python", "-I", "/work/PythonEnergyHarness.py", "/work/function.py", entry]
                   if language == "python" else ["/work/runner"])
        command = [self.docker, "exec", "--interactive", name, *program]
        with input_path.open("rb") as input_file, tempfile.TemporaryFile() as error_file:
            with tempfile.TemporaryFile() as output_file:
                started = time.monotonic()
                try:
                    completed = subprocess.run(
                        command, stdin=input_file, stdout=output_file if capture else subprocess.DEVNULL,
                        stderr=error_file, timeout=self.timeout_seconds, check=False,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(f"Timeout esecuzione {language}") from exc
                elapsed = time.monotonic() - started
                if completed.returncode:
                    error_file.seek(0)
                    message = error_file.read(1000).decode("utf-8", "replace")
                    raise RuntimeError(f"Errore nel container {language}: {message[:900]}")
                if not capture:
                    return elapsed, None
                output_file.seek(0)
                try:
                    rows = [json.loads(line) for line in output_file]
                except ValueError as exc:
                    raise RuntimeError(f"Protocollo {language} non valido") from exc
                if len(rows) != 1 or rows[0].get("index") != 0 or "value" not in rows[0] or rows[0].get("error"):
                    raise RuntimeError(f"Il controllo preliminare di {language} non ha prodotto un risultato valido")
                return elapsed, rows[0]

    def _call_timer(self, name: str, language: str, input_path: Path, *, entry: str,
                    seconds: float, max_invocations: int) -> dict:
        """Esegue un processo che cicla input JSON fino al tempo richiesto.

        I metadati di fine lavoro provengono dallo stesso harness che ha
        eseguito le invocazioni, non da una stima della velocita' esterna.
        """
        marker = "__SPEC2FAAS_TIMER_V1__"
        extra = ["--timer-seconds", str(seconds), "--timer-max-invocations", str(max_invocations)]
        program = (["python", "-I", "/work/PythonEnergyHarness.py", "/work/function.py", entry, *extra]
                   if language == "python" else ["/work/runner", *extra])
        command = [self.docker, "exec", "--interactive", name, *program]
        with input_path.open("rb") as input_file, tempfile.TemporaryFile() as error_file:
            try:
                completed = subprocess.run(
                    command, stdin=input_file, stdout=subprocess.DEVNULL,
                    stderr=error_file, timeout=self.timeout_seconds, check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"Timeout Timer {language}") from exc
            if completed.returncode:
                error_file.seek(0, os.SEEK_END)
                size = error_file.tell()
                error_file.seek(max(0, size - 4096))
                detail = error_file.read().decode("utf-8", "replace")
                raise RuntimeError(f"Errore Timer {language}: {detail[-900:]}")
            # Legge soltanto la coda: le eventuali stampe della funzione su
            # stderr non devono saturare la memoria del processo host.
            error_file.seek(0, os.SEEK_END)
            size = error_file.tell()
            error_file.seek(max(0, size - 8192))
            tail = error_file.read().decode("utf-8", "replace")
            markers = [line[len(marker):] for line in tail.splitlines() if line.startswith(marker)]
            if len(markers) != 1:
                raise RuntimeError(f"Timer {language}: risultato finale mancante o ambiguo")
            try:
                result = json.loads(markers[0])
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Timer {language}: metadati JSON invalidi") from exc
            n = result.get("invocations")
            elapsed = result.get("workload_seconds")
            reason = result.get("stop_reason")
            if (type(n) is not int or not 1 <= n <= max_invocations or
                    not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0 or
                    reason not in ("duration_reached", "invocation_cap_reached") or
                    (reason == "duration_reached" and elapsed < seconds) or
                    (reason == "invocation_cap_reached" and elapsed >= seconds)):
                raise RuntimeError(f"Timer {language}: contatore o tempo non valido")
            return result

    @staticmethod
    def _stream(directory: Path, cases: list[list], invocations: int) -> Path:
        """Prepara una sola volta gli input per N, fuori dalla finestra energetica.

        Gli stessi casi e lo stesso N sono riutilizzati in ogni ripetizione:
        riscrivere un milione di righe per ogni prova non misura la funzione.
        """
        target = directory / f"cases_{invocations}.jsonl"
        if target.is_file():
            return target
        # Codifica gli argomenti una sola volta: cambia solo l'indice della riga.
        encoded = [json.dumps(case, ensure_ascii=False, allow_nan=False,
                              separators=(",", ":")) for case in cases]
        temporary = target.with_suffix(".jsonl.part")
        try:
            with temporary.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
                for index in range(invocations):
                    handle.write(f'{{"index":{index},"args":{encoded[index % len(encoded)]}}}\n')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    @staticmethod
    def _same_output(py_value, go_value) -> bool:
        """Evita falsi mismatch per ordine delle chiavi nel body JSON Lambda.

        Differenze nei valori e codici HTTP restano differenze reali.
        """
        if not isinstance(py_value, dict) or not isinstance(go_value, dict):
            return py_value == go_value
        py_value, go_value = py_value.copy(), go_value.copy()
        if isinstance(py_value.get("body"), str) and isinstance(go_value.get("body"), str):
            try:
                py_value["body"] = json.loads(py_value["body"])
                go_value["body"] = json.loads(go_value["body"])
            except ValueError:
                pass
        return py_value == go_value

    @classmethod
    def _check_profile_preflight(cls, py_value, go_value, *,
                                 validation_suite=None, validation_case_id=None,
                                 cases=None) -> str:
        """Usa l'oracolo originale per input di test attestati; altrimenti confronto esatto.

        Non normalizza o cancella campi volatili in modo implicito. I validatori
        originali definiscono che cosa e' funzionalmente rilevante per quel task.
        """
        if validation_suite is None:
            if not cls._same_output(py_value, go_value):
                raise RuntimeError("Output Python/Go diverso nella verifica preliminare")
            return "strict_cross_language"

        if (not isinstance(validation_case_id, str) or not validation_case_id
                or not isinstance(cases, list) or len(cases) != 1):
            raise ValueError("Preflight con suite: specificare un unico test originale")
        matches = [c for c in validation_suite.cases if c.case_id == validation_case_id]
        if len(matches) != 1 or list(matches[0].args) != cases[0]:
            raise ValueError("L'input di profilazione non coincide con il test originale selezionato")

        # Riproduce l'identica verifica delle traduzioni (SeBS, ReFaaS, HumanEval+),
        # usando la suite originale invariata e il solo caso effettivamente profilato.
        from dataclasses import replace
        from app.validation.ExecutionModels import CaseExecution, ExecutionResult
        from app.validation.OutputComparator import OutputComparator
        tested_suite = replace(validation_suite, cases=(matches[0],))
        comparator = OutputComparator()
        for language, value in (("python", py_value), ("go", go_value)):
            execution = ExecutionResult(
                candidate_id=tested_suite.candidate_id,
                test_sha256=tested_suite.test_sha256, language=language,
                code_sha256="0" * 64, status="completed",
                cases=(CaseExecution(matches[0].case_id, "returned", value),),
                compile_seconds=0.0, execution_seconds=0.0)
            verified = comparator.compare(tested_suite, execution)
            if not verified.passed:
                raise RuntimeError(
                    f"Preflight {language}: test originale {validation_case_id} non superato: "
                    + (verified.error_message or verified.status))
        return "original_test_oracle"

    @staticmethod
    def _net(raw_j: float, elapsed_s: float, idle_w: float) -> float:
        """Calcola l'energia incrementale stimata, distinta dal package lordo."""
        return raw_j - idle_w * elapsed_s

    def _paired_rapl_points(self, language: str, name: str, directory: Path,
                            cases: list[list], n: int, entry: str,
                            container_id: str) -> tuple[dict, dict, dict]:
        """Misura idle prima, fra e dopo i punti 1/N dello stesso linguaggio.

        Il riferimento locale di ogni punto e' la media delle due finestre
        idle adiacenti. Non modifica l'energia lorda registrata da RAPL.
        """
        before = self.rapl.idle_power()
        short = self._measure_run(language, name, directory, cases, 1, entry,
                                  before, container_id)
        middle = self.rapl.idle_power()
        long = self._measure_run(language, name, directory, cases, n, entry,
                                 middle, container_id)
        after = self.rapl.idle_power()
        readings = (before, middle, after)
        if not all(math.isfinite(item) and item >= 0 for item in readings):
            raise RuntimeError("Riferimenti idle RAPL non validi")
        short_idle = (before + middle) / 2
        long_idle = (middle + after) / 2
        for point, power in ((short, short_idle), (long, long_idle)):
            point["idle_reference_w"] = power
            point["rapl_package_incremental_estimate_j"] = self._net(
                point["rapl_package_gross_j"], point["elapsed_seconds"], power)
        spread = max(readings) - min(readings)
        # Flag diagnostico, non cancellazione arbitraria dei risultati.
        median = statistics.median(readings)
        unstable = spread > max(3.0, 0.25 * median)
        baseline = {
            "method": "bracketed_three_idle_windows_per_1_N_pair",
            "before_one_w": before,
            "between_one_and_n_w": middle,
            "after_n_w": after,
            "one_reference_w": short_idle,
            "many_reference_w": long_idle,
            "range_w": spread,
            "idle_three_point_statistics_w": sample_statistics(readings),
            "stability": "needs_validation" if unstable else "nominal",
            "note": ("Il riferimento idle varia durante la coppia 1/N: valutare "
                     "la sensibilita' dell'energia incrementale." if unstable else None),
        }
        return short, long, baseline

    def _measure_run(self, language: str, name: str, directory: Path,
                     cases: list[list], n: int, entry: str, idle_w: float,
                     container_id: str, *, observe_scaph: bool = False) -> dict:
        """Acquisisce sia l'energia RAPL sia, nel tratto lungo, la stima Scaphandre."""
        path = self._stream(directory, cases, n)
        sampler = None
        if self.scaphandre_url and observe_scaph:
            # I PID già presenti appartengono al container in attesa (`sleep`),
            # non all'invocazione della funzione: escluderli dall'attribuzione.
            idle_container_pids = self._docker_host_pids(name)
            sampler = ScaphandreSampler(
                self.scaphandre_url, container_id, name,
                host_pids=lambda: self._docker_host_pids(name),
                exclude_pids=idle_container_pids,
            )
            sampler.start()
        try:
            result, _ = self.rapl.measure(lambda: self._call(name, language, path, entry=entry))
        finally:
            if sampler:
                scaph = sampler.stop()
            else:
                scaph = {"status": "disabled", "attributed_j": None}
        if sampler:
            diagnostics = scaph.get("diagnostics", {})
            self._notify(
                f"Scaphandre {language}: {scaph['status']}; "
                f"PID Docker visibili={diagnostics.get('workload_pid_visible_samples', 0)}, "
                f"campioni associati={diagnostics.get('process_power_matched_samples', 0)}, "
                f"copertura={scaph.get('coverage_fraction', 0):.1%}"
            )
        return {
            "invocations": n, "elapsed_seconds": result.elapsed_s,
            "rapl_package_gross_j": result.joules,
            "rapl_package_incremental_estimate_j": self._net(result.joules, result.elapsed_s, idle_w),
            "scaphandre": scaph,
        }

    def run(self, *, python_source: Path, go_source: Path,
            entry: str, cases: list[list], invocations: int = 1000,
            max_invocations: int = 1000000, repetitions: int = 5,
            min_long_seconds: float = 3.0,
            validation_suite=None, validation_case_id: str | None = None) -> dict:
        """Misura entrambi i linguaggi allo stesso N, senza confondere codice e LLM."""
        if not self._entry.fullmatch(entry) or entry == "main":
            raise ValueError("Entry point invalido")
        if not isinstance(cases, list) or not cases or any(not isinstance(item, list) for item in cases):
            raise ValueError("I casi devono essere un array JSON di array di argomenti")
        if not 2 <= invocations <= max_invocations <= 1_000_000:
            raise ValueError("Numero di invocazioni fuori range")
        if not 1 <= repetitions <= 20:
            raise ValueError("Ripetizioni fuori range")
        if not math.isfinite(min_long_seconds) or min_long_seconds < 0:
            raise ValueError("Durata minima non valida")
        python_code = python_source.read_text(encoding="utf-8")
        go_code = go_source.read_text(encoding="utf-8")
        if not python_code.strip() or not go_code.strip():
            raise ValueError("File sorgente vuoto")
        # Non modifichiamo le traduzioni: il template di validazione è identico.
        harness = self._validation_harness.read_text(encoding="utf-8")
        if harness.count(self._marker) != 1:
            raise RuntimeError("Marker GoHarness inatteso")
        environment = self.preflight()
        environment["go_runtime"] = {
            "gomaxprocs_mode": "automatic" if self.go_gomaxprocs is None else "explicit",
            "gomaxprocs_override": self.go_gomaxprocs,
        }
        names = {lang: "function-energy-" + lang + "-" + uuid.uuid4().hex[:10]
                 for lang in ("python", "go")}
        containers = {}
        with tempfile.TemporaryDirectory(prefix="function_energy_") as work:
            root = Path(work)
            folders = {lang: root / lang for lang in names}
            for folder in folders.values():
                folder.mkdir()
            (folders["python"] / "function.py").write_text(python_code, encoding="utf-8")
            (folders["python"] / "PythonEnergyHarness.py").write_text(
                self._python_harness.read_text(encoding="utf-8"), encoding="utf-8")
            (folders["go"] / "function.go").write_text(go_code, encoding="utf-8")
            (folders["go"] / "harness.go").write_text(
                harness.replace(self._marker, entry), encoding="utf-8")
            try:
                if self.scaphandre_url:
                    import urllib.request
                    # La v1.0.2 può avere metriche non ancora popolate al primo
                    # scrape. Attendere >2 s evita anche il refresh throttle noto.
                    available = False
                    for attempt in range(3):
                        try:
                            with urllib.request.urlopen(self.scaphandre_url, timeout=5) as response:
                                # Leggere fino a EOF: risposte troncate causano BrokenPipe.
                                metrics = response.read().decode("utf-8", "replace")
                        except (OSError, ValueError) as exc:
                            raise RuntimeError("Scaphandre Prometheus non raggiungibile sull'host") from exc
                        if "scaph_process_power_consumption_microwatts" in metrics:
                            available = True
                            break
                        if attempt < 2:
                            self._notify("Scaphandre: metriche di processo non ancora disponibili, nuovo controllo tra 2,5 s")
                            time.sleep(2.5)
                    if not available:
                        raise RuntimeError("Scaphandre raggiungibile ma metrica dei processi assente dopo 3 tentativi")
                self._notify("Preparazione: compilazione Go fuori dal benchmark")
                self._compile_go(folders["go"])
                self._notify("Preparazione: avvio dei container Python e Go")
                for language, image in (("python", self.python_image), ("go", self.go_image)):
                    containers[language] = self._launch(names[language], folders[language], image)
                self._notify("Preparazione: verifica preliminare dell'output Python/Go")
                # Controlla una chiamata in ciascuna lingua prima di misurare l'energia.
                checked = {}
                for language in ("python", "go"):
                    path = self._stream(folders[language], cases, 1)
                    _, checked[language] = self._call(
                        names[language], language, path, entry=entry, capture=True)
                preflight_method = self._check_profile_preflight(
                    checked["python"]["value"], checked["go"]["value"],
                    validation_suite=validation_suite, validation_case_id=validation_case_id,
                    cases=cases)
                self._notify(f"Preflight superato: {preflight_method}")
                self._notify("RAPL: misura di riferimento del package CPU")
                idle_w = self.rapl.idle_power()
                # N viene aumentato insieme per entrambi i linguaggi, mai indipendentemente.
                n = invocations
                while True:
                    self._notify(f"Calibrazione: N={n} invocazioni per linguaggio (minimo {min_long_seconds:g} s)")
                    probe = []
                    for language in ("python", "go"):
                        sample = self._measure_run(language, names[language], folders[language],
                                                   cases, n, entry, idle_w, containers[language])
                        probe.append(sample["elapsed_seconds"])
                    self._notify(f"Calibrazione: Python={probe[0]:.3f} s; Go={probe[1]:.3f} s")
                    if min(probe) >= min_long_seconds or n >= max_invocations:
                        break
                    n = min(n * 10, max_invocations)
                self._notify(f"N definitivo={n}; inizio ripetizioni RAPL (1 e N) e Scaphandre separato")
                results = {language: [] for language in names}
                for repetition in range(repetitions):
                    order = ("python", "go") if repetition % 2 == 0 else ("go", "python")
                    for language in order:
                        self._notify(f"Ripetizione {repetition + 1}/{repetitions} — {language} — RAPL 1/N")
                        short, long, local_idle = self._paired_rapl_points(
                            language, names[language], folders[language], cases, n,
                            entry, containers[language])
                        if local_idle["stability"] != "nominal":
                            self._notify(
                                f"RAPL {language}: riferimento idle variabile "
                                f"({local_idle['range_w']:.2f} W); risultato da validare")
                        steady = two_point(short["rapl_package_incremental_estimate_j"],
                                           long["rapl_package_incremental_estimate_j"], n)
                        gross_steady = two_point(short["rapl_package_gross_j"],
                                                 long["rapl_package_gross_j"], n)
                        # Acquisizione Scaphandre separata: il polling HTTP non contamina E1/EN.
                        scaph_read = None
                        if self.scaphandre_url:
                            self._notify(f"Ripetizione {repetition + 1}/{repetitions} — {language} — Scaphandre N")
                            scaph_read = self._measure_run(
                                language, names[language], folders[language], cases, n, entry,
                                idle_w, containers[language], observe_scaph=True)["scaphandre"]
                        scaph_j = scaph_read.get("attributed_j") if scaph_read else None
                        results[language].append({
                            "repetition": repetition + 1, "one": short, "many": long,
                            "idle_bracketing": local_idle,
                            "rapl_incremental_j_per_invocation": steady,
                            "rapl_gross_j_per_invocation": gross_steady,
                            "scaphandre_attributed_j_per_invocation": scaph_j / n if scaph_j is not None else None,
                            "scaphandre_separate_run": scaph_read,
                        })
                summary = {}
                for language, rows in results.items():
                    incremental = [row["rapl_incremental_j_per_invocation"] for row in rows]
                    gross = [row["rapl_gross_j_per_invocation"] for row in rows]
                    scaph = [row["scaphandre_attributed_j_per_invocation"] for row in rows]
                    valid_incremental = [number for number in incremental if number is not None]
                    valid_gross = [number for number in gross if number is not None]
                    valid_scaph = [number for number in scaph if number is not None]
                    # Un solo campione Scaphandre valido su cinque non giustifica
                    # un gain aggregato; stessa regola di maggioranza di RAPL.
                    minimum = repetitions // 2 + 1
                    summary[language] = {
                        "rapl_incremental_j_per_invocation": (
                            median_or_none(incremental) if len(valid_incremental) >= minimum else None),
                        "rapl_resolved_repetitions": len(valid_incremental),
                        "rapl_incremental_std_j_per_invocation": (
                            statistics.stdev(valid_incremental) if len(valid_incremental) >= 2 else None),
                        "rapl_gross_j_per_invocation": (
                            median_or_none(gross) if len(valid_gross) >= minimum else None),
                        "scaphandre_attributed_j_per_invocation": (
                            median_or_none(scaph) if len(valid_scaph) >= minimum else None),
                        "scaphandre_resolved_repetitions": len(valid_scaph),
                        "scaphandre_std_j_per_invocation": (
                            statistics.stdev(valid_scaph) if len(valid_scaph) >= 2 else None),
                        "rapl_idle_drift_flagged_repetitions": sum(
                            row["idle_bracketing"]["stability"] != "nominal" for row in rows),
                        "repetitions": rows,
                    }
                energy_variability = {
                    language: summarize_energy_variability(
                        rows, scaphandre_key="scaphandre_separate_run")
                    for language, rows in results.items()
                }
                gains = {}
                for key in ("rapl_incremental_j_per_invocation", "scaphandre_attributed_j_per_invocation"):
                    p = summary["python"][key]
                    g = summary["go"][key]
                    gains[key] = (100.0 * (p - g) / p if p is not None and g is not None and p > 0 else None)
                return {
                    "schema_version": 2,
                    "kind": "local_cpu_container_function_energy_comparison",
                    "method": "two_point_rapl_plus_optional_scaphandre_attribution",
                    "definition": "JSON-lines invocation, includes harness overhead; Docker container already running",
                    "execution_protocol": {
                        "container_start_inside_energy_window": False,
                        "go_compilation_inside_energy_window": False,
                        "preflight_and_calibration_inside_energy_window": False,
                        "docker_exec_starts_new_process_for_each_E1_and_EN": True,
                        "module_initialization_once_per_process": True,
                        "N_calls_share_one_process": True,
                        "application_state_reset_between_calls_within_N": False,
                        "json_decode_and_encode_per_invocation": True,
                        "inputs_cycle_over_test_cases": True,
                        "note": "Ogni E1/EN include startup processo/harness; il container e' gia' avviato.",
                    },
                    "entry_point": entry, "python_sha256": hashlib.sha256(python_code.encode()).hexdigest(),
                    "go_sha256": hashlib.sha256(go_code.encode()).hexdigest(),
                    "input_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
                    "input_cases": len(cases), "invocations_at_long_point": n,
                    "repetitions": repetitions, "cpu_idle_reference_w": idle_w,
                    "idle_reference_method": "per_pair_before_between_after",
                    "cpu_idle_reference_initial_w": idle_w,
                    "environment": {**environment, "timeout_docker_seconds": self.timeout_seconds},
                    "summary": summary, "energy_variability": energy_variability,
                    "energy_gain_percent": gains,
                    "limitations": [
                        "RAPL misura il package intero, non il processo; baseline idle locale 1/N sottratta come stima e registrata per ripetizione.",
                        "Scaphandre attribuisce potenza al container; la stima dell'intera finestra estrapola la potenza media osservata sui tratti non coperti.",
                        "Scaphandre conserva anche i joule realmente integrati e la frazione temporale non osservata; la stima resta modellistica.",
                        "I/O JSON, harness e docker exec sono inclusi, non la creazione del container o compilazione Go.",
                        "La stima a due punti assume startup comparabile tra i due esperimenti.",
                    ],
                }
            finally:
                for name in names.values():
                    try:
                        self._execute([self.docker, "rm", "-f", name], timeout=12)
                    except RuntimeError:
                        pass
