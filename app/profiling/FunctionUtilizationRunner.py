"""Confronta l'utilizzo CPU/RAM Python e Go senza coinvolgere i sensori energetici."""

import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
import uuid
from pathlib import Path

from app.profiling.FunctionEnergyRunner import FunctionEnergyRunner
from app.profiling.UtilizationProfiler import CgroupV2Reader, UtilizationSampler, calculate_utilization


class FunctionUtilizationRunner(FunctionEnergyRunner):
    """Riusa gli stessi container, input e harness della pipeline energetica v6.

    Non chiama mai preflight/measure/idle_power di RAPL o Scaphandre.
    Il riferimento al runner energetico serve soltanto a evitare duplicazione
    delle operazioni Docker già validate.
    """

    def __init__(self, *, docker: str = "docker", python_image: str = "python:3.11-alpine",
                 build_image: str = "golang:1.25-alpine", go_image: str = "alpine:3.22",
                 timeout_seconds: float = 90.0, progress=None,
                 sampling_interval_seconds: float = 0.1,
                 go_gomaxprocs: int | None = 2,
                 go_modules_enabled: bool = False, go_cgo_enabled: bool = False) -> None:
        super().__init__(rapl=None, docker=docker, python_image=python_image,
                         build_image=build_image, go_image=go_image,
                         scaphandre_url=None, timeout_seconds=timeout_seconds,
                         progress=progress, go_modules_enabled=go_modules_enabled,
                         go_cgo_enabled=go_cgo_enabled)
        if not math.isfinite(sampling_interval_seconds) or not 0.05 <= sampling_interval_seconds <= 5:
            raise ValueError("Intervallo di campionamento non valido")
        if go_gomaxprocs is not None and (
                type(go_gomaxprocs) is not int or not 1 <= go_gomaxprocs <= 256):
            raise ValueError("GOMAXPROCS Go deve essere un intero fra 1 e 256")
        self.sampling_interval_seconds = sampling_interval_seconds
        self.go_gomaxprocs = go_gomaxprocs

    def _container_flags(self, name: str, directory: Path, *, writable: bool) -> list[str]:
        """Applica GOMAXPROCS solo al runtime Go, mai a Python o alla compilazione.

        Docker exec eredita l'ambiente del container: ogni invocazione Go nella
        prova (preflight, calibrazione e misura) usa quindi lo stesso valore.
        Senza override resta attivo il comportamento automatico di Go 1.25.
        """
        flags = super()._container_flags(name, directory, writable=writable)
        if (self.go_gomaxprocs is not None and not writable
                and name.startswith("function-util-go-")):
            flags.extend(["--env", f"GOMAXPROCS={self.go_gomaxprocs}"])
        return flags

    def preflight(self) -> dict:
        """Verifica Docker/cgroups v2 senza richiedere sudo né RAPL."""
        if os.getuid() == 0:
            raise RuntimeError("Avviare il profiler come utente host non root")
        if shutil.which(self.docker) is None:
            raise RuntimeError("Docker non disponibile nel PATH")
        if not Path("/sys/fs/cgroup/cgroup.controllers").exists():
            raise RuntimeError("Linux cgroups v2 non disponibile; il profiler non supporta cgroups v1")
        for image in (self.python_image, self.build_image, self.go_image):
            result = self._execute([self.docker, "image", "inspect", image])
            if result.returncode:
                raise RuntimeError(f"Immagine Docker mancante: {image}")
        info = self._execute([self.docker, "info", "--format", "{{.ServerVersion}}"])
        if info.returncode:
            raise RuntimeError("Docker daemon non accessibile")
        return {
            "docker_version": info.stdout.decode("utf-8", "replace").strip(),
            "cgroup_version": "v2",
            "images": [self.python_image, self.build_image, self.go_image],
            "docker_limits": {"cpus": 1, "memory": "1g", "memory_swap": "1g"},
            "go_runtime": {
                "gomaxprocs_mode": "automatic" if self.go_gomaxprocs is None else "explicit",
                "gomaxprocs_override": self.go_gomaxprocs,
            },
        }

    def _reader(self, name: str) -> CgroupV2Reader:
        """Ottiene il cgroup dal PID host del container, non dal nome della funzione."""
        result = self._execute([self.docker, "inspect", "--format", "{{.State.Pid}}", name])
        if result.returncode:
            raise RuntimeError("Impossibile determinare il PID host del container")
        raw = result.stdout.decode("ascii", "replace").strip()
        if not re.fullmatch(r"[0-9]+", raw):
            raise RuntimeError("Docker inspect ha restituito un PID non valido")
        reader = CgroupV2Reader.from_host_pid(int(raw))
        cores = reader.limits["cpu_quota_cores"]
        limit = reader.limits["memory_limit_bytes"]
        if cores is None or abs(cores - 1.0) > 0.001:
            raise RuntimeError(f"Limite --cpus=1 non applicato dal cgroup: {cores}")
        if limit is None or limit != 1024 ** 3:
            raise RuntimeError(f"Limite Docker --memory=1g non applicato: {limit}")
        return reader

    def _profile_run(self, language: str, name: str, folder: Path,
                     cases: list[list], n: int, entry: str,
                     reader: CgroupV2Reader) -> dict:
        """Contatori kernel delimitano la stessa chiamata; RAM campionata a parte."""
        path = self._stream(folder, cases, n)
        sampler = UtilizationSampler(reader, interval_seconds=self.sampling_interval_seconds)
        sampler.start()
        try:
            # Contatori CPU il più vicino possibile alla finestra docker exec.
            before = reader.snapshot()
            elapsed, _ = self._call(name, language, path, entry=entry)
            after = reader.snapshot()
        finally:
            sampled = sampler.stop()
        # I contatori CPU sono differenze monotone del cgroup, non valori istantanei.
        return calculate_utilization(before, after, sampled, elapsed, n, reader.limits)

    def run(self, *, python_source: Path, go_source: Path,
            entry: str, cases: list[list], invocations: int = 1000,
            max_invocations: int = 1000000, repetitions: int = 5,
            min_long_seconds: float = 3.0,
            validation_suite=None, validation_case_id: str | None = None) -> dict:

        """Misura N invocazioni ripetute nei medesimi container a 1 CPU / 1 GiB."""

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
            raise ValueError("Sorgente Python o Go vuoto")
        harness = self._validation_harness.read_text(encoding="utf-8")
        if harness.count(self._marker) != 1:
            raise RuntimeError("Marker GoHarness inatteso")
        environment = self.preflight()
        environment["go_runtime"] = {
            "gomaxprocs_mode": "automatic" if self.go_gomaxprocs is None else "explicit",
            "gomaxprocs_override": self.go_gomaxprocs,
        }
        names = {language: "function-util-" + language + "-" + uuid.uuid4().hex[:10]
                 for language in ("python", "go")}
        with tempfile.TemporaryDirectory(prefix="function_utilization_") as work:
            root = Path(work)
            folders = {language: root / language for language in names}
            for folder in folders.values():
                folder.mkdir()
            (folders["python"] / "function.py").write_text(python_code, encoding="utf-8")
            (folders["python"] / "PythonEnergyHarness.py").write_text(
                self._python_harness.read_text(encoding="utf-8"), encoding="utf-8")
            (folders["go"] / "function.go").write_text(go_code, encoding="utf-8")
            (folders["go"] / "harness.go").write_text(
                harness.replace(self._marker, entry), encoding="utf-8")
            try:
                self._notify("Preparazione: compilazione Go fuori dal benchmark")
                self._compile_go(folders["go"])
                self._notify("Preparazione: container Python e Go a 1 CPU / 1 GiB")
                for language, image in (("python", self.python_image), ("go", self.go_image)):
                    self._launch(names[language], folders[language], image)
                self._notify("Preflight: controllo equivalenza Python/Go")
                values = {}
                for language in names:
                    path = self._stream(folders[language], cases, 1)
                    _, values[language] = self._call(
                        names[language], language, path, entry=entry, capture=True)
                preflight_method = self._check_profile_preflight(
                    values["python"]["value"], values["go"]["value"],
                    validation_suite=validation_suite, validation_case_id=validation_case_id,
                    cases=cases)
                self._notify(f"Preflight superato: {preflight_method}")
                readers = {language: self._reader(names[language]) for language in names}
                n = invocations
                while True:
                    times = []
                    for language in ("python", "go"):
                        path = self._stream(folders[language], cases, n)
                        seconds, _ = self._call(names[language], language, path, entry=entry)
                        times.append(seconds)
                    self._notify(f"Calibrazione N={n}: Python={times[0]:.3f} s; Go={times[1]:.3f} s")
                    if min(times) >= min_long_seconds or n >= max_invocations:
                        break
                    n = min(n * 10, max_invocations)
                self._notify(f"Inizio misure cgroups v2: N={n}, ripetizioni={repetitions}")
                results = {language: [] for language in names}
                for repetition in range(repetitions):
                    order = ("python", "go") if repetition % 2 == 0 else ("go", "python")
                    for language in order:
                        self._notify(f"Ripetizione {repetition + 1}/{repetitions}: {language}")
                        data = self._profile_run(
                            language, names[language], folders[language], cases,
                            n, entry, readers[language])
                        results[language].append({"repetition": repetition + 1, **data})
                        self._notify(
                            f"{language}: CPU={data['cpu_utilization_of_quota_percent']:.1f}% quota, "
                            f"CPU-s/invocazione={data['cpu_seconds_per_invocation']:.6f}, "
                            f"memoria picco campionato={data['memory_current_peak_sampled_bytes']/1048576:.1f} MiB")
                summary = {}
                metrics = (
                    "cpu_seconds_per_invocation", "cpu_utilization_of_quota_percent",
                    "memory_current_peak_sampled_bytes", "memory_anon_peak_sampled_bytes",
                    "memory_current_peak_above_baseline_bytes", "wall_seconds_per_invocation",
                )
                for language, records in results.items():
                    stats = {}
                    for metric in metrics:
                        numbers = [row[metric] for row in records if row[metric] is not None]
                        stats[metric] = statistics.median(numbers) if numbers else None
                        stats[metric + "_std"] = statistics.stdev(numbers) if len(numbers) >= 2 else None
                    summary[language] = {**stats, "repetitions": records}
                gain = {}
                for metric in ("cpu_seconds_per_invocation", "memory_current_peak_sampled_bytes",
                               "memory_anon_peak_sampled_bytes", "wall_seconds_per_invocation"):
                    p, g = summary["python"][metric], summary["go"][metric]
                    gain[metric] = 100 * (p - g) / p if p is not None and g is not None and p > 0 else None
                return {
                    "schema_version": 1,
                    "kind": "local_container_function_cpu_memory_utilization_comparison",
                    "method": "cgroups_v2_cpu_counters_and_sampled_memory",
                    "definition": "JSON-lines invocation incl. runtime/harness/JSON; one long N stream; containers prestarted",
                    "execution_protocol": {
                        "container_start_inside_measurement_window": False,
                        "go_compilation_inside_measurement_window": False,
                        "preflight_and_calibration_inside_measurement_window": False,
                        "docker_exec_starts_new_process_for_each_E1_and_EN": True,
                        "module_initialization_once_per_process": True,
                        "N_calls_share_one_process": True,
                        "application_state_reset_between_calls_within_N": False,
                        "json_decode_and_encode_per_invocation": True,
                        "inputs_cycle_over_test_cases": True,
                        "note": "Ogni E1/EN include startup processo/harness; il container e' gia' avviato.",
                    },
                    "entry_point": entry,
                    "python_sha256": hashlib.sha256(python_code.encode()).hexdigest(),
                    "go_sha256": hashlib.sha256(go_code.encode()).hexdigest(),
                    "input_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
                    "input_cases": len(cases),
                    "invocations_at_long_point": n, "repetitions": repetitions,
                    "memory_sampling_interval_seconds": self.sampling_interval_seconds,
                    "environment": {**environment, "timeout_docker_seconds": self.timeout_seconds},
                    "summary": summary, "gain_percent": gain,
                    "limitations": [
                        "CPU misurata da cpu.stat per l'intero container, non soltanto dal corpo funzione.",
                        "CPU% e' rispetto alla quota Docker di un core; CPU-s per invocazione misura il lavoro CPU svolto.",
                        "memory.current comprende anon, file cache e memoria del kernel attribuita al cgroup, non e' RSS del solo processo.",
                        "I picchi nel tratto N sono massimi campionati: allocazioni piu' brevi dell'intervallo potrebbero sfuggire.",
                        "memory.peak e' riportato dalla creazione del container: include preflight/calibrazione e ripetizioni precedenti.",
                        "Gli input sono preparati prima della misurazione; startup docker exec e JSON runtime restano nella finestra.",
                        "CPU/RAM vengono rilevate in una prova separata da RAPL e Scaphandre.",
                    ],
                }
            finally:
                for name in names.values():
                    try:
                        self._execute([self.docker, "rm", "-f", name], timeout=12)
                    except RuntimeError:
                        pass
