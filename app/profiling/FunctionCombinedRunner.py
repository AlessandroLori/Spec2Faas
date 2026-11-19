"""Profilazione congiunta CPU/RAM, RAPL e Scaphandre su uno stesso stream N."""

import hashlib
import json
import math
import statistics
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path

from app.profiling.EnergyProfiler import RaplMeter, ScaphandreSampler, median_or_none, two_point
from app.profiling.EnergyVariability import sample_statistics, summarize_energy_variability
from app.profiling.FunctionUtilizationRunner import FunctionUtilizationRunner
from app.profiling.UtilizationProfiler import UtilizationSampler, calculate_utilization


class FunctionCombinedRunner(FunctionUtilizationRunner):
    """Conserva le formule validate e misura ogni N una sola volta per replica."""

    MAX_INVOCATIONS = 10_000_000  # Limite per common/per-language (stream JSONL esplicito).
    MAX_TIMER_INVOCATIONS = 1_000_000_000_000  # Nel timer e' solo una salvaguardia.

    def __init__(self, *, rapl: RaplMeter, scaphandre_url: str,
                 sampling_interval_seconds: float = 0.1, timeout_seconds: float = 180.0,
                 go_gomaxprocs: int | None = 2, progress=None, **kwargs) -> None:
        super().__init__(timeout_seconds=timeout_seconds,
                         sampling_interval_seconds=sampling_interval_seconds,
                         go_gomaxprocs=go_gomaxprocs, progress=progress, **kwargs)
        if rapl is None:
            raise ValueError("RAPL e' obbligatorio per la profilazione congiunta")
        if not scaphandre_url:
            raise ValueError("Scaphandre e' obbligatorio per la profilazione congiunta")
        self.rapl = rapl
        self.scaphandre_url = scaphandre_url

    def _scaphandre_preflight(self) -> None:
        """Non avviare campagne se le metriche di processo non sono disponibili."""
        for attempt in range(3):
            try:
                with urllib.request.urlopen(self.scaphandre_url, timeout=5) as response:
                    data = response.read().decode("utf-8", "replace")
            except (OSError, ValueError) as exc:
                raise RuntimeError("Scaphandre Prometheus non raggiungibile") from exc
            if "scaph_process_power_consumption_microwatts" in data:
                return
            if attempt < 2:
                time.sleep(2.5)
        raise RuntimeError("Metriche Scaphandre dei processi assenti dopo 3 tentativi")

    def _measure_combined_long(self, language: str, name: str, directory: Path,
                               cases: list[list], n: int, entry: str,
                               container_id: str, reader, *,
                               timer_seconds: float | None = None) -> dict:
        """Una sola chiamata docker exec; i tre osservatori la delimitano insieme."""
        path = self._stream(directory, cases, len(cases) if timer_seconds is not None else n)
        idle_pids = self._docker_host_pids(name)
        scaph = ScaphandreSampler(
            self.scaphandre_url, container_id, name,
            host_pids=lambda: self._docker_host_pids(name), exclude_pids=idle_pids,
        )
        memory = UtilizationSampler(reader, interval_seconds=self.sampling_interval_seconds)
        scaph_started = False
        memory_started = False
        completed = False
        timer_metadata: dict = {}
        def execute_workload():
            if timer_seconds is None:
                self._call(name, language, path, entry=entry)
            else:
                timer_metadata.update(self._call_timer(
                    name, language, path, entry=entry, seconds=timer_seconds,
                    max_invocations=n))

        try:
            scaph.start()
            scaph_started = True
            memory.start()
            memory_started = True
            before = reader.snapshot()
            energy, _ = self.rapl.measure(execute_workload)
            after = reader.snapshot()
            completed = True
        finally:
            try:
                sampled = memory.stop() if memory_started else None
            finally:
                attributed = scaph.stop() if scaph_started else None
        if not completed:
            raise RuntimeError("Misurazione congiunta non completata")
        actual_n = timer_metadata["invocations"] if timer_seconds is not None else n
        utilization = calculate_utilization(before, after, sampled, energy.elapsed_s,
                                            actual_n, reader.limits)
        diagnostics = attributed.get("diagnostics", {})
        self._notify(
            f"Scaphandre {language}: {attributed['status']}; "
            f"campioni associati={diagnostics.get('process_power_matched_samples', 0)}, "
            f"copertura={attributed.get('coverage_fraction', 0):.1%}"
        )
        return {
            "invocations": actual_n,
            "timer": timer_metadata if timer_seconds is not None else None,
            "elapsed_seconds": energy.elapsed_s,
            "rapl_package_gross_j": energy.joules,
            "rapl_package_incremental_estimate_j": None,  # Assegnato dopo il riferimento idle.
            "scaphandre": attributed,
            "utilization": utilization,
        }

    def _paired_combined(self, language: str, name: str, directory: Path,
                         cases: list[list], n: int, entry: str,
                         container_id: str, reader, *,
                         timer_seconds: float | None = None) -> tuple[dict, dict, dict]:
        """Mantiene idle-E1-idle-EN-idle; EN include tutti i sensori."""
        before = self.rapl.idle_power()
        short = self._measure_run(language, name, directory, cases, 1, entry,
                                  before, container_id)
        middle = self.rapl.idle_power()
        if timer_seconds is None:
            long = self._measure_combined_long(language, name, directory, cases, n,
                                               entry, container_id, reader)
        else:
            long = self._measure_combined_long(
                language, name, directory, cases, n, entry, container_id, reader,
                timer_seconds=timer_seconds)
        after = self.rapl.idle_power()
        readings = (before, middle, after)
        if not all(math.isfinite(item) and item >= 0 for item in readings):
            raise RuntimeError("Riferimenti idle RAPL non validi")
        one_idle = (before + middle) / 2
        many_idle = (middle + after) / 2
        for point, power in ((short, one_idle), (long, many_idle)):
            point["idle_reference_w"] = power
            point["rapl_package_incremental_estimate_j"] = self._net(
                point["rapl_package_gross_j"], point["elapsed_seconds"], power)
        spread = max(readings) - min(readings)
        unstable = spread > max(3.0, 0.25 * statistics.median(readings))
        baseline = {
            "method": "bracketed_three_idle_windows_per_1_N_pair",
            "before_one_w": before,
            "between_one_and_n_w": middle,
            "after_n_w": after,
            "one_reference_w": one_idle,
            "many_reference_w": many_idle,
            "range_w": spread,
            "idle_three_point_statistics_w": sample_statistics(readings),
            "stability": "needs_validation" if unstable else "nominal",
            "note": "Riferimento idle variabile" if unstable else None,
        }
        return short, long, baseline

    def _calibrate_per_language(self, language: str, name: str, directory: Path,
                                cases: list[list], start_n: int, max_n: int,
                                target_seconds: float, entry: str) -> tuple[int, list[dict]]:
        """Sceglie N con misure reali e una stima progressiva.

        Misura il tempo totale di ciascuno fuori dalle finestre RAPL/Scaphandre.
        """
        n = start_n
        history: list[dict] = []
        while True:
            path = self._stream(directory, cases, n)
            seconds, _ = self._call(name, language, path, entry=entry)
            if not math.isfinite(seconds) or seconds <= 0:
                raise RuntimeError(f"Calibrazione {language}: durata non valida")
            history.append({"language": language, "invocations": n, "seconds": seconds})
            self._notify(f"Calibrazione {language} N={n}: {seconds:.3f} s")
            if seconds >= target_seconds or n >= max_n or target_seconds == 0:
                return n, history
            # Una stima conservativa con margine del 7%, massimo 10x per passo.
            # Non supponiamo che i tempi siano proporzionali a N per N piccoli.
            proposal = math.ceil(n * (target_seconds / seconds) * 1.07)
            n_next = min(max_n, max(n + 1, min(n * 10, proposal)))
            if n_next <= n:
                raise RuntimeError(f"Calibrazione {language}: impossibile progredire")
            n = n_next

    @staticmethod
    def _median_stats(values: list[float | None], minimum: int = 1) -> tuple[float | None, float | None, int]:
        valid = [v for v in values if v is not None]
        return (statistics.median(valid) if len(valid) >= minimum else None,
                statistics.stdev(valid) if len(valid) >= 2 else None,
                len(valid))

    def run(self, *, python_source: Path, go_source: Path, entry: str,
            cases: list[list], invocations: int = 1000,
            max_invocations: int = 1_000_000, repetitions: int = 5,
            min_long_seconds: float = 20.0,
            calibration_mode: str = "common", timer_seconds: float | None = None,
            min_invocations_warning: int = 30,
            validation_suite=None, validation_case_id: str | None = None) -> dict:
        """Calibra N una volta, quindi misura entrambi i linguaggi su una finestra comune."""
        if not self._entry.fullmatch(entry) or entry == "main":
            raise ValueError("Entry point invalido")
        if not isinstance(cases, list) or not cases or any(not isinstance(c, list) for c in cases):
            raise ValueError("I casi devono essere array JSON di argomenti")
        if calibration_mode not in ("common", "per-language", "timer"):
            raise ValueError("Modalita' di calibrazione non valida")
        max_allowed = (self.MAX_TIMER_INVOCATIONS if calibration_mode == "timer"
                       else self.MAX_INVOCATIONS)
        if not 2 <= invocations <= max_invocations <= max_allowed:
            raise ValueError(f"Numero di invocazioni fuori range (massimo {max_allowed})")
        if not 1 <= repetitions <= 20:
            raise ValueError("Ripetizioni fuori range")
        if not math.isfinite(min_long_seconds) or min_long_seconds < 0:
            raise ValueError("Durata minima non valida")
        if type(min_invocations_warning) is not int or min_invocations_warning < 2:
            raise ValueError("Soglia avviso invocazioni non valida (minimo 2)")
        if timer_seconds is not None and calibration_mode != "timer":
            raise ValueError("timer_seconds richiede calibration_mode=timer")
        timer_target = (min_long_seconds if timer_seconds is None else timer_seconds) if calibration_mode == "timer" else None
        if timer_target is not None and (not math.isfinite(timer_target) or timer_target <= 0):
            raise ValueError("Durata Timer non valida")
        if timer_target is not None and self.timeout_seconds is not None and timer_target >= self.timeout_seconds:
            raise ValueError("Timer deve essere inferiore al timeout del processo")
        py_code = python_source.read_text(encoding="utf-8")
        go_code = go_source.read_text(encoding="utf-8")
        if not py_code.strip() or not go_code.strip():
            raise ValueError("Sorgente Python o Go vuoto")
        harness = self._validation_harness.read_text(encoding="utf-8")
        if harness.count(self._marker) != 1:
            raise RuntimeError("Marker GoHarness inatteso")
        environment = self.preflight()
        environment["rapl"] = self.rapl.preflight()
        self._scaphandre_preflight()
        names = {lang: "function-util-" + lang + "-" + uuid.uuid4().hex[:10]
                 for lang in ("python", "go")}
        with tempfile.TemporaryDirectory(prefix="function_combined_") as work:
            root = Path(work)
            folders = {lang: root / lang for lang in names}
            for folder in folders.values():
                folder.mkdir()
            (folders["python"] / "function.py").write_text(py_code, encoding="utf-8")
            (folders["python"] / "PythonEnergyHarness.py").write_text(
                self._python_harness.read_text(encoding="utf-8"), encoding="utf-8")
            (folders["go"] / "function.go").write_text(go_code, encoding="utf-8")
            (folders["go"] / "harness.go").write_text(
                harness.replace(self._marker, entry), encoding="utf-8")
            containers = {}
            try:
                self._notify("Preparazione: compilazione Go (fuori dalle misure)")
                self._compile_go(folders["go"])
                self._notify("Preparazione: avvio container Python/Go a 1 CPU e 1 GiB")
                for language, image in (("python", self.python_image), ("go", self.go_image)):
                    containers[language] = self._launch(names[language], folders[language], image)
                self._notify("Preflight: equivalenza di una chiamata Python/Go")
                checked = {}
                for language in names:
                    path = self._stream(folders[language], cases, 1)
                    _, checked[language] = self._call(names[language], language, path,
                                                      entry=entry, capture=True)
                preflight_method = self._check_profile_preflight(
                    checked["python"]["value"], checked["go"]["value"],
                    validation_suite=validation_suite, validation_case_id=validation_case_id,
                    cases=cases)
                self._notify(f"Preflight superato: {preflight_method}")
                readers = {language: self._reader(names[language]) for language in names}
                n = invocations
                calibration = []
                fixed_n = invocations == max_invocations and calibration_mode != "timer"
                n_by_language = {"python": n, "go": n}
                if calibration_mode == "timer":
                    self._notify(
                        f"Timer: durata minima {timer_target:g} s per linguaggio e replica, "
                        f"cap={max_invocations} chiamate; N effettivo registrato da ciascun harness")
                elif fixed_n:
                    self._notify(f"N={n} fissato: calibrazione lunga saltata")
                elif calibration_mode == "per-language":
                    for language in ("python", "go"):
                        chosen, history = self._calibrate_per_language(
                            language, names[language], folders[language], cases,
                            invocations, max_invocations, min_long_seconds, entry)
                        n_by_language[language] = chosen
                        calibration.extend(history)
                    self._notify(
                        f"Calibrazione indipendente: Python N={n_by_language['python']}, "
                        f"Go N={n_by_language['go']}; ripetizioni={repetitions}")
                else:
                    # Unico N per entrambi i linguaggi: il piu' veloce deve
                    # raggiungere la durata minima. Stimiamo il passo da tempi
                    # effettivi, anziche' moltiplicare sempre N per dieci.
                    # I primi campioni includono uno startup quasi fisso: il
                    # tetto 10x evita stime enormi quando N e' ancora piccolo.
                    while True:
                        measurements = []
                        for language in ("python", "go"):
                            path = self._stream(folders[language], cases, n)
                            duration, _ = self._call(names[language], language, path, entry=entry)
                            if not math.isfinite(duration) or duration <= 0:
                                raise RuntimeError(f"Calibrazione {language}: durata non valida")
                            measurements.append(duration)
                        calibration.append({"invocations": n, "python_seconds": measurements[0],
                                            "go_seconds": measurements[1]})
                        self._notify(f"Calibrazione N={n}: Python={measurements[0]:.3f} s, "
                                     f"Go={measurements[1]:.3f} s")
                        fastest_seconds = min(measurements)
                        if fastest_seconds >= min_long_seconds or n >= max_invocations or min_long_seconds == 0:
                            break
                        # Stessa stima adattiva già utilizzata nella modalità
                        # per-language: rapporto dei tempi + margine del 7%.
                        # Nessuna modifica alla quantita' di lavoro misurata:
                        # il nuovo N comune e' verificato con un altro probe.
                        proposal = math.ceil(n * (min_long_seconds / fastest_seconds) * 1.07)
                        n_next = min(max_invocations, max(n + 1, min(n * 10, proposal)))
                        if n_next <= n:
                            raise RuntimeError("Calibrazione common: impossibile progredire")
                        self._notify(f"Calibrazione common: prossimo N stimato={n_next} "
                                     f"(limite={max_invocations})")
                        n = n_next
                    n_by_language = {"python": n, "go": n}
                if calibration_mode != "timer":
                    self._notify(
                        f"Profilazione congiunta: N python={n_by_language['python']}, "
                        f"N go={n_by_language['go']}, ripetizioni={repetitions}")
                else:
                    self._notify(f"Profilazione Timer: {timer_target:g} s per linguaggio, "
                                 f"ripetizioni={repetitions}")
                records = {lang: [] for lang in names}
                for rep in range(repetitions):
                    order = ("python", "go") if rep % 2 == 0 else ("go", "python")
                    for language in order:
                        self._notify(f"Ripetizione {rep + 1}/{repetitions}: {language} "
                                     f"(E1 + {'Timer' if calibration_mode == 'timer' else 'N'} con tutti i sensori)")
                        if calibration_mode == "timer":
                            short, long, idle = self._paired_combined(
                                language, names[language], folders[language], cases,
                                max_invocations, entry, containers[language], readers[language],
                                timer_seconds=timer_target)
                        else:
                            short, long, idle = self._paired_combined(
                                language, names[language], folders[language], cases,
                                n_by_language[language], entry, containers[language], readers[language])
                        actual_n = long["invocations"]
                        if idle["stability"] != "nominal":
                            self._notify(f"Idle RAPL {language} variabile: {idle['range_w']:.2f} W")
                        scaph = long["scaphandre"]
                        attributed = scaph.get("attributed_j")
                        resources = long["utilization"]
                        record = {
                            "repetition": rep + 1, "one": short, "many": long,
                            "idle_bracketing": idle,
                            "actual_invocations": actual_n,
                            "rapl_incremental_j_per_invocation": (
                                two_point(short["rapl_package_incremental_estimate_j"],
                                          long["rapl_package_incremental_estimate_j"], actual_n)
                                if actual_n >= 2 else None),
                            "rapl_gross_j_per_invocation": (
                                two_point(short["rapl_package_gross_j"],
                                          long["rapl_package_gross_j"], actual_n)
                                if actual_n >= 2 else None),
                            "scaphandre_attributed_j_per_invocation": (
                                attributed / actual_n if attributed is not None else None),
                            "resources": resources,
                        }
                        records[language].append(record)
                        if calibration_mode == "timer":
                            self._notify(
                                f"Timer {language}: invocazioni={actual_n}, "
                                f"tempo ciclo={long['timer']['workload_seconds']:.3f} s, "
                                f"stop={long['timer']['stop_reason']}")
                        if actual_n < min_invocations_warning:
                            self._notify(f"AVVISO: {language} replica {rep+1}, "
                                         f"solo {actual_n} chiamate (<{min_invocations_warning})")
                        self._notify(f"{language}: T={resources['elapsed_seconds']:.3f} s; "
                                     f"CPU={resources['cpu_utilization_of_quota_percent']:.1f}%; "
                                     f"RAM picco={resources['memory_current_peak_sampled_bytes']/1048576:.1f} MiB; "
                                     f"Scaphandre={scaph['status']}")
                summary = {}
                majority = repetitions // 2 + 1
                energy_keys = ("rapl_incremental_j_per_invocation", "rapl_gross_j_per_invocation",
                               "scaphandre_attributed_j_per_invocation")
                resource_keys = ("cpu_seconds_per_invocation", "cpu_utilization_of_quota_percent",
                                 "memory_current_peak_sampled_bytes", "memory_anon_peak_sampled_bytes",
                                 "memory_current_peak_above_baseline_bytes", "wall_seconds_per_invocation",
                                 "cpu_throttled_seconds")
                for language, rows in records.items():
                    stats = {}
                    for key in energy_keys + resource_keys:
                        values = [row[key] if key in row else row["resources"].get(key)
                                  for row in rows]
                        median, std, valid = self._median_stats(
                            values, minimum=majority if key in energy_keys else 1)
                        stats[key] = median
                        stats[key + "_std"] = std
                        if key in energy_keys:
                            stats[key + "_resolved_repetitions"] = valid
                    stats["rapl_idle_drift_flagged_repetitions"] = sum(
                        row["idle_bracketing"]["stability"] != "nominal" for row in rows)
                    stats["repetitions"] = rows
                    summary[language] = stats
                energy_variability = {
                    language: summarize_energy_variability(
                        rows, scaphandre_key="scaphandre")
                    for language, rows in records.items()
                }
                if calibration_mode == "timer":
                    for details in energy_variability.values():
                        details["timer_interpretation"] = (
                            "Il numero di invocazioni EN varia tra repliche; la varianza "
                            "dell'energia EN totale include anche la quantita' di lavoro. "
                            "Per confrontare la ripetibilita' preferire J/invocazione, "
                            "tenendo conto del numero di chiamate registrato per replica.")
                invocation_counts = {language: [row["actual_invocations"] for row in rows]
                                     for language, rows in records.items()}
                quality_warnings = []
                for language, counts in invocation_counts.items():
                    for i, count in enumerate(counts, 1):
                        if count < min_invocations_warning:
                            quality_warnings.append(
                                f"{language}: ripetizione {i} ha solo {count} invocazioni "
                                f"(soglia avviso {min_invocations_warning})")
                        if count < 2:
                            quality_warnings.append(
                                f"{language}: ripetizione {i}, RAPL E1/EN non stimabile per N<2")
                        if calibration_mode == "timer":
                            detail = records[language][i-1]["many"].get("timer") or {}
                            if detail.get("stop_reason") == "invocation_cap_reached":
                                quality_warnings.append(
                                    f"{language}: ripetizione {i}, cap invocazioni raggiunto "
                                    f"prima di {timer_target:g} s")
                gains = {}
                for key in energy_keys + resource_keys:
                    if key in ("cpu_utilization_of_quota_percent", "cpu_throttled_seconds"):
                        continue
                    p, g = summary["python"][key], summary["go"][key]
                    gains[key] = 100 * (p - g) / p if p is not None and g is not None and p > 0 else None
                return {
                    "schema_version": 1,
                    "kind": "local_combined_function_cpu_memory_energy_comparison",
                    "method": ("timer_variable_N_cgroups_v2_RAPL_two_point_scaphandre"
                               if calibration_mode == "timer" else
                               "single_N_stream_cgroups_v2_RAPL_two_point_scaphandre_same_window"),
                    "definition": "La finestra N comprende docker exec, harness e JSON; compila fuori misura",
                    "preflight_comparison_method": preflight_method,
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
                        "timer_input_protocol": (
                            "template_JSON_in_memoria_decodificato_a_ogni_chiamata"
                            if calibration_mode == "timer" else None),
                        "timer_counts_independent_per_replica": calibration_mode == "timer",
                    },
                    "entry_point": entry,
                    "python_sha256": hashlib.sha256(py_code.encode()).hexdigest(),
                    "go_sha256": hashlib.sha256(go_code.encode()).hexdigest(),
                    "input_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
                    "input_cases": len(cases),
                    # In modalita' indipendente non esiste un unico N comune.
                    "invocations_at_long_point": (
                        None if calibration_mode == "timer" else
                        n_by_language["python"] if n_by_language["python"] == n_by_language["go"] else None),
                    "invocations_by_language": (invocation_counts if calibration_mode == "timer"
                                                else n_by_language),
                    "invocations_by_repetition": invocation_counts,
                    "min_invocations_warning": min_invocations_warning,
                    "quality_warnings": quality_warnings,
                    "timer_target_seconds": timer_target,
                    "calibration_mode": calibration_mode,
                    "repetitions": repetitions,
                    "calibration": calibration,
                    "calibration_skipped_fixed_n": fixed_n,
                    "minimum_long_seconds_requested": (
                        timer_target if calibration_mode == "timer" else min_long_seconds),
                    "minimum_long_seconds_met": min(
                        row["many"]["timer"]["workload_seconds"]
                        if calibration_mode == "timer" else row["resources"]["elapsed_seconds"]
                        for language in records for row in records[language]) >= (
                            timer_target if calibration_mode == "timer" else min_long_seconds),
                    "memory_sampling_interval_seconds": self.sampling_interval_seconds,
                    "environment": {**environment, "timeout_docker_seconds": self.timeout_seconds},
                    "summary": summary,
                    "energy_variability": energy_variability,
                    "gain_percent": gains,
                    "energy_gain_percent": {k: gains[k] for k in energy_keys},
                    "limitations": [
                        "RAPL misura il package CPU e include il costo host dei sampler concorrenti; non e' direttamente confrontabile col RAPL v7 senza polling concorrente.",
                        "RAPL incrementale usa idle locali prima/tra/dopo E1/EN: l'idle non sottrae precisamente il costo di polling durante EN.",
                        "Scaphandre stima potenza attribuita al processo e la estrapola sui tratti non coperti: non e' energia fisica misurata dal processo.",
                        "CPU/RAM sono contabilizzate nel cgroup Docker; non misurano solo il corpo funzione.",
                        "Picco RAM e' campionato, non il massimo esatto; memory.peak include l'intera vita del container.",
                        "Un solo stream N per replica/linguaggio per i tre sensori; E1 separato e' richiesto dal metodo RAPL.",
                        "Le attivita' di calibrazione, Docker e Go build sono fuori dalle finestre di misura; e' esclusa l'energia degli agenti LLM.",
                        "Con per-language, Python e Go possono usare N differenti; i risultati sono normalizzati per chiamata, ma non hanno la stessa dimensione di stream.",
                        "Con timer, N varia anche tra le repliche: gli input sono template JSON in memoria, nuovamente deserializzati a ogni chiamata, non uno stream pre-generato da disco.",
                        "Timer interrompe il ciclo soltanto al termine della chiamata: una funzione lenta puo' superare la durata richiesta.",
                        "La soglia minima di invocazioni e' soltanto diagnostica, non certifica significativita' statistica.",
                    ],
                }
            finally:
                for name in names.values():
                    try:
                        self._execute([self.docker, "rm", "-f", name], timeout=12)
                    except RuntimeError:
                        pass
