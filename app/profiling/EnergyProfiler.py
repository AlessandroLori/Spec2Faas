"""Misura RAPL per esperimenti 1/N e osserva l'attribuzione Scaphandre."""

import math
import re
import statistics
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.profiling.EnergyVariability import sample_statistics
from app.profiling.HardwareProbe import RaplProbe


@dataclass(frozen=True)
class EnergySample:
    """Energia e tempo della finestra delimitata da due letture RAPL."""

    joules: float
    elapsed_s: float


class RaplMeter:
    """Legge un solo package RAPL, serve sudo per il comando cat."""

    def __init__(self, *, access: str = "sudo-cat", root: Path = Path("/sys/class/powercap"),
                 reader: Callable[[Path], int] | None = None) -> None:
        if access not in {"direct", "sudo-cat"}:
            raise ValueError("L'accesso RAPL deve essere direct oppure sudo-cat")
        self.access = access
        self.root = Path(root)
        self._reader = reader
        self.zone, self.maximum = self._select_package()

    def _select_package(self) -> tuple[Path, int]:
        """Preferisce MSR rispetto all'alias MMIO dello stesso package."""
        zones = []
        probe = RaplProbe(self.root)
        for directory in probe._zones():
            try:
                name = (directory / "name").read_text(encoding="utf-8").strip()
                maximum = int((directory / "max_energy_range_uj").read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue
            if name.startswith("package-") and maximum > 0:
                zones.append((directory, maximum))
        if not zones:
            raise RuntimeError("Nessun package RAPL disponibile nell'host")
        preferred = [item for item in zones if re.fullmatch(r"intel-rapl:\d+", item[0].name)]
        # La profilazione RAPL misura un solo package fisico: niente somme implicite.
        chosen_pool = preferred or zones
        if len(chosen_pool) != 1:
            raise RuntimeError("Più package RAPL rilevati: configurazione multi-socket non ancora supportata")
        return chosen_pool[0]

    def _read(self) -> int:
        """Legge il contatore attraverso un path fisso e una chiamata priva di shell."""
        path = self.zone / "energy_uj"
        if self._reader is not None:
            value = self._reader(path)
        elif self.access == "direct":
            try:
                value = int(path.read_text(encoding="utf-8").strip())
            except PermissionError as exc:
                raise RuntimeError("RAPL non leggibile: scegliere sudo-cat o configurare l'accesso") from exc
        else:
            # Non passiamo mai comandi o percorsi arbitrari dal codice delle funzioni a sudo.
            command = ["sudo", "-n", "cat", str(path)]
            try:
                result = subprocess.run(command, capture_output=True, text=True,
                                        timeout=5, check=False)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError("Lettura sudo RAPL non disponibile") from exc
            if result.returncode:
                raise RuntimeError("RAPL richiede sudo: eseguire 'sudo -v' nel terminale; "
                                   + result.stderr.strip()[:300])
            try:
                value = int(result.stdout.strip())
            except ValueError as exc:
                raise RuntimeError("Contatore RAPL non numerico") from exc
        if not isinstance(value, int) or not 0 <= value < self.maximum:
            raise RuntimeError("Valore RAPL fuori dal range del contatore")
        return value

    def preflight(self) -> dict:
        """Verifica l'accesso effettivo prima di creare container."""
        self._read()
        return {"source": str(self.zone / "energy_uj"), "access": self.access,
                "max_energy_range_uj": self.maximum, "scope": "cpu_package"}

    def delta(self, start: int, end: int) -> float:
        """Calcola joule, considerando al massimo un rollover del contatore."""
        difference = end - start if end >= start else self.maximum - start + end
        return difference / 1_000_000.0

    def measure(self, action: Callable[[], object]) -> tuple[EnergySample, object]:
        """Delimita la finestra del processo; propaga eventuali errori del workload."""
        first = self._read()
        t0 = time.monotonic()
        result = action()
        t1 = time.monotonic()
        last = self._read()
        return EnergySample(joules=self.delta(first, last), elapsed_s=t1 - t0), result

    def idle_power(self, seconds: float = 1.5) -> float:
        """Stima la potenza del package a riposo con i container già avviati."""
        if not math.isfinite(seconds) or seconds < 0.25:
            raise ValueError("La finestra idle deve durare almeno 0,25 secondi")
        reading, _ = self.measure(lambda: time.sleep(seconds))
        return reading.joules / reading.elapsed_s


class ScaphandreSampler:
    """Campiona le metriche Prometheus del container senza avviare servizi root."""

    _metric = "scaph_process_power_consumption_microwatts"

    def __init__(self, url: str, container_id: str, name: str,
                 *, interval: float = 2.5,
                 host_pids: Callable[[], set[int]] | None = None,
                 exclude_pids: set[int] | None = None) -> None:
        if not url.startswith(("http://127.0.0.1:", "http://localhost:", "http://[::1]:")):
            raise ValueError("Scaphandre deve essere esposto esclusivamente sulla loopback")
        if not re.fullmatch(r"[a-f0-9]{64}", container_id):
            raise ValueError("ID Docker completo non valido")
        # Scaphandre 1.0.2: il refresh Prometheus può bloccarsi quando
        # /metrics viene interrogato più frequentemente di ogni 2 secondi.
        if not math.isfinite(interval) or interval < 2.2:
            raise ValueError("Scaphandre 1.0.2 richiede almeno 2,2 s tra gli scrape")
        self.url = url
        self.container_id = container_id
        self.name = name
        self.interval = interval
        self.host_pids = host_pids
        # Il processo `sleep` esistente prima della prova non è la funzione.
        self.exclude_pids = set(exclude_pids or ())
        self.identity_modes: set[str] = set()
        self.samples: list[tuple[float, float | None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors: list[str] = []
        # Diagnostica aggregata, senza salvare tutte le metriche dell'host.
        self.pid_visible_samples = 0
        self.pid_stable_samples = 0
        self.matched_power_samples = 0
        self.positive_power_samples = 0
        self.missing_process_metric_samples = 0
        self.no_workload_pid_samples = 0
        self.workload_pids_seen: set[int] = set()
        self.started_at: float | None = None
        self._previous_host_energy_uj: float | None = None
        self.host_energy_changed_samples = 0
        self.host_energy_unchanged_samples = 0
        self.host_energy_unavailable_samples = 0

    @classmethod
    def _identified_power(cls, response: str, container_id: str, name: str,
                          host_pids: set[int] | None = None) -> tuple[float | None, str | None]:
        """Attribuisce soltanto metriche con identità certa: label o PID Docker.

        Scaphandre 1.0.2 può omettere le etichette Docker su cgroups v2.
        In tal caso intersechiamo i PID *host* di `docker top` con `pid` di
        Scaphandre. Un'eseguibile con lo stesso nome non è una prova d'identità.
        """
        total = 0.0
        matched: set[int | str] = set()
        modes: set[str] = set()
        for line_index, line in enumerate(response.splitlines()):
            if not line.startswith(cls._metric + "{"):
                continue
            labels, sep, suffix = line.partition("}")
            if not sep:
                continue
            match = re.search(r'pid="(\d+)"', labels)
            pid = int(match.group(1)) if match else None
            ids = re.findall(r'container_id="([a-f0-9]+)"', labels)
            names = re.findall(r'container_names="([^"\\]*)"', labels)
            has_container_labels = bool(ids or names)
            by_label = (any(len(cid) >= 12 and container_id.startswith(cid) for cid in ids)
                        or any(name == item.strip("/") for item in names))
            # Se Scaphandre dichiara un altro container, non forziamo un match.
            by_pid = (not has_container_labels and pid is not None
                      and host_pids is not None and pid in host_pids)
            identity_key = pid if pid is not None else f"unknown_pid_line_{line_index}"
            if not (by_label or by_pid) or identity_key in matched:
                continue
            try:
                power_uw = float(suffix.strip().split()[0])
            except (ValueError, IndexError):
                continue
            if not math.isfinite(power_uw) or power_uw < 0:
                continue
            matched.add(identity_key)
            modes.add("scaphandre_labels" if by_label else "docker_host_pids")
            total += power_uw / 1_000_000.0
        return (total, "+".join(sorted(modes))) if matched else (None, None)

    @classmethod
    def parse_power(cls, response: str, container_id: str, name: str,
                    host_pids: set[int] | None = None) -> float | None:
        """Conserva l'interfaccia semplice; senza PID richiede etichette Docker."""
        return cls._identified_power(response, container_id, name, host_pids)[0]

    def _observe_host_energy(self, metrics: str) -> None:
        """Rileva contatori host fermi senza attribuire energia ad altri processi."""
        match = re.search(r"(?m)^scaph_host_energy_microjoules\s+([0-9.eE+\-]+)\s*$", metrics)
        if match is None:
            self.host_energy_unavailable_samples += 1
            return
        try:
            value = float(match.group(1))
        except ValueError:
            self.host_energy_unavailable_samples += 1
            return
        if not math.isfinite(value) or value < 0:
            self.host_energy_unavailable_samples += 1
            return
        if self._previous_host_energy_uj is not None:
            if value == self._previous_host_energy_uj:
                self.host_energy_unchanged_samples += 1
            else:
                self.host_energy_changed_samples += 1
        self._previous_host_energy_uj = value

    def _collect(self) -> None:
        """Associa al campione solo PID del container stabili durante lo scrape.

        Docker viene interrogato *prima e dopo* l'HTTP: in questo modo
        l'identità del workload è controllata nello stesso intervallo delle
        metriche, non solo dopo lo scrape. Il polling rimane esterno a RAPL.
        """
        while not self._stop.is_set():
            try:
                before = (set(self.host_pids()) - self.exclude_pids
                          if self.host_pids is not None else None)
                with urllib.request.urlopen(self.url, timeout=4.0) as response:
                    raw = response.read().decode("utf-8", "replace")
                after = (set(self.host_pids()) - self.exclude_pids
                         if self.host_pids is not None else None)
                self._observe_host_energy(raw)
                # Il processo deve appartenere al container in entrambi gli
                # snapshot per evitare attribuzioni di PID riutilizzati.
                active_pids = (before & after if before is not None and after is not None
                               else None)
                if before is not None:
                    if before or after:
                        self.pid_visible_samples += 1
                        self.workload_pids_seen.update(before)
                        self.workload_pids_seen.update(after)
                    else:
                        self.no_workload_pid_samples += 1
                    if active_pids:
                        self.pid_stable_samples += 1
                value, mode = self._identified_power(raw, self.container_id,
                                                     self.name, active_pids)
                if mode:
                    self.identity_modes.add(mode)
                    self.matched_power_samples += 1
                    if value is not None and value > 0:
                        self.positive_power_samples += 1
                elif active_pids:
                    self.missing_process_metric_samples += 1
            except (OSError, TimeoutError, ValueError, RuntimeError) as exc:
                value = None
                if len(self.errors) < 5:
                    self.errors.append(type(exc).__name__)
            # Non usare misure terminate dopo la fine della finestra del workload.
            if self._stop.is_set():
                break
            # Timestamp dopo l'acquisizione: non prima di una chiamata HTTP
            # che può richiedere una frazione significativa di secondo.
            self.samples.append((time.monotonic(), value))
            self._stop.wait(self.interval)

    def start(self) -> None:
        """Inizia il campionamento prima del tratto N del benchmark."""
        self.started_at = time.monotonic()
        self._stop.clear()
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        """Termina il monitor e calcola l'energia solo con copertura sufficiente."""
        stopped_at = time.monotonic()
        self._stop.set()
        if self._thread:
            # HTTP (4 s) e i due docker top (5 s l'uno) possono essere lenti.
            # Non leggere liste e statistiche mentre il sampler è in attività.
            self._thread.join(timeout=16)
            if self._thread.is_alive():
                self.errors.append("sampler_thread_not_stopped")
                return {"status": "sampler_thread_not_stopped", "attributed_j": None,
                        "sample_count": len(self.samples), "covered_seconds": 0.0,
                        "coverage_fraction": 0.0, "errors": list(self.errors),
                        "identification_methods": sorted(self.identity_modes),
                        "diagnostics": {},
                        "note": "La raccolta non si è arrestata entro il timeout."}
        records = self.samples
        # Potenza attribuita osservata a intervalli Scaphandre: descrive
        # la variabilita' *temporale* nella finestra EN, non l'incertezza
        # della stima energetica. Non richiede polling o processi aggiuntivi.
        power_stats = sample_statistics([power for _, power in records])
        integrated = 0.0
        covered = 0.0
        for (first_t, first_w), (last_t, last_w) in zip(records, records[1:]):
            if first_w is not None and last_w is not None and last_t > first_t:
                span = last_t - first_t
                covered += span
                integrated += (first_w + last_w) * span / 2
        # Il tratto campionato e la finestra completa sono grandezze distinte.
        # I test unitari che usano campioni sintetici senza start() conservano
        # come riferimento il tratto primo-ultimo campione.
        total = (stopped_at - self.started_at if self.started_at is not None
                 else records[-1][0] - records[0][0] if len(records) >= 2 else 0.0)
        fraction = min(1.0, covered / total) if total > 0 else 0.0
        valid = len(records) >= 3 and fraction >= 0.8 and covered >= 1.0
        # Stimiamo i margini non osservati usando la potenza media della
        # porzione effettivamente campionata. E' un'estrapolazione dichiarata,
        # NON una misura fisica ai bordi.
        observed_mean_power_w = integrated / covered if covered > 0 else None
        whole_window_estimate_j = (observed_mean_power_w * total
                                   if valid and observed_mean_power_w is not None else None)
        if valid:
            status = "estimated_process_attribution"
        elif self.host_pids is not None and not self.pid_visible_samples and not self.identity_modes:
            status = "workload_pid_not_observed"
        elif (self.pid_stable_samples and not self.matched_power_samples
              and not self.identity_modes):
            status = "process_power_metric_not_matched"
        else:
            status = "insufficient_coverage"
        return {
            "status": status,
            # Per compatibilita' il campo attribuito e' ora la stima dell'intera finestra.
            # Il dato realmente integrato e' separato, mai scalato implicitamente.
            "attributed_j": whole_window_estimate_j,
            "estimated_full_window_j": whole_window_estimate_j,
            "observed_attributed_j": integrated if covered > 0 else None,
            "observed_mean_power_w": observed_mean_power_w,
            "full_window_seconds": total,
            "uncovered_seconds": max(0.0, total - covered),
            "uncovered_fraction": 1.0 - fraction,
            "estimation_method": ("mean_observed_power_extrapolated_to_full_window"
                                  if valid else None),
            "sample_count": len(records), "covered_seconds": covered,
            "power_sample_statistics_w": power_stats,
            "coverage_fraction": fraction, "errors": list(self.errors),
            "identification_methods": sorted(self.identity_modes),
            "diagnostics": {
                "workload_pid_visible_samples": self.pid_visible_samples,
                "workload_pid_stable_samples": self.pid_stable_samples,
                "process_power_matched_samples": self.matched_power_samples,
                "process_power_positive_samples": self.positive_power_samples,
                "process_power_missing_samples": self.missing_process_metric_samples,
                "no_workload_pid_samples": self.no_workload_pid_samples,
                "distinct_workload_pids": len(self.workload_pids_seen),
                "scaphandre_host_energy_changed_samples": self.host_energy_changed_samples,
                "scaphandre_host_energy_unchanged_samples": self.host_energy_unchanged_samples,
                "scaphandre_host_energy_unavailable_samples": self.host_energy_unavailable_samples,
                "scaphandre_scrape_interval_seconds": self.interval,
            },
            "note": ("Scaphandre attribuisce potenza ai processi. L'energia 'observed' copre "
                     "solo gli intervalli validi; la stima 'full_window' assume "
                     "che la potenza media osservata rappresenti i margini non campionati."),
        }


def two_point(first: float, long: float, n: int) -> float | None:
    """Stima il costo per invocazione senza presentare differenze negative come zero."""
    if n < 2 or not all(math.isfinite(value) for value in (first, long)):
        raise ValueError("Punti invalidi per la differenza a due punti")
    result = (long - first) / (n - 1)
    return result if result > 0 else None


def median_or_none(values: list[float | None]) -> float | None:
    """Aggrega esclusivamente ripetizioni effettivamente risolte."""
    present = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.median(present) if present else None
