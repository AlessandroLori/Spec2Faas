"""Misura CPU e memoria dei container mediante contatori Linux cgroups v2."""

import math
import threading
import time
from pathlib import Path
from typing import Callable


def _key_values(path: Path) -> dict[str, int]:
    """Legge i contatori cgroup indipendentemente dall'ordine delle righe."""
    result = {}
    for line in path.read_text(encoding="ascii").splitlines():
        parts = line.split()
        if len(parts) != 2:
            raise RuntimeError(f"Contatore cgroup non valido: {path.name}")
        key, value = parts
        number = int(value)
        if number < 0:
            raise RuntimeError(f"Contatore cgroup negativo: {path.name}/{key}")
        result[key] = number
    return result


def _nonnegative_delta(after: dict, before: dict, key: str) -> int | None:
    """Non inventa valori se un contatore manca o diminuisce."""
    if key not in before or key not in after:
        return None
    change = after[key] - before[key]
    if change < 0:
        raise RuntimeError(f"Contatore cgroup {key} diminuito durante la misura")
    return change


class CgroupV2Reader:
    """Lettura host in sola lettura del cgroup di un container Docker."""

    def __init__(self, path: Path) -> None:
        self.path = path
        for filename in ("cpu.stat", "cpu.max", "memory.current", "memory.max", "memory.stat"):
            if not (self.path / filename).is_file():
                raise RuntimeError(f"Cgroup v2 non leggibile: {self.path / filename}")
        self.limits = self._limits()

    @classmethod
    def from_host_pid(cls, pid: int, *, proc_root: Path = Path("/proc"),
                      cgroup_root: Path = Path("/sys/fs/cgroup")) -> "CgroupV2Reader":

        """Risoluzione protetta del cgroup unified dal PID host di Docker."""

        if not isinstance(pid, int) or pid <= 1:
            raise RuntimeError("PID host del container non valido")
        try:
            memberships = (proc_root / str(pid) / "cgroup").read_text(encoding="ascii")
        except OSError as exc:
            raise RuntimeError("Impossibile leggere /proc/PID/cgroup del container") from exc
        unified = [line.split("::", 1)[1] for line in memberships.splitlines()
                   if line.startswith("0::")]
        if len(unified) != 1 or not unified[0].startswith("/"):
            raise RuntimeError("Docker non espone un cgroup v2 unified per il container")
        # Il PID e il file /proc non sono input della funzione o dell'LLM.
        root = cgroup_root.resolve()
        path = (root / unified[0].lstrip("/")).resolve()
        if path == root or not path.is_relative_to(root):
            raise RuntimeError("Cgroup radice o esterno al filesystem: misurazione rifiutata")
        return cls(path)

    def _limits(self) -> dict:
        quota, period = (self.path / "cpu.max").read_text(encoding="ascii").split()
        if not period.isdigit() or int(period) == 0:
            raise RuntimeError("Periodo CPU cgroup non valido")
        if quota != "max" and not quota.isdigit():
            raise RuntimeError("Quota CPU cgroup non valida")
        cores = None if quota == "max" else int(quota) / int(period)
        memory = (self.path / "memory.max").read_text(encoding="ascii").strip()
        if memory != "max" and not memory.isdigit():
            raise RuntimeError("Limite memoria cgroup non valido")
        return {
            "cpu_quota_cores": cores,
            "memory_limit_bytes": None if memory == "max" else int(memory),
        }

    def snapshot(self) -> dict:
        """Contatori monotoni CPU e stato memoria; memory.peak è dalla creazione."""
        try:
            cpu = _key_values(self.path / "cpu.stat")
            memory_stat = _key_values(self.path / "memory.stat")
            memory_current = int((self.path / "memory.current").read_text(encoding="ascii"))
            peak_path = self.path / "memory.peak"
            peak_since_creation = int(peak_path.read_text(encoding="ascii")) if peak_path.exists() else None
            events_path = (self.path / "memory.events.local")
            if not events_path.is_file():
                events_path = self.path / "memory.events"
            events = _key_values(events_path) if events_path.exists() else {}
        except (OSError, ValueError) as exc:
            raise RuntimeError("Lettura contatori cgroups v2 fallita") from exc
        if "usage_usec" not in cpu or memory_current < 0:
            raise RuntimeError("Contatori CPU/memoria cgroups v2 non disponibili: usage_usec o memory.current")
        if peak_since_creation is not None and peak_since_creation < 0:
            raise RuntimeError("Picco memoria cgroup non valido")
        return {
            "cpu": cpu, "memory_current_bytes": memory_current,
            "memory_anon_bytes": memory_stat.get("anon"),
            "memory_file_bytes": memory_stat.get("file"),
            "memory_peak_since_container_start_bytes": peak_since_creation,
            "memory_events": events,
        }


class UtilizationSampler:
    """Legge il cgroup host; non esegue comandi nel container sotto misura."""

    def __init__(self, reader: CgroupV2Reader, *, interval_seconds: float = 0.1) -> None:
        if not math.isfinite(interval_seconds) or not 0.05 <= interval_seconds <= 5:
            raise ValueError("Intervallo campionamento memoria non valido")
        self.reader = reader
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[tuple[float, dict]] = []
        self._error: Exception | None = None

    def _collect_once(self) -> None:
        reading = self.reader.snapshot()
        self._samples.append((time.monotonic(), reading))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self._collect_once()
            except Exception as exc:
                self._error = exc
                self._stop.set()
                return

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Campionatore già avviato")
        self._collect_once()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        if self._thread is None:
            raise RuntimeError("Campionatore non avviato")
        self._stop.set()
        self._thread.join(timeout=max(5, self.interval_seconds + 1))
        if self._thread.is_alive():
            raise RuntimeError("Thread del campionatore ancora attivo")
        if self._error is not None:
            raise RuntimeError("Campionamento cgroups v2 interrotto") from self._error
        self._collect_once()
        snapshots = [s for _, s in self._samples]
        def max_available(key: str) -> int | None:
            values = [s[key] for s in snapshots if s[key] is not None]
            return max(values) if values else None
        return {
            "sample_count": len(snapshots),
            "sampling_interval_seconds": self.interval_seconds,
            "memory_current_peak_sampled_bytes": max_available("memory_current_bytes"),
            "memory_current_mean_sampled_bytes": sum(s["memory_current_bytes"] for s in snapshots) / len(snapshots),
            "memory_anon_peak_sampled_bytes": max_available("memory_anon_bytes"),
            "memory_file_peak_sampled_bytes": max_available("memory_file_bytes"),
            "memory_peak_since_container_start_bytes": max_available("memory_peak_since_container_start_bytes"),
        }


def calculate_utilization(before: dict, after: dict, sampled: dict,
                          elapsed_seconds: float, invocations: int,
                          limits: dict) -> dict:

    """Contatori CPU esatti nel cgroup; picchi RAM campionati e dichiarati."""

    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0 or invocations < 1:
        raise ValueError("Durata o numero di invocazioni non validi")
    cpu_diff = {}
    for key in ("usage_usec", "user_usec", "system_usec", "nr_periods", "nr_throttled", "throttled_usec"):
        cpu_diff[key] = _nonnegative_delta(after["cpu"], before["cpu"], key)
    if cpu_diff["usage_usec"] is None:
        raise RuntimeError("cpu.stat/usage_usec non disponibile")
    cores = limits["cpu_quota_cores"]
    if cores is None or cores <= 0:
        raise RuntimeError("Quota CPU cgroup non applicata: benchmark non confrontabile")
    if limits["memory_limit_bytes"] is None:
        raise RuntimeError("Limite RAM del container non applicato")
    cpu_seconds = cpu_diff["usage_usec"] / 1_000_000
    peak_bytes = sampled["memory_current_peak_sampled_bytes"]
    baseline_bytes = before["memory_current_bytes"]
    event_delta = {key: _nonnegative_delta(after["memory_events"], before["memory_events"], key)
                   for key in ("oom", "oom_kill", "max")}
    if event_delta["oom_kill"] is not None and event_delta["oom_kill"] > 0:
        raise RuntimeError("OOM kill nel container durante la misurazione")
    return {
        "invocations": invocations,
        "elapsed_seconds": elapsed_seconds,
        "wall_seconds_per_invocation": elapsed_seconds / invocations,
        "cpu_seconds_total": cpu_seconds,
        "cpu_seconds_per_invocation": cpu_seconds / invocations,
        "cpu_user_seconds": (cpu_diff["user_usec"] / 1e6 if cpu_diff["user_usec"] is not None else None),
        "cpu_system_seconds": (cpu_diff["system_usec"] / 1e6 if cpu_diff["system_usec"] is not None else None),
        "cpu_utilization_one_core_percent": 100 * cpu_seconds / elapsed_seconds,
        "cpu_utilization_of_quota_percent": 100 * cpu_seconds / (elapsed_seconds * cores),
        "cpu_throttled_periods": cpu_diff["nr_throttled"],
        "cpu_throttled_seconds": (cpu_diff["throttled_usec"] / 1e6
                                  if cpu_diff["throttled_usec"] is not None else None),
        "memory_baseline_bytes": baseline_bytes,
        "memory_end_bytes": after["memory_current_bytes"],
        "memory_current_peak_sampled_bytes": peak_bytes,
        "memory_current_mean_sampled_bytes": sampled["memory_current_mean_sampled_bytes"],
        "memory_current_peak_above_baseline_bytes": max(0, peak_bytes - baseline_bytes),
        "memory_anon_peak_sampled_bytes": sampled["memory_anon_peak_sampled_bytes"],
        "memory_file_peak_sampled_bytes": sampled["memory_file_peak_sampled_bytes"],
        "memory_peak_since_container_start_bytes": sampled["memory_peak_since_container_start_bytes"],
        "memory_limit_bytes": limits["memory_limit_bytes"],
        "memory_peak_sampled_limit_percent": 100 * peak_bytes / limits["memory_limit_bytes"],
        "memory_events_delta": event_delta,
        "memory_sample_count": sampled["sample_count"],
        "memory_sampling_interval_seconds": sampled["sampling_interval_seconds"],
        "memory_sampling_quality": "nominal" if sampled["sample_count"] >= 3 else "few_samples",
    }
