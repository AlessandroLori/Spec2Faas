"""Rileva i contatori RAPL dell'host; nessun sensore GPU viene richiesto."""

import os
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


class RaplProbe:
    """Individua i domini RAPL senza modificarne configurazione o permessi."""

    def __init__(self, root: Path = Path("/sys/class/powercap")) -> None:
        self.root = Path(root)

    @staticmethod
    def _read_int(path: Path) -> tuple[int | None, str | None]:
        """Legge un contatore senza confondere errori e valori nulli."""
        try:
            value = int(path.read_text(encoding="utf-8").strip())
            return (value, None) if value >= 0 else (None, "invalid_value")
        except PermissionError:
            return None, "permission_denied"
        except FileNotFoundError:
            return None, "not_present"
        except (OSError, ValueError, UnicodeError):
            return None, "unreadable"

    def _zones(self) -> list[Path]:
        """Individua le zone anche quando sysfs espone alias simbolici."""
        if not self.root.is_dir():
            return []
        candidates: list[Path] = []
        visited: set[Path] = set()
        try:
            for directory in self.root.glob("intel-rapl*"):
                candidates.append(directory)
                candidates.extend(directory.glob("intel-rapl*"))
                for subdir in directory.glob("intel-rapl*"):
                    candidates.extend(subdir.glob("intel-rapl*"))
            zones = []
            for item in sorted(candidates, key=str):
                if not item.is_dir() or not (item / "name").is_file():
                    continue
                identity = item.resolve()
                if identity not in visited:
                    visited.add(identity)
                    zones.append(item)
            return zones
        except OSError:
            return []

    def probe(self) -> dict:
        """Restituisce la leggibilità dei domini, non misure di workload."""
        zones = []
        for directory in self._zones():
            try:
                name = (directory / "name").read_text(encoding="utf-8").strip()
            except OSError:
                name = "unknown"
            value, error = self._read_int(directory / "energy_uj")
            maximum, maximum_error = self._read_int(directory / "max_energy_range_uj")
            zones.append({
                "zone": directory.name, "name": name,
                "energy_readable": error is None, "energy_read_error": error,
                "energy_uj_snapshot": value, "max_energy_range_uj": maximum,
                "max_energy_range_error": maximum_error,
                "is_package": name.startswith("package-"),
            })
        readable = any(item["energy_readable"] for item in zones)
        return {
            "status": "readable" if readable else ("unreadable" if zones else "not_detected"),
            "package_energy_readable": any(item["energy_readable"] and item["is_package"] for item in zones),
            "zones": zones,
            "note": "Package e sottodomini si sovrappongono; RAPL-MMIO e RAPL-MSR non vanno sommati.",
        }


class ProfilingHardwareProbe:
    """Preflight della piattaforma CPU usata per la profilazione delle funzioni Python/Go."""

    def __init__(self, *, rapl: RaplProbe | None = None,
                 which: Callable = shutil.which) -> None:
        self.rapl = rapl or RaplProbe()
        self.which = which

    @staticmethod
    def _cpu_model() -> str | None:
        """Legge il nome CPU per documentare l'hardware del report."""
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.partition(":")[2].strip()
        except OSError:
            pass
        return platform.processor() or None

    def probe(self) -> dict:
        """Descrive hardware e strumenti senza eseguire funzioni."""
        return {
            "schema_version": 2,
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "os": platform.system(), "release": platform.release(),
                "cpu_model": self._cpu_model(), "logical_cpus": os.cpu_count(),
                "python": platform.python_version(),
                "tools": {name: self.which(name) is not None for name in ("docker", "go", "scaphandre")},
            },
            "rapl": self.rapl.probe(),
            "measurement_status": "capability_probe_only",
        }
