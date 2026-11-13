"""Preflight dei sensori CPU usati per i benchmark energetici locali."""

import argparse
import json
from pathlib import Path

from app.profiling.HardwareProbe import ProfilingHardwareProbe
from app.profiling.EnergyProfiler import RaplMeter


def main(argv: list[str] | None = None) -> int:
    """Presenta la leggibilità dei contatori e salva il report opzionale."""
    parser = argparse.ArgumentParser(description="Preflight energetico CPU in sola lettura")
    parser.add_argument("--output", type=Path, help="Percorso del report JSON")
    arguments = parser.parse_args(argv)
    report = ProfilingHardwareProbe().probe()
    print("=== Preflight | Capacità di misura energetica CPU ===")
    print("CPU:", report["environment"]["cpu_model"] or "non rilevata")
    rapl = report["rapl"]
    print(f"RAPL (utente normale): {rapl['status']} | package leggibile: "
          f"{rapl['package_energy_readable']}")
    for zone in rapl["zones"]:
        print(f"  {zone['zone']} ({zone['name']}): "
              f"{'leggibile' if zone['energy_readable'] else zone['energy_read_error']}")
    # Il preflight diretto e quello usato dal runner non hanno gli stessi privilegi.
    try:
        access = RaplMeter(access="sudo-cat").preflight()
        report["runner_rapl_access"] = {"status": "available", **access}
        print("RAPL (runner via sudo -n cat): disponibile")
    except (OSError, RuntimeError, ValueError) as exc:
        report["runner_rapl_access"] = {"status": "unavailable", "reason": str(exc)}
        print("RAPL (runner via sudo -n cat): non disponibile —", str(exc))
    print("Strumenti:", ", ".join(
        f"{name}={'sì' if available else 'no'}"
        for name, available in report["environment"]["tools"].items()
    ))
    print("Nessuna funzione eseguita; nessuna energia attribuita a processi.")
    if arguments.output:
        path = arguments.output.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Report:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
