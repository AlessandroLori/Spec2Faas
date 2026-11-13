"""CLI per misurare energia, CPU, RAM e prestazioni nella stessa esecuzione."""

import argparse
import json
import sys
from pathlib import Path

from app.profiling.EnergyProfiler import RaplMeter
from app.profiling.EnergyVariability import energy_variability_lines
from app.profiling.FunctionCombinedRunner import FunctionCombinedRunner


def main(argv: list[str] | None = None) -> int:
    """Esegue il protocollo congiunto e salva un report JSON dettagliato."""
    parser = argparse.ArgumentParser(allow_abbrev=False, description="Profilazione congiunta Python/Go")
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--go", required=True, type=Path)
    parser.add_argument("--entry", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--args-json", help="Array JSON degli argomenti della funzione")
    inputs.add_argument("--args-file", type=Path, help="File JSON degli argomenti")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rapl-access", default="sudo-cat", choices=("sudo-cat", "direct"))
    parser.add_argument("--scaphandre-url", required=True,
                        help="Endpoint Prometheus locale, es. http://127.0.0.1:8085/metrics")
    parser.add_argument("--invocations", type=int, default=1000)
    parser.add_argument("--max-invocations", type=int, default=1000000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--min-long-seconds", type=float, default=20.0)
    parser.add_argument("--calibration-mode", choices=("common", "per-language", "timer"), default="common")
    parser.add_argument("--timer-seconds", type=float, default=None)
    parser.add_argument("--min-invocations-warning", type=int, default=30)
    parser.add_argument("--timeout-docker", type=float, default=180.0,
                        help="Timeout per invocazione Docker (secondi); -1 disattiva il limite applicativo")
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--go-gomaxprocs", type=int, default=2,
                        help="Default 2 (container sempre limitato a 1 CPU)")
    arguments = parser.parse_args(argv)
    try:
        args = (json.loads(arguments.args_json) if arguments.args_json is not None else
                json.loads(arguments.args_file.read_text(encoding="utf-8")))
        if not isinstance(args, list):
            raise ValueError("Argomenti funzione non validi: atteso array JSON")
        if not arguments.python.is_file() or not arguments.go.is_file():
            raise ValueError("Sorgente Python/Go mancante")
        runner = FunctionCombinedRunner(
            rapl=RaplMeter(access=arguments.rapl_access),
            scaphandre_url=arguments.scaphandre_url,
            timeout_seconds=arguments.timeout_docker,
            sampling_interval_seconds=arguments.sample_interval,
            go_gomaxprocs=arguments.go_gomaxprocs,
            progress=lambda message: print("[Profilazione congiunta] " + message, flush=True),
        )
        print("Profilazione congiunta energia + CPU/RAM + tempo", flush=True)
        report = runner.run(
            python_source=arguments.python, go_source=arguments.go,
            entry=arguments.entry, cases=[args],
            invocations=arguments.invocations, max_invocations=arguments.max_invocations,
            repetitions=arguments.repetitions, min_long_seconds=arguments.min_long_seconds,
            calibration_mode=arguments.calibration_mode,
            timer_seconds=arguments.timer_seconds,
            min_invocations_warning=arguments.min_invocations_warning,
        )
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
        print("Invocazioni:", report.get("invocations_by_language") or report["invocations_at_long_point"])
        if not report["minimum_long_seconds_met"]:
            print("ATTENZIONE: una o piu' finestre sono inferiori alla durata minima richiesta")
        for warning in report.get("quality_warnings", []):
            print("ATTENZIONE:", warning)
        for language, row in report["summary"].items():
            def mj(key):
                value = row.get(key)
                return f"{value * 1000:.4f}" if value is not None else "N/D"
            print(f"{language}: tempo={row['wall_seconds_per_invocation'] * 1e6:.3f} us/chiamata; "
                  f"CPU={row['cpu_seconds_per_invocation'] * 1e6:.3f} us/chiamata; "
                  f"RAM picco={row['memory_current_peak_sampled_bytes'] / 1048576:.2f} MiB; "
                  f"RAPL incr.={mj('rapl_incremental_j_per_invocation')} mJ/chiamata; "
                  f"Scaphandre={mj('scaphandre_attributed_j_per_invocation')} mJ/chiamata")
        for line in energy_variability_lines(report):
            print(line)
        print("Gain (%):", report["gain_percent"])
        print("Report:", output)
        return 0
    except KeyboardInterrupt:
        print("\nInterrotto; container temporanei rimossi.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print("ERRORE:", str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
