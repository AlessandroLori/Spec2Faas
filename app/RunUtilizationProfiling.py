"""CLI per il confronto dell'utilizzo CPU e RAM Python/Go."""

import argparse
import json
import sys
from pathlib import Path

from app.profiling.FunctionUtilizationRunner import FunctionUtilizationRunner


def main(argv: list[str] | None = None) -> int:
    """Esegue il profiler cgroups v2 e salva un report JSON riproducibile."""
    parser = argparse.ArgumentParser(allow_abbrev=False, description="Profilazione | Utilizzo CPU/RAM delle funzioni Python/Go")
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--go", required=True, type=Path)
    parser.add_argument("--entry", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--args-json", help="Argomenti di una chiamata, come array JSON")
    inputs.add_argument("--args-file", type=Path, help="File JSON degli argomenti di una chiamata")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--invocations", type=int, default=1000)
    parser.add_argument("--max-invocations", type=int, default=1000000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--min-long-seconds", type=float, default=3.0)
    parser.add_argument("--sample-interval", type=float, default=0.1,
                        help="Intervallo in secondi di campionamento RAM (default 0.1)")
    parser.add_argument("--go-gomaxprocs", type=int, default=2,
                        help="Parallelismo runtime Go: default 2, quota Docker 1 CPU")
    parser.add_argument("--timeout-docker", type=float, default=90.0,
                        help="Timeout per invocazione Docker (secondi); -1 disattiva il limite applicativo")
    arguments = parser.parse_args(argv)
    try:
        args = (json.loads(arguments.args_json) if arguments.args_json is not None
                else json.loads(arguments.args_file.read_text(encoding="utf-8")))
        if not isinstance(args, list):
            raise ValueError("Gli argomenti devono essere un array JSON")
        if not arguments.python.is_file() or not arguments.go.is_file():
            raise ValueError("Sorgente Python o Go non trovato")
        runner = FunctionUtilizationRunner(
            timeout_seconds=arguments.timeout_docker,
            sampling_interval_seconds=arguments.sample_interval,
            go_gomaxprocs=arguments.go_gomaxprocs,
            progress=lambda message: print("[Profilazione risorse] " + message, flush=True),
        )
        print("Profilazione | Utilizzo CPU/RAM — solo funzioni, nessun agente LLM", flush=True)
        report = runner.run(
            python_source=arguments.python, go_source=arguments.go,
            entry=arguments.entry, cases=[args], invocations=arguments.invocations,
            max_invocations=arguments.max_invocations, repetitions=arguments.repetitions,
            min_long_seconds=arguments.min_long_seconds,
        )
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                          encoding="utf-8")
        for language, metrics in report["summary"].items():
            print(f"{language}: CPU={metrics['cpu_utilization_of_quota_percent']:.1f}% quota; "
                  f"CPU-s/invocazione={metrics['cpu_seconds_per_invocation']:.6f}; "
                  f"picco RAM cgroup campionato={metrics['memory_current_peak_sampled_bytes']/1048576:.1f} MiB")
        print("Gain (%):", report["gain_percent"])
        print("Report:", output)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print("ERRORE:", str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
