"""CLI sperimentale per il confronto energetico locale Python/Go con Docker."""

import argparse
import json
import sys
from pathlib import Path

from app.profiling.EnergyProfiler import RaplMeter
from app.profiling.EnergyVariability import energy_variability_lines
from app.profiling.FunctionEnergyRunner import FunctionEnergyRunner


def main(argv: list[str] | None = None) -> int:
    """Acquisisce parametri espliciti e produce un report riproducibile."""
    parser = argparse.ArgumentParser(allow_abbrev=False, description="Energia delle funzioni CPU Python/Go")
    parser.add_argument("--python", required=True, type=Path, help="Sorgente Python originale")
    parser.add_argument("--go", required=True, type=Path, help="Sorgente Go già validato")
    parser.add_argument("--entry", required=True, help="Nome comune dell'entry point")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--args-json",
                        help="Array JSON degli argomenti della chiamata; es. '[[1,2,-4,5]]'")
    input_group.add_argument("--args-file", type=Path,
                        help="File JSON con l'array di argomenti, per input ReFaaS grandi")
    parser.add_argument("--output", required=True, type=Path, help="Report JSON fuori dal dataset")
    parser.add_argument("--rapl-access", default="sudo-cat", choices=("direct", "sudo-cat"))
    parser.add_argument("--scaphandre-url", help="Prometheus locale: http://127.0.0.1:8085/metrics")
    parser.add_argument("--invocations", type=int, default=1000)
    parser.add_argument("--max-invocations", type=int, default=1000000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--min-long-seconds", type=float, default=3.0)
    parser.add_argument("--timeout-docker", type=float, default=90.0,
                        help="Timeout per invocazione Docker (secondi); -1 disattiva il limite applicativo")
    parser.add_argument("--go-gomaxprocs", type=int, default=2,
                        help="Parallelismo Go nel container: default 2, quota Docker 1 CPU")
    arguments = parser.parse_args(argv)
    try:
        args = (json.loads(arguments.args_json) if arguments.args_json is not None
                else json.loads(arguments.args_file.read_text(encoding="utf-8")))
        if not isinstance(args, list):
            raise ValueError("--args-json deve essere una lista di argomenti")
        # Una chiamata con più parametri: [x, y]; un parametro lista: [[x,y]].
        cases = [args]
        if not arguments.python.is_file() or not arguments.go.is_file():
            raise ValueError("Uno dei due sorgenti non esiste")
        meter = RaplMeter(access=arguments.rapl_access)
        runner = FunctionEnergyRunner(
            rapl=meter, scaphandre_url=arguments.scaphandre_url,
            timeout_seconds=arguments.timeout_docker,
            go_gomaxprocs=arguments.go_gomaxprocs,
            progress=lambda message: print("[Profilazione energia] " + message, flush=True),
        )
        print("Profilazione energetica CPU — nessun agente LLM coinvolto", flush=True)
        report = runner.run(
            python_source=arguments.python, go_source=arguments.go,
            entry=arguments.entry, cases=cases, invocations=arguments.invocations,
            max_invocations=arguments.max_invocations, repetitions=arguments.repetitions,
            min_long_seconds=arguments.min_long_seconds,
        )
        path = arguments.output.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
        print("Invocazioni per punto lungo:", report["invocations_at_long_point"])
        for language, result in report["summary"].items():
            print(f"{language}: RAPL incrementale={result['rapl_incremental_j_per_invocation']} J/chiamata; "
                  f"Scaphandre={result['scaphandre_attributed_j_per_invocation']} J/chiamata")
        for line in energy_variability_lines(report):
            print(line)
        print("Gain (%):", report["energy_gain_percent"])
        print("Report:", path)
        return 0
    except KeyboardInterrupt:
        print("\nProfilazione interrotta dall'utente; container temporanei rimossi.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print("ERRORE:", exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
