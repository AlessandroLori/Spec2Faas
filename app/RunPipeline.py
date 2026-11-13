"""CLI della pipeline: traduci o riusa codice Go validato e opzionalmente profila.

Esempio: python -u -m app.RunPipeline --mode translate ...
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from app.orchestration.PipelineOrchestrator import PipelineOrchestrator


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline: traduzione LLM, equivalenza deterministica, profilazione unica",
        allow_abbrev=False,
    )
    parser.add_argument("--task-id", required=True, help="ReFaaS/f03 oppure HumanEval/3")
    parser.add_argument("--mode", required=True, choices=("translate", "reuse"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", help="Richiesto in mode=translate")
    parser.add_argument("--debug-model", help="Default: stesso modello del Translator")
    parser.add_argument("--debug-attempts", type=int, choices=range(31), default=10)
    parser.add_argument("--translation-run", type=Path, help="Directory contenente report.json e Go validato")
    parser.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--model-timeout", type=float, default=300.0,
                        help="Timeout Translator in secondi (default 300); -1 disattiva il timeout applicativo")
    parser.add_argument("--translator-num-predict", type=int, default=None,
                        help="Max token output Translator; -1 richiede nessun tetto numerico ordinario")
    parser.add_argument("--debugger-num-predict", type=int, default=None,
                        help="Max token output Debugger; -1 richiede nessun tetto numerico ordinario")
    parser.add_argument("--debug-timeout", type=float, default=300.0,
                        help="Timeout Debugger in secondi (default 300); -1 disattiva il timeout applicativo")
    think_group = parser.add_mutually_exclusive_group()
    think_group.add_argument("--ollama-think", action="store_true",
                             help="Richiede thinking attivo per Translator e Debugger (solo translate)")
    think_group.add_argument("--ollama-no-think", action="store_true",
                             help="Disattiva thinking per Translator e Debugger (solo translate)")
    parser.add_argument("--no-profile", action="store_true",
                        help="Esegue solo traduzione e validazione; utile su server senza RAPL")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--profile-args-json", help="Array JSON posizionale della singola invocazione")
    inputs.add_argument("--profile-args-file", type=Path, help="File JSON dell'array posizionale")
    inputs.add_argument("--profile-case-id", help="Usa un caso deterministico per la profilazione")
    parser.add_argument("--invocations", type=int, default=1000)
    parser.add_argument("--max-invocations", type=int, default=1000000)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--min-long-seconds", type=float, default=35.0)
    parser.add_argument("--calibration-mode", choices=("common", "per-language", "timer"), default="common",
                        help="common: stesso N; per-language: N calibrati; timer: ciclo fino al tempo scelto")
    parser.add_argument("--timer-seconds", type=float, default=None,
                        help="Solo timer: durata obiettivo del ciclo di chiamate (default min-long-seconds)")
    parser.add_argument("--min-invocations-warning", type=int, default=30,
                        help="Soglia diagnostica di avviso per poche chiamate completate (default 30)")
    parser.add_argument("--timeout-docker", type=float, default=180.0,
                        help="Timeout in secondi per le invocazioni Docker durante la profilazione; -1 lo disattiva")
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--go-gomaxprocs", type=int, default=2,
                        help="Default: 2 nel runtime Go (container limitato a 1 CPU)")
    parser.add_argument("--scaphandre-url", default="http://127.0.0.1:8085/metrics")
    parser.add_argument("--rapl-access", choices=("sudo-cat", "direct"), default="sudo-cat")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    cfg = arguments(argv)
    orchestrator = PipelineOrchestrator(progress=lambda msg: print("[Pipeline] " + msg, flush=True))
    try:
        report, path = asyncio.run(orchestrator.run(
            task_id=cfg.task_id, mode=cfg.mode, output_dir=cfg.output_dir,
            model=cfg.model, debug_model=cfg.debug_model, debug_attempts=cfg.debug_attempts,
            translation_run=cfg.translation_run, ollama_host=cfg.ollama_host,
            model_timeout=cfg.model_timeout, debug_timeout=cfg.debug_timeout,
            ollama_no_think=cfg.ollama_no_think, ollama_think=cfg.ollama_think,
            translator_num_predict=cfg.translator_num_predict,
            debugger_num_predict=cfg.debugger_num_predict,
            no_profile=cfg.no_profile,
            profile_args_json=cfg.profile_args_json, profile_args_file=cfg.profile_args_file,
            profile_case_id=cfg.profile_case_id,
            invocations=cfg.invocations, max_invocations=cfg.max_invocations,
            repetitions=cfg.repetitions, min_long_seconds=cfg.min_long_seconds,
            calibration_mode=cfg.calibration_mode, timer_seconds=cfg.timer_seconds,
            min_invocations_warning=cfg.min_invocations_warning,
            timeout_docker=cfg.timeout_docker, sample_interval=cfg.sample_interval,
            go_gomaxprocs=cfg.go_gomaxprocs,
            scaphandre_url=cfg.scaphandre_url, rapl_access=cfg.rapl_access,
        ))
    except KeyboardInterrupt:
        print("Interrotto dall'utente", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, TypeError) as exc:
        print("ERRORE:", exc, file=sys.stderr)
        return 2
    print("Esito:", report["status"])
    print("Traduzione:", report["translation"].get("run_directory"))
    if "path" in report["profiling"]:
        print("Profilazione:", report["profiling"]["path"])
        if report["profiling"].get("warnings"):
            print("Qualita' misure:", "; ".join(report["profiling"]["warnings"]))
    print("Report orchestrazione:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
