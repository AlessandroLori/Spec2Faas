"""CLI della pipeline: traduci o riusa codice Go validato e opzionalmente profila.

Esempio: python -u -m app.RunPipeline --mode translate ...
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from app.agents.translation.DebugPipeline import DebugPipeline
from app.agents.translation.PromptProfiles import (
    TRANSLATOR_CHOICES, DEBUGGER_CHOICES, FEEDBACK_CHOICES, profile_help,
)
from app.orchestration.PipelineOrchestrator import PipelineOrchestrator


def environment_repairs_argument(value: str) -> int:
    """Un numero di retry non negativo oppure -1, senza limiti artificiali."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("environment-max-repairs deve essere un intero") from exc
    if number < -1:
        raise argparse.ArgumentTypeError("environment-max-repairs richiede -1 oppure >=0")
    return number


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    requested = "--list-prompt-profiles" in (sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description="Pipeline: traduzione LLM, equivalenza deterministica, profilazione unica",
        allow_abbrev=False,
    )
    parser.add_argument("--task-id", required=not requested, help="ReFaaS/f03 oppure HumanEval/3")
    parser.add_argument("--mode", required=not requested, choices=("translate", "reuse"))
    parser.add_argument("--output-dir", required=not requested, type=Path)
    parser.add_argument("--model", help="Richiesto in mode=translate")
    parser.add_argument("--debug-model", help="Default: stesso modello del Translator")
    parser.add_argument(
        "--debug-attempts", type=int, default=10,
        choices=(-1, *range(DebugPipeline.MAX_DEBUG_ATTEMPTS + 1)),
        help=f"Correzioni LLM: 0-{DebugPipeline.MAX_DEBUG_ATTEMPTS}; -1 senza limite (default: 10)",
    )
    parser.add_argument("--debug-strategy", choices=("linear", "checkpoint"), default="linear",
                        help="linear: baseline storica sequenziale (default); checkpoint: sperimentale")
    parser.add_argument("--prompt-translator", choices=TRANSLATOR_CHOICES, default="baseline",
                        help="Translator: baseline storica oppure format-v1 (solo linear; default baseline)")
    parser.add_argument("--prompt-debugger", choices=DEBUGGER_CHOICES, default="baseline",
                        help="Debugger: baseline storica oppure semantics-v1 (solo linear; default baseline)")
    parser.add_argument("--prompt-feedback", choices=FEEDBACK_CHOICES, default="baseline",
                        help="Diagnostica al Debugger: baseline, feedback-v1, feedback-history-v1 (solo linear)")
    parser.add_argument("--list-prompt-profiles", action="store_true",
                        help="Elenca i profili e termina; con --verbose mostra i prompt completi")
    parser.add_argument("--verbose", action="store_true",
                        help="Solo con --list-prompt-profiles: mostra prompt e regole dinamiche")
    parser.add_argument("--translation-run", type=Path, help="Directory contenente report.json e Go validato")
    parser.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--model-timeout", type=float, default=300.0,
                        help="Timeout Translator in secondi (default 300); -1 disattiva il timeout applicativo")
    parser.add_argument("--translator-retries", type=int, default=2,
                        help="Retry LLM per output iniziale vuoto, da 0 a 5 (default 2)")
    parser.add_argument("--translator-retry-no-think", action="store_true",
                        help="Solo nei retry del Translator, richiedi no-think; registrato nel report")
    parser.add_argument("--translator-num-predict", type=int, default=None,
                        help="Max token output Translator; -1 richiede nessun tetto numerico ordinario")
    parser.add_argument("--debugger-num-predict", type=int, default=None,
                        help="Max token output Debugger; -1 richiede nessun tetto numerico ordinario")
    parser.add_argument("--ollama-num-ctx", type=int, default=None,
                        help="Finestra di contesto richiesta a Ollama per Translator/Debugger (intero positivo)")
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
    parser.add_argument("--prepare-environment", action="store_true",
                        help="Prepara Docker Python/Go con FaaSDeployer prima di validare e profilare")
    parser.add_argument("--environment-model",
                        help="Modello per il recupero LLM degli errori Docker; default: --model")
    parser.add_argument("--environment-max-repairs", type=environment_repairs_argument, default=10,
                        help="Riparazioni LLM della configurazione Docker: >=0; -1 senza limite (default: 10)")
    parser.add_argument("--deploy-language", choices=("none", "go"), default="none",
                        help="Fase 1: richiede al FaaSDeployer MAF il deployment Go, dopo validazione")
    parser.add_argument("--deploy-model", help="Modello Ollama dell'agente FaaSDeployer")
    parser.add_argument("--serverledge-url", help="URL API ServerlEdge (es. http://127.0.0.1:1323)")
    parser.add_argument("--deploy-go-image", help="Immagine runtime custom presente sul Docker del nodo")
    parser.add_argument("--deploy-function-name", help="Nome univoco della funzione (opzionale)")
    parser.add_argument("--deploy-memory-mb", type=int, default=256)
    parser.add_argument("--deploy-cpu-demand", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.list_prompt_profiles:
        print(profile_help(verbose=args.verbose))
        parser.exit(0)
    if args.verbose:
        parser.error("--verbose richiede --list-prompt-profiles")
    return args


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
            translator_retries=cfg.translator_retries,
            translator_retry_no_think=cfg.translator_retry_no_think,
            debug_strategy=cfg.debug_strategy,
            prompt_translator=cfg.prompt_translator,
            prompt_debugger=cfg.prompt_debugger,
            prompt_feedback=cfg.prompt_feedback,
            debugger_num_predict=cfg.debugger_num_predict,
            ollama_num_ctx=cfg.ollama_num_ctx,
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
            deploy_language=cfg.deploy_language, deploy_model=cfg.deploy_model,
            serverledge_url=cfg.serverledge_url, deploy_go_image=cfg.deploy_go_image,
            deploy_function_name=cfg.deploy_function_name,
            deploy_memory_mb=cfg.deploy_memory_mb,
            deploy_cpu_demand=cfg.deploy_cpu_demand,
            prepare_environment=cfg.prepare_environment,
            environment_model=cfg.environment_model,
            environment_max_repairs=cfg.environment_max_repairs,
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
    if report.get("deployment", {}).get("status") == "completed":
        print("ServerlEdge:", report["deployment"].get("name"))
    print("Report orchestrazione:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
