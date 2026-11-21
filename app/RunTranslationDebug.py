"""Traduzione Python → Go con debugging esclusivamente LLM, via Ollama e Docker."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
from pathlib import Path

from app.RunTranslation import (
    _close_client, check_ollama, save_report, select_candidate,
)
from app.agents.translation.DebugPipeline import DebugPipeline, DebugPipelineEvaluation
from app.agents.translation.Debugger import TranslationDebugger
from app.agents.translation.Pipeline import TranslationPipeline
from app.agents.translation.Translator import Translator
from app.redataset import ReDatasetRepository
from app.validation.DockerGoRunner import DockerGoRunner


def _compact_execution(execution) -> dict | None:
    """Registra l'esito dell'esecuzione senza duplicare migliaia di output."""
    if execution is None:
        return None
    return {
        "status": execution.status,
        "code_sha256": execution.code_sha256,
        "test_sha256": execution.test_sha256,
        "compile_seconds": execution.compile_seconds,
        "execution_seconds": execution.execution_seconds,
        "executed_cases": len(execution.cases),
        "error_message": execution.error_message,
    }


def _compact_comparison(comparison) -> dict | None:
    if comparison is None:
        return None
    return {
        "status": comparison.status,
        "passed_cases": comparison.passed_cases,
        "total_cases": comparison.total_cases,
        "failed_case_id": comparison.failed_case_id,
        "error_message": comparison.error_message,
    }


def save_debug_report(
    result: DebugPipelineEvaluation, *, task_id: str, output_dir: Path,
    user_prompt_builder=None,
) -> Path:
    """Salva versioni LLM distinte; non sovrascrive il primo sorgente Go."""
    run_dir = save_report(result.initial, task_id=task_id, output_dir=output_dir)
    report_file = run_dir / "report.json"
    report = json.loads(report_file.read_text(encoding="utf-8"))
    # Archivia tutte le generazioni del Translator: la risposta vuota non
    # deve sparire quando un successivo retry produce codice valido.
    translation_trace = []
    for number, response in enumerate(result.initial.translation_attempts, 1):
        if response.raw_response is not None:
            (run_dir / f"translation_attempt_{number:02d}_response.txt").write_text(
                response.raw_response, encoding="utf-8")
        if response.code is not None:
            (run_dir / f"translation_attempt_{number:02d}.go").write_text(
                response.code, encoding="utf-8")
        translation_trace.append({
            "attempt": number,
            "status": response.status,
            "error_reason": response.error_reason,
            "generation_seconds": response.generation_seconds,
            "total_tokens": response.total_tokens,
            "diagnostics": response.diagnostics,
            "think": (result.initial.translation_attempt_think[number - 1]
                      if number <= len(result.initial.translation_attempt_think) else None),
            "code_sha256": (hashlib.sha256(response.code.encode("utf-8")).hexdigest()
                            if response.code is not None else None),
        })
    entries = []
    for item in result.attempts:
        number = item.request.attempt
        label = f"debug_{number:02d}"
        if item.result.raw_response is not None:
            (run_dir / f"{label}_response.txt").write_text(
                item.result.raw_response, encoding="utf-8",
            )
        # Archivia il prompt utente esatto ricostruito dalla richiesta tipizzata;
        # nessuna modifica alle istruzioni, ai sorgenti o alle risposte LLM.
        user_prompt = (user_prompt_builder(item.request) if user_prompt_builder
                       else TranslationDebugger._prompt(item.request))
        (run_dir / f"{label}_prompt.txt").write_text(user_prompt, encoding="utf-8")
        if item.result.code is not None:
            (run_dir / f"{label}.go").write_text(item.result.code, encoding="utf-8")
            # L'artefatto salvato è identico a quello effettivamente testato.
            if hashlib.sha256((run_dir / f"{label}.go").read_bytes()).hexdigest() != item.result.code_sha256:
                raise RuntimeError("Hash del file Go corretto non coerente")
        entries.append({
            "attempt": number,
            "status": item.status,
            "failure_kind": item.request.failure_kind,
            "failure_details": item.request.failure_details,
            "failed_case_id": item.request.failed_case_id,
            "previous_code_sha256": item.request.current_code_sha256,
            "checkpoint_passed_cases": item.request.checkpoint_passed_cases,
            "checkpoint_total_cases": item.request.checkpoint_total_cases,
            "selected_as_checkpoint": item.selected_as_checkpoint,
            "selected_for_exploration": item.selected_for_exploration,
            "checkpoint_passed_case_ids": list(item.request.checkpoint_passed_case_ids)
                if item.request.checkpoint_passed_case_ids is not None else None,
            "verified_case_passes": item.verified_case_passes,
            "verified_passed_case_ids": list(item.verified_passed_case_ids)
                if item.verified_passed_case_ids is not None else None,
            "verified_failed_case_ids": list(item.verified_failed_case_ids)
                if item.verified_failed_case_ids is not None else None,
            "user_prompt_sha256": hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
            "user_prompt_characters": len(user_prompt),
            "debugger": {
                "status": item.result.status,
                "model": item.result.model,
                "prompt_version": TranslationDebugger.PROMPT_VERSION,
                "code_sha256": item.result.code_sha256,
                "generation_seconds": item.result.generation_seconds,
                "total_tokens": item.result.total_tokens,
                "error_reason": item.result.error_reason,
                "diagnostics": item.result.diagnostics,
            },
            "execution": _compact_execution(item.execution),
            "comparison": _compact_comparison(item.comparison),
            "reason": item.reason,
        })
    report["schema_version"] = 2
    report["translation_generations"] = translation_trace
    report["final_status"] = result.status
    report["debugging"] = {
        "method": "llm_only",
        "attempted": len(entries),
        "final_status": result.status,
        "passed": result.passed,
        "best_checkpoint_source_attempt": result.checkpoint_source_attempt,
        "attempts": entries,
    }
    report_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return run_dir


async def run_once(
    *, task_id: str, model: str, ollama_host: str, output_dir: Path,
    debug_model: str | None = None, debug_attempts: int = 1,
    ollama_think: bool | None = None,
    translator_num_predict: int | None = None,
    translator_retries: int = 2, translator_retry_no_think: bool = False,
    debugger_num_predict: int | None = None,
    ollama_num_ctx: int | None = None,
    model_timeout_seconds: float | None = 300.0, debug_timeout_seconds: float | None = 300.0,
    environment_directory: Path | None = None, environment_model: str | None = None,
    environment_max_repairs: int = 10, debug_strategy: str = "linear",
) -> tuple[DebugPipelineEvaluation, Path]:
    """Seleziona task e modelli prima dell'esecuzione, senza routing LLM."""
    if debug_strategy not in ("linear", "checkpoint"):
        raise ValueError("debug_strategy non valida")
    if ollama_think is not None and type(ollama_think) is not bool:
        raise TypeError("ollama_think deve essere bool oppure None")
    if type(translator_retries) is not int or not 0 <= translator_retries <= 5:
        raise ValueError("translator_retries deve essere un intero fra 0 e 5")
    if type(translator_retry_no_think) is not bool:
        raise TypeError("translator_retry_no_think deve essere bool")
    if translator_num_predict is not None and (type(translator_num_predict) is not int
                                               or translator_num_predict == 0 or translator_num_predict < -1):
        raise ValueError("translator_num_predict non valido")
    if debugger_num_predict is not None and (type(debugger_num_predict) is not int
                                             or debugger_num_predict == 0 or debugger_num_predict < -1):
        raise ValueError("debugger_num_predict non valido")
    if ollama_num_ctx is not None and (type(ollama_num_ctx) is not int or ollama_num_ctx <= 0):
        raise ValueError("ollama_num_ctx deve essere un intero positivo oppure None")
    if model_timeout_seconds == -1:
        model_timeout_seconds = None
    if model_timeout_seconds is not None and (
        isinstance(model_timeout_seconds, bool)
        or not isinstance(model_timeout_seconds, (int, float))
        or not math.isfinite(model_timeout_seconds)
        or model_timeout_seconds <= 0
    ):
        raise ValueError("model_timeout_seconds deve essere positivo, oppure -1/None per nessun timeout")
    # -1 e' il sentinel pubblico della CLI; None e' la modalita' interna senza wait_for.
    if debug_timeout_seconds == -1:
        debug_timeout_seconds = None
    if debug_timeout_seconds is not None and (
        isinstance(debug_timeout_seconds, bool)
        or not isinstance(debug_timeout_seconds, (int, float))
        or not math.isfinite(debug_timeout_seconds)
        or debug_timeout_seconds <= 0
    ):
        raise ValueError("debug_timeout_seconds deve essere positivo, oppure -1/None per nessun timeout")
    repository = ReDatasetRepository()
    output_path = output_dir.expanduser().resolve()
    if output_path == repository.root or output_path.is_relative_to(repository.root):
        raise ValueError("I report devono rimanere fuori da ReDataset")
    task = select_candidate(repository, task_id)
    go_runner = DockerGoRunner(enabled=True)
    problem = go_runner.preflight()
    if problem is not None:
        raise RuntimeError("Docker non pronto: " + problem)

    chosen_debug_model = debug_model or model
    print(f"Task: {task_id} | Traduttore: {model} | Debugger: {chosen_debug_model}", flush=True)
    print("Verifica Ollama e disponibilità dei modelli...", flush=True)
    await asyncio.to_thread(check_ollama, ollama_host, model)
    if debug_attempts:
        await asyncio.to_thread(check_ollama, ollama_host, chosen_debug_model)

    from agent_framework.ollama import OllamaChatClient

    translation_client = OllamaChatClient(model=model, host=ollama_host)
    debugging_client = (
        translation_client if chosen_debug_model == model or debug_attempts == 0
        else OllamaChatClient(model=chosen_debug_model, host=ollama_host)
    )
    environment_client = (
        translation_client if environment_directory is None or environment_model in (None, model)
        else OllamaChatClient(model=environment_model, host=ollama_host)
    )
    manager = None
    agent = None
    if environment_directory is not None:
        from app.agents.EnvironmentPreparation import EnvironmentPreparation
        from app.agents.FaaSDeployer import FaasDeployer
        manager = EnvironmentPreparation(directory=environment_directory)
        agent = FaasDeployer(llm=environment_model or model, model_client=environment_client)
        agent.environment_max_repairs = environment_max_repairs
    try:
        # Conserva l'interfaccia storica della pipeline se la modalita'
        # opzionale di preparazione ambienti non e' stata richiesta.
        environment_options = (
            {"environment_manager": manager, "environment_agent": agent}
            if manager is not None else {}
        )
        translator_agent = Translator(
            llm=model, model_client=translation_client,
            timeout_seconds=model_timeout_seconds,
            ollama_think=ollama_think,
            num_predict=translator_num_predict,
            num_ctx=ollama_num_ctx,
            allow_external_dependencies=manager is not None,
            legacy_prompt=debug_strategy == "linear",
        )
        debugger_agent = TranslationDebugger(
            llm=chosen_debug_model, model_client=debugging_client,
            timeout_seconds=debug_timeout_seconds,
            ollama_think=ollama_think,
            num_predict=debugger_num_predict,
            num_ctx=ollama_num_ctx,
            allow_external_dependencies=manager is not None,
            legacy_prompt=debug_strategy == "linear",
        )
        pipeline = TranslationPipeline(
            repository=repository, go_runner=go_runner,
            **environment_options, translator=translator_agent,
            translator_retries=translator_retries,
            translator_retry_no_think=translator_retry_no_think,
            translation_progress=lambda msg: print("[Translator] " + msg, flush=True),
        )
        coordinator = DebugPipeline(
            pipeline=pipeline, debugger=debugger_agent,
            max_debug_attempts=debug_attempts, strategy=debug_strategy,
        )
        debugging_budget = ("tentativi di debugging LLM senza limite numerico"
                            if debug_attempts == -1
                            else f"massimo {debug_attempts} tentativi di debugging LLM")
        print(f"Avvio: baseline completa, traduzione iniziale e {debugging_budget}.", flush=True)
        result = await coordinator.run(task)
    finally:
        await _close_client(translation_client)
        if debugging_client is not translation_client:
            await _close_client(debugging_client)
        if environment_client is not translation_client and environment_client is not debugging_client:
            await _close_client(environment_client)

    directory = save_debug_report(
        result, task_id=task_id, output_dir=output_dir,
        user_prompt_builder=debugger_agent.build_user_prompt,
    )
    # Registriamo il prompt DI SISTEMA effettivamente usato dai due agenti.
    # Questo distingue la variante con moduli esterni senza disabilitarla.
    translator_prompt = translator_agent._prompt
    debugger_prompt = debugger_agent._system_prompt
    (directory / "translator_system_prompt.txt").write_text(
        translator_prompt, encoding="utf-8")
    (directory / "debugger_system_prompt.txt").write_text(
        debugger_prompt, encoding="utf-8")
    saved_report = directory / "report.json"
    report = json.loads(saved_report.read_text(encoding="utf-8"))
    # Il parametro e' rilevante per la riproducibilita' sperimentale.
    if manager is not None:
        # La baseline puo' fallire prima che il Translator sia interrogato:
        # in quel caso il report contiene translation=None.
        if isinstance(report.get("translation"), dict):
            report["translation"]["prompt_version"] = Translator.DEPENDENCY_PROMPT_VERSION
        for item in report.get("debugging", {}).get("attempts", []):
            item["debugger"]["prompt_version"] = TranslationDebugger.DEPENDENCY_PROMPT_VERSION
        report["environment"] = {
            "status": ("completed" if result.passed and manager.python_result is not None
                       else "python_ready" if manager.python_result is not None
                       else "not_completed"),
            "directory": str(manager.directory),
            "python": manager.python_result, "go": manager.go_result,
            "model": environment_model or model,
            "max_repairs": environment_max_repairs,
        }
    if debug_strategy == "linear":
        if isinstance(report.get("translation"), dict):
            report["translation"]["prompt_version"] = Translator.LEGACY_PROMPT_VERSION
        for item in report.get("debugging", {}).get("attempts", []):
            item["debugger"]["prompt_version"] = TranslationDebugger.LEGACY_PROMPT_VERSION
    report["debugging"]["strategy"] = debug_strategy
    report["prompt_audit"] = {
        "translator_system_sha256": hashlib.sha256(translator_prompt.encode("utf-8")).hexdigest(),
        "translator_system_characters": len(translator_prompt),
        "debugger_system_sha256": hashlib.sha256(debugger_prompt.encode("utf-8")).hexdigest(),
        "debugger_system_characters": len(debugger_prompt),
        "debug_attempt_user_prompts_saved": len(result.attempts),
    }
    report["inference_options"] = {"debug_strategy": debug_strategy, "ollama_think": ollama_think,
                                   "translator_retries": translator_retries,
                                   "translator_retry_no_think": translator_retry_no_think,
                                   "translator_num_predict": translator_num_predict,
                                   "debugger_num_predict": debugger_num_predict,
                                   "ollama_num_ctx_requested": ollama_num_ctx,
                                   "translator_timeout_seconds": model_timeout_seconds,
                                   "debugger_timeout_seconds": debug_timeout_seconds}
    length_limited_generations = [
        item["attempt"] for item in report["translation_generations"]
        if (item.get("diagnostics") or {}).get("length_limited") is True
    ]
    length_limited_debug_attempts = [
        item["attempt"] for item in report["debugging"]["attempts"]
        if (item["debugger"].get("diagnostics") or {}).get("length_limited") is True
    ]
    report["inference_diagnostics"] = {
        "length_limited_translation_attempts": length_limited_generations,
        "length_limited_debug_attempts": length_limited_debug_attempts,
        "length_limited_total": len(length_limited_generations) + len(length_limited_debug_attempts),
        "note": "finish_reason=length: limite raggiunto; non identifica da solo num_ctx effettivo",
    }
    saved_report.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                            encoding="utf-8")
    if result.initial.reference is not None and not result.initial.reference.passed:
        reference = result.initial.reference
        print(f"[Reference Python] {reference.status}: "
              f"{reference.error_type or 'errore'}: "
              f"{reference.error_message or 'nessun dettaglio disponibile'} "
              f"(caso {reference.failed_case_id or 'n/d'}, "
              f"passati {reference.checked_cases}/{reference.total_cases})", flush=True)
    # Anche in RunPipeline un esito non validato deve mostrare il numero di
    # revisioni realmente tentate e la directory da cui leggere i dettagli.
    attempts_label = "senza limite" if debug_attempts == -1 else str(debug_attempts)
    for item in result.attempts:
        case_progress = (f" | test={item.verified_case_passes}/{item.comparison.total_cases}"
                         if item.verified_case_passes is not None and item.comparison is not None
                         else "")
        checkpoint_label = ("promosso" if item.selected_as_checkpoint else
                            "esplorazione" if item.selected_for_exploration else "conservato")
        print(f"[Debugger] Tentativo {item.request.attempt}/{attempts_label}: "
              f"{item.result.status} | esito={item.status} | "
              f"token={item.result.total_tokens if item.result.total_tokens is not None else 'n/d'}"
              f" | checkpoint={checkpoint_label}" + case_progress
              + (f" | motivo={item.reason[:180]}" if item.reason else ""), flush=True)
    limited = report["inference_diagnostics"]["length_limited_total"]
    if limited:
        print(f"[Ollama] AVVISO: {limited} risposte con finish_reason=length; "
              f"num_ctx richiesto={ollama_num_ctx if ollama_num_ctx is not None else 'non esplicito'}. "
              "Il limite effettivo non e' deducibile dal solo report.", flush=True)
    print(f"[Debugger] Totale tentativi: {len(result.attempts)}/{attempts_label}; "
          f"esito finale: {result.status}; report: {directory / 'report.json'}", flush=True)
    return result, directory


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traduzione Python→Go e debugging LLM, senza fix deterministici.",
    )
    parser.add_argument("--task-id", required=True, help="Esempi: HumanEval/3 oppure ReFaaS/f12")
    parser.add_argument("--model", required=True, help="Modello Ollama del Translator")
    parser.add_argument("--debug-model", help="Modello del Debugger; default: --model")
    parser.add_argument(
        "--debug-attempts", type=int, default=1,
        choices=(-1, *range(0, DebugPipeline.MAX_DEBUG_ATTEMPTS + 1)),
        help=("Massimo di correzioni LLM: -1 senza limite oppure da 0 a "
              f"{DebugPipeline.MAX_DEBUG_ATTEMPTS} (default: 1; uso consigliato: 10)"),
    )
    parser.add_argument(
        "--ollama-host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-timeout", type=float, default=300.0,
                        help="Timeout Translator in secondi; -1 disattiva il timeout applicativo")
    parser.add_argument("--translator-retries", type=int, default=2,
                        help="Retry del Translator per risposta vuota (0-5, default 2)")
    parser.add_argument("--translator-retry-no-think", action="store_true",
                        help="Nei retry del Translator usa no-think (esplicito, tracciato)")
    parser.add_argument("--translator-num-predict", type=int, default=None,
                        help="Max token output del Translator; -1 richiede generazione senza limite numerico Ollama")
    parser.add_argument("--debugger-num-predict", type=int, default=None,
                        help="Max token output del Debugger; -1 richiede generazione senza limite numerico Ollama")
    parser.add_argument("--ollama-num-ctx", type=int, default=None,
                        help="Finestra di contesto richiesta a Ollama per Translator/Debugger (intero positivo)")
    parser.add_argument("--debug-timeout", type=float, default=300.0,
                        help="Timeout Debugger in secondi (default 300); -1 disattiva il timeout applicativo")
    think_group = parser.add_mutually_exclusive_group()
    think_group.add_argument("--ollama-think", action="store_true",
                             help="Richiede esplicitamente thinking attivo per Translator e Debugger")
    think_group.add_argument("--ollama-no-think", action="store_true",
                             help="Disattiva esplicitamente thinking per Translator e Debugger")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        result, folder = asyncio.run(run_once(
            task_id=args.task_id, model=args.model, debug_model=args.debug_model,
            ollama_host=args.ollama_host, output_dir=args.output_dir,
            debug_attempts=args.debug_attempts,
            ollama_think=True if args.ollama_think else (False if args.ollama_no_think else None),
            translator_num_predict=args.translator_num_predict,
            translator_retries=args.translator_retries,
            translator_retry_no_think=args.translator_retry_no_think,
            debugger_num_predict=args.debugger_num_predict,
            ollama_num_ctx=args.ollama_num_ctx,
            model_timeout_seconds=args.model_timeout,
            debug_timeout_seconds=args.debug_timeout,
        ))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        return 2

    initial = result.initial
    print(f"\nPrima traduzione: {initial.status}")
    if initial.reference:
        print(f"Baseline Python: {initial.reference.checked_cases}/{initial.reference.total_cases}")
    if initial.translation:
        print(f"Traduttore: {initial.translation.status} | "
              f"{initial.translation.generation_seconds:.3f} s | "
              f"Token: {initial.translation.total_tokens if initial.translation.total_tokens is not None else 'n/d'}")
    if initial.execution:
        print(f"Go iniziale: {initial.execution.status}")
    if initial.comparison:
        print(f"Test iniziali: {initial.comparison.passed_cases}/{initial.comparison.total_cases}")
    for item in result.attempts:
        print(f"Debug {item.request.attempt}: {item.result.status} | "
              f"{item.result.generation_seconds:.3f} s | "
              f"Token: {item.result.total_tokens if item.result.total_tokens is not None else 'n/d'}")
        if item.execution:
            print(f"Go dopo debug: {item.execution.status}")
        if item.comparison:
            print(f"Test dopo debug: {item.comparison.passed_cases}/{item.comparison.total_cases}")
        if item.reason:
            print("Diagnostica: " + item.reason[:350])
    print(f"Esito finale: {result.status}")
    print(f"Risultati salvati in: {folder}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
