"""Orchestrazione: traduzione con suite verificata -> profilazione congiunta.

Riusa le interfacce esistenti. Nessuna riparazione Go deterministica,
nessuna inferenza LLM durante la misura, deployment Go su ServerlEdge solo quando richiesto esplicitamente.
"""

import asyncio
import hashlib
import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agents.translation.DebugPipeline import DebugPipeline
from app.agents.translation.PromptProfiles import validate_profiles
from app.profiling.EnergyProfiler import RaplMeter
from app.profiling.EnergyVariability import energy_variability_lines
from app.profiling.FunctionCombinedRunner import FunctionCombinedRunner
from app.redataset import ReDatasetRepository
from app.validation.DockerGoRunner import DockerGoRunner
from app.validation.HumanevalCases import HumanEvalCaseExtractor, original_python_test_source
from app.validation.OutputComparator import OutputComparator
from app.validation.PythonReference import PythonReferenceRunner
from app.validation.ReFaaSCases import ReFaaSCaseExtractor
from app.validation.SeBSCases import SeBSCaseExtractor

MAX_PROFILE_INVOCATIONS = FunctionCombinedRunner.MAX_INVOCATIONS
MAX_PROFILE_TIMER_INVOCATIONS = FunctionCombinedRunner.MAX_TIMER_INVOCATIONS


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, content: dict) -> None:
    """Salva i risultati senza lasciare JSON troncati in caso di interruzione."""
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(json.dumps(content, indent=2, ensure_ascii=False,
                                   allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def prepare_suite(repository: ReDatasetRepository, candidate_id: str, *, provider_runtime=None):
    """Usa gli adattatori di test già presenti; non inventa casi per funzione."""
    source = repository.load_source(candidate_id)
    if source.candidate.source_id in SeBSCaseExtractor.SOURCE_IDS:
        extractor = provider_runtime if provider_runtime is not None else SeBSCaseExtractor()
    elif source.candidate.role == "refaas_original_function_source":
        extractor = ReFaaSCaseExtractor()
    else:
        extractor = HumanEvalCaseExtractor()
    suite = extractor.prepare(source)
    if not suite.cases:
        raise ValueError("La funzione non dispone di test deterministici supportati")
    if suite.source_sha256 != source.candidate.source_sha256 or suite.candidate_id != candidate_id:
        raise ValueError("I test non appartengono al sorgente selezionato")
    entry = suite.entry_point or source.candidate.entry_point
    if not entry or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", entry) or entry == "main":
        raise ValueError("Entry point non compatibile con gli harness attuali")
    relative = Path(source.candidate.corpus_path)
    source_path = (repository.root / relative).resolve()
    if not source_path.is_relative_to(repository.root) or not source_path.is_file():
        raise ValueError("Sorgente ReDataset non valido")
    if sha256_file(source_path) != suite.source_sha256:
        raise ValueError("SHA-256 del sorgente Python non coerente con la suite")
    return source, suite, entry, source_path


def verified_translation(run_dir: Path, *, task_id: str, candidate_id: str,
                         source_sha256: str, test_sha256: str,
                         total_cases: int) -> tuple[Path, dict]:
    """Accetta solo un artefatto attestato da TUTTA la suite e con hash immutato.

    I vecchi report esistono anche quando la traduzione NON e' passata:
    percio' non basta trovare un file debug_XX.go o translation.go.
    """
    run_dir = run_dir.expanduser().resolve()
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("task_id") != task_id or report.get("candidate_id") != candidate_id:
        raise ValueError("Report di traduzione riferito a un altro task/candidato")
    reference = report.get("reference") or {}
    translation = report.get("translation") or {}
    if (reference.get("status") != "passed" or
            reference.get("source_sha256") != source_sha256 or
            reference.get("test_sha256") != test_sha256 or
            reference.get("checked_cases") != total_cases or
            reference.get("total_cases") != total_cases or
            translation.get("source_sha256") != source_sha256):
        raise ValueError("La baseline Python o gli hash dei test non coincidono")

    final_status = report.get("final_status", report.get("status"))
    if final_status in ("passed_initially", "passed"):
        go_path = run_dir / "translation.go"
        expected_hash = translation.get("code_sha256")
        execution = report.get("execution") or {}
        comparison = report.get("comparison") or {}
    elif final_status == "passed_after_debug":
        debugging = report.get("debugging") or {}
        attempts = debugging.get("attempts") or []
        if (debugging.get("method") != "llm_only" or
                debugging.get("final_status") != "passed_after_debug" or
                not attempts or attempts[-1].get("status") != "passed"):
            raise ValueError("Il report del Debugger non attesta il superamento finale")
        last = attempts[-1]
        number = last.get("attempt")
        if type(number) is not int or number < 1:
            raise ValueError("Numero della revisione Go validata non valido")
        go_path = run_dir / f"debug_{number:02d}.go"
        expected_hash = (last.get("debugger") or {}).get("code_sha256")
        execution = last.get("execution") or {}
        comparison = last.get("comparison") or {}
    else:
        raise ValueError(f"La traduzione non ha superato la suite: {final_status}")

    if (comparison.get("status") != "passed" or
            comparison.get("passed_cases") != total_cases or
            comparison.get("total_cases") != total_cases or
            execution.get("status") != "completed" or
            execution.get("executed_cases") != total_cases or
            execution.get("test_sha256") != test_sha256 or
            not isinstance(expected_hash, str) or len(expected_hash) != 64 or
            execution.get("code_sha256") != expected_hash):
        raise ValueError("Mancano prove di compilazione/esecuzione ed equivalenza complete")
    if not go_path.is_file() or sha256_file(go_path) != expected_hash:
        raise ValueError("Codice Go mancante o modificato rispetto alla versione validata")
    return go_path, report


async def prepare_python_for_profiling(*, directory: Path, source_code: str,
                                         entry: str, model: str, ollama_host: str,
                                         max_repairs: int,
                                         pip_requirements: str | None = None,
                                         test_source: str | None = None) -> dict:
    """Prepara l'ambiente Python prima della rivalidazione in modalita' reuse.

    Il Docker Python e' comune a baseline e profilazione. Nessun LLM modifica
    i sorgenti: puo' soltanto proporre pacchetti dopo un errore di build.
    """
    from app.RunTranslation import check_ollama
    from agent_framework.ollama import OllamaChatClient
    from app.agents.EnvironmentPreparation import EnvironmentPreparation
    from app.agents.FaaSDeployer import FaasDeployer

    await asyncio.to_thread(check_ollama, ollama_host, model)
    client = OllamaChatClient(model=model, host=ollama_host)
    try:
        manager = EnvironmentPreparation(directory=directory)
        agent = FaasDeployer(llm=model, model_client=client)
        return await agent.prepare_environment(
            manager=manager, language="python", source=source_code,
            entry=entry, max_repairs=max_repairs,
            pip_requirements=pip_requirements,
            **({"test_source": test_source} if test_source is not None else {}))
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            result = close()
            if asyncio.iscoroutine(result):
                await result


def select_profile_inputs(suite, *, args_json: str | None = None,
                          args_file: Path | None = None,
                          case_id: str | None = None) -> tuple[list[list], dict]:
    """Input di benchmark come dati: test esistente o argomenti espliciti JSON.

    Il primo test e' solo un default tecnico, NON un workload rappresentativo.
    """
    if sum(x is not None for x in (args_json, args_file, case_id)) > 1:
        raise ValueError("Scegliere un solo input: --profile-args-json/file o --profile-case-id")
    if args_file is not None:
        raw = args_file.expanduser().read_text(encoding="utf-8")
        origin = {"kind": "explicit_file", "path": str(args_file.expanduser().resolve())}
    elif args_json is not None:
        raw = args_json
        origin = {"kind": "explicit_json"}
    else:
        matching = [c for c in suite.cases if c.case_id == case_id] if case_id is not None else list(suite.cases[:1])
        if len(matching) != 1:
            raise ValueError("Caso di test richiesto non disponibile o non univoco")
        case = matching[0]
        args = list(case.args)
        origin = {"kind": "validation_test_case", "case_id": case.case_id,
                  "selection": "explicit" if case_id is not None else "first_test_fallback"}
        raw = json.dumps(args, ensure_ascii=False, allow_nan=False)
    try:
        args = json.loads(raw)
        if not isinstance(args, list):
            raise ValueError("Gli argomenti devono essere un array JSON posizionale")
        normalized = json.dumps([args], sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Input di profilazione non rappresentabile come array JSON") from exc
    origin["cases_sha256"] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    origin["note"] = "Un caso ripetuto; non rappresenta automaticamente tutti gli input della funzione"
    return [args], origin


def revalidate_existing(source, suite, go_path: Path, entry: str,
                        environment: dict | None = None) -> dict:
    """Verifica nuovamente tutti i test, senza richiamare alcun agente LLM."""
    python_info = (environment or {}).get("python")
    if environment is not None and python_info is None:
        raise RuntimeError("Ambiente Python assente nella rivalidazione")
    if python_info is not None:
        baseline = PythonReferenceRunner().verify(
            source, suite, python_environment=python_info)
    else:
        baseline = PythonReferenceRunner().verify(source, suite)
    if not baseline.passed:
        raise RuntimeError("La baseline Python non supera piu' la suite di test")
    info = (environment or {}).get("go") or {}
    runner = DockerGoRunner(enabled=True,
        build_image=info.get("build_image") or "golang:1.25-alpine",
        runtime_image=info.get("go_image") or "alpine:3.22",
        go_modules_enabled=info.get("go_modules_enabled", False),
        go_cgo_enabled=info.get("go_cgo_enabled", False))
    problem = runner.preflight()
    if problem:
        raise RuntimeError("Docker per la validazione non disponibile: " + problem)
    code = go_path.read_text(encoding="utf-8")
    execution = runner.run(code, entry_point=entry, suite=suite)
    if (not execution.completed or execution.code_sha256 != sha256_file(go_path) or
            execution.test_sha256 != suite.test_sha256 or execution.candidate_id != suite.candidate_id):
        raise RuntimeError("Rivalidazione Go incompleta: " + execution.status)
    comparison = OutputComparator().compare(suite, execution)
    if not comparison.passed or comparison.passed_cases != len(suite.cases):
        raise RuntimeError("La traduzione riutilizzata non supera tutti i test")
    return {"status": "passed", "python_cases": baseline.checked_cases,
            "go_cases": comparison.passed_cases, "test_sha256": suite.test_sha256,
            "go_sha256": execution.code_sha256}


class PipelineOrchestrator:
    """Orchestratore per un task, i risultati precedenti non vengono sovrascritti."""

    def __init__(self, *, progress=None) -> None:
        self.progress = progress or (lambda message: None)

    async def run(self, *, task_id: str, mode: str, output_dir: Path,
                  translation_run: Path | None = None,
                  model: str | None = None, debug_model: str | None = None,
                  debug_attempts: int = 10, ollama_host: str = "http://127.0.0.1:11434",
                  ollama_no_think: bool = False, ollama_think: bool = False,
                  translator_num_predict: int | None = None,
                  translator_retries: int = 2, translator_retry_no_think: bool = False,
                  debug_strategy: str = "linear",
                  prompt_translator: str = "baseline", prompt_debugger: str = "baseline",
                  prompt_feedback: str = "baseline",
                  debugger_num_predict: int | None = None,
                  ollama_num_ctx: int | None = None,
                  model_timeout: float = 300.0, debug_timeout: float = 300.0,
                  no_profile: bool = False,
                  provider_fixture: Path | None = None,
                  profile_args_json: str | None = None,
                  profile_args_file: Path | None = None,
                  profile_case_id: str | None = None,
                  invocations: int = 1000, max_invocations: int = 1_000_000,
                  repetitions: int = 5, min_long_seconds: float = 35.0,
                  calibration_mode: str = "common", timer_seconds: float | None = None,
                  min_invocations_warning: int = 30,
                  timeout_docker: float = 180.0, sample_interval: float = 0.1,
                  go_gomaxprocs: int = 2, scaphandre_url: str = "http://127.0.0.1:8085/metrics",
                  rapl_access: str = "sudo-cat",
                  deploy_language: str = "none", deploy_model: str | None = None,
                  serverledge_url: str | None = None, deploy_go_image: str | None = None,
                  deploy_function_name: str | None = None, deploy_memory_mb: int = 256,
                  deploy_cpu_demand: float = 1.0,
                  prepare_environment: bool = False, environment_model: str | None = None,
                  environment_max_repairs: int = 10) -> tuple[dict, Path]:
        if (type(prepare_environment) is not bool or type(environment_max_repairs) is not int
                or environment_max_repairs < -1):
            raise ValueError("Configurazione preparazione ambienti non valida")
        # Il provider verra' risolto dai sorgenti per ogni singola funzione.
        # Per un task provider la preparazione dell'ambiente e' automatica.
        if prepare_environment and mode == "reuse" and not environment_model:
            raise ValueError("--mode reuse con --prepare-environment richiede --environment-model")
        if deploy_language not in ("none", "go"):
            raise ValueError("Deployment disponibile in questa fase: none oppure go")
        if deploy_language == "go":
            if not deploy_model or not serverledge_url or not deploy_go_image:
                raise ValueError("--deploy-language go richiede --deploy-model, --serverledge-url e --deploy-go-image")
            if type(deploy_memory_mb) is not int or not 64 <= deploy_memory_mb <= 4096:
                raise ValueError("--deploy-memory-mb: valore atteso 64-4096")
            if type(deploy_cpu_demand) not in (int, float) or not 0 < deploy_cpu_demand <= 1:
                raise ValueError("--deploy-cpu-demand: valore atteso maggiore di zero e <= 1")
            if deploy_function_name is not None and not re.fullmatch(r"[a-z][a-z0-9_]{2,62}", deploy_function_name):
                raise ValueError("--deploy-function-name: nome ServerlEdge non valido")
        elif any(x is not None for x in (deploy_model, serverledge_url, deploy_go_image, deploy_function_name)):
            raise ValueError("Le opzioni di deployment richiedono --deploy-language go")
        if mode not in ("translate", "reuse"):
            raise ValueError("Modalita' supportate: translate oppure reuse")
        if (mode == "translate" and (not model or translation_run is not None)) or (mode == "reuse" and (translation_run is None or model is not None)):
            raise ValueError("translate richiede --model; reuse richiede --translation-run senza --model")
        if type(ollama_no_think) is not bool or type(ollama_think) is not bool:
            raise TypeError("ollama_think e ollama_no_think devono essere bool")
        if ollama_no_think and ollama_think:
            raise ValueError("--ollama-think e --ollama-no-think si escludono")
        if type(translator_retries) is not int or not 0 <= translator_retries <= 5:
            raise ValueError("--translator-retries deve essere un intero fra 0 e 5")
        if debug_strategy not in ("linear", "checkpoint"):
            raise ValueError("debug-strategy non valida")
        validate_profiles(translator=prompt_translator, debugger=prompt_debugger,
                          feedback=prompt_feedback, strategy=debug_strategy)
        if mode == "reuse" and any(value != "baseline" for value in
                                   (prompt_translator, prompt_debugger, prompt_feedback)):
            raise ValueError("I prompt sperimentali sono utilizzabili solo in --mode translate")
        if type(translator_retry_no_think) is not bool:
            raise TypeError("--translator-retry-no-think deve essere bool")
        if translator_num_predict is not None and (type(translator_num_predict) is not int
                                                   or translator_num_predict == 0 or translator_num_predict < -1):
            raise ValueError("translator_num_predict non valido")
        if debugger_num_predict is not None and (type(debugger_num_predict) is not int
                                                 or debugger_num_predict == 0 or debugger_num_predict < -1):
            raise ValueError("debugger_num_predict non valido")
        if ollama_num_ctx is not None and (type(ollama_num_ctx) is not int or ollama_num_ctx <= 0):
            raise ValueError("--ollama-num-ctx deve essere un intero positivo")
        if isinstance(model_timeout, bool) or not isinstance(model_timeout, (int, float)) or (
            not math.isfinite(model_timeout) or (model_timeout <= 0 and model_timeout != -1)
        ):
            raise ValueError("--model-timeout deve essere positivo, oppure -1 per nessun timeout")
        if isinstance(debug_timeout, bool) or not isinstance(debug_timeout, (int, float)) or (
            not math.isfinite(debug_timeout) or (debug_timeout <= 0 and debug_timeout != -1)
        ):
            raise ValueError("--debug-timeout deve essere positivo, oppure -1 per nessun timeout")
        if isinstance(timeout_docker, bool) or not isinstance(timeout_docker, (int, float)) or (
            not math.isfinite(timeout_docker) or (timeout_docker <= 0 and timeout_docker != -1)
        ):
            raise ValueError("--timeout-docker deve essere positivo, oppure -1 per nessun timeout")
        if mode == "reuse" and (ollama_no_think or ollama_think or translator_num_predict is not None
                                or debugger_num_predict is not None or ollama_num_ctx is not None
                                or translator_retry_no_think or translator_retries != 2):
            raise ValueError("Le opzioni Ollama/Translator sono utilizzabili solo in translate")
        if (type(debug_attempts) is not int
                or (debug_attempts != -1
                    and not 0 <= debug_attempts <= DebugPipeline.MAX_DEBUG_ATTEMPTS)):
            raise ValueError(
                f"debug-attempts deve essere -1 oppure fra 0 e "
                f"{DebugPipeline.MAX_DEBUG_ATTEMPTS}"
            )
        if not task_id or not (re.fullmatch(r"(?:ReFaaS/f\d{2}|HumanEval/\d+)", task_id)
                               or re.fullmatch(r"SeBS/(?:master|feature-workflows)/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", task_id)):
            raise ValueError("Task-id non supportato (esempi: ReFaaS/f03, HumanEval/3, SeBS/master/010.sleep)")
        root = output_dir.expanduser().resolve()
        repository = ReDatasetRepository()
        if root == repository.root or root.is_relative_to(repository.root):
            raise ValueError("I risultati non possono essere salvati nel ReDataset")
        if no_profile and any(x is not None for x in (profile_args_json, profile_args_file, profile_case_id)):
            raise ValueError("Con --no-profile non indicare input di profilazione")
        if not no_profile:
            # Fallire prima di interrogare Ollama se la configurazione e' palesemente errata.
            if calibration_mode not in ("common", "per-language", "timer"):
                raise ValueError("Modalita' di calibrazione non valida")
            max_allowed = (MAX_PROFILE_TIMER_INVOCATIONS if calibration_mode == "timer"
                           else MAX_PROFILE_INVOCATIONS)
            if not 2 <= invocations <= max_invocations <= max_allowed:
                raise ValueError("Numero di invocazioni non valido")
            if not 1 <= repetitions <= 20:
                raise ValueError("Ripetizioni non valide")
            if not 1 <= go_gomaxprocs <= 256:
                raise ValueError("GOMAXPROCS non valido")
            if timer_seconds is not None and calibration_mode != "timer":
                raise ValueError("timer-seconds richiede calibration-mode=timer")
            if calibration_mode == "timer":
                target = min_long_seconds if timer_seconds is None else timer_seconds
                if not isinstance(target, (int, float)) or not math.isfinite(target) or target <= 0:
                    raise ValueError("Durata Timer non valida")
                if timeout_docker != -1 and target >= timeout_docker:
                    raise ValueError("La durata Timer deve essere minore di --timeout-docker")
            if type(min_invocations_warning) is not int or min_invocations_warning < 2:
                raise ValueError("Soglia avviso invocazioni non valida")

        from app.RunTranslation import select_candidate
        candidate_id = select_candidate(repository, task_id)
        from app.providers.ProviderDependencyResolver import (
            ProviderDependencyResolver, ProviderDependencyError,
        )
        selected_source = repository.load_source(candidate_id)
        dependency = ProviderDependencyResolver(repository).resolve(selected_source)
        self.progress(dependency.announcement(task_id))
        if dependency.status == "unsupported":
            provider_name = dependency.provider or "multipli/ambigui"
            raise ProviderDependencyError(
                f"Dipendenza provider rilevata ({provider_name}: "
                f"{', '.join(dependency.services) or dependency.detection}), "
                f"ma l'adattatore/test originale non e' ancora supportato: "
                f"{dependency.detection}. Non viene usato il runner senza provider."
            )
        provider_runtime = None
        if dependency.status == "supported":
            if not no_profile or mode != "translate" or deploy_language != "none":
                raise ValueError(
                    "Funzione con provider riconosciuto automaticamente: "
                    "in questa prima fase sono disponibili traduzione e validazione "
                    "(--mode translate --no-profile --deploy-language none). "
                    "La profilazione con Floci sara' integrata separatamente; "
                    "non vengono prodotte misure non confrontabili."
                )
            if not prepare_environment:
                prepare_environment = True
                self.progress("[Dependencies] Preparazione ambiente Python/Go attivata automaticamente")
            from app.providers.ProviderTestRuntime import ProviderTestRuntime
            provider_runtime = ProviderTestRuntime(
                repository=repository, source=selected_source,
                provider=dependency.provider, fixture_path=provider_fixture,
            )
        elif provider_fixture is not None:
            raise ValueError("--provider-fixture e' utilizzabile soltanto per una funzione con provider")
        if environment_model is not None and not prepare_environment:
            raise ValueError("--environment-model richiede --prepare-environment")
        source, suite, entry, python_path = prepare_suite(
            repository, candidate_id, provider_runtime=provider_runtime)
        validation_test_source = original_python_test_source(source, suite)
        if not no_profile:
            cases, input_metadata = select_profile_inputs(
                suite, args_json=profile_args_json, args_file=profile_args_file,
                case_id=profile_case_id)
        else:
            cases, input_metadata = None, None

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = timestamp + "_" + uuid.uuid4().hex[:8]
        task_dir = root / task_id.replace("/", "-")
        directory = task_dir / "orchestration" / run_id
        directory.mkdir(parents=True, exist_ok=False)
        report_path = directory / "report.json"
        report: dict[str, Any] = {
            "schema_version": 1, "kind": "function_translation_validation_profiling_orchestration",
            "run_id": run_id, "task_id": task_id, "candidate_id": candidate_id,
            "mode": mode, "status": "running", "stage": "initializing",
            "source": {"path": str(python_path), "sha256": suite.source_sha256},
            "validation": {"test_sha256": suite.test_sha256, "total_cases": len(suite.cases),
                           "entry_point": entry},
            "profiling": {"status": "disabled" if no_profile else "pending",
                          "workload": input_metadata,
                          "configuration": None if no_profile else {
                              "invocations": invocations, "max_invocations": max_invocations,
                              "repetitions": repetitions, "min_long_seconds": min_long_seconds,
                              "calibration_mode": calibration_mode,
                              "timer_seconds": timer_seconds,
                              "min_invocations_warning": min_invocations_warning,
                              "timeout_docker_seconds": None if timeout_docker == -1 else timeout_docker,
                              "go_gomaxprocs": go_gomaxprocs,
                              "scaphandre_url": scaphandre_url,
                              "sample_interval": sample_interval}},
            "translation": {"status": "pending"},
            "prompt_configuration": {"translator": prompt_translator,
                                     "debugger": prompt_debugger,
                                     "feedback": prompt_feedback},
            "deployment": {"status": "disabled" if deploy_language == "none" else "pending",
                           "language": deploy_language},
            "dependencies": dependency.as_dict(),
            "provider": ({"status": "pending", "provider": dependency.provider,
                          "services": list(dependency.services),
                          "automatic_detection": True,
                          "profiling_supported": False} if provider_runtime is not None else
                         {"status": "disabled"}),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json_atomic(report_path, report)

        try:
            report["stage"] = "translation" if mode == "translate" else "reuse_verification"
            write_json_atomic(report_path, report)
            if mode == "translate":
                self.progress("Traduzione e debugging LLM: inizio")
                from app.RunTranslationDebug import run_once
                from contextlib import nullcontext
                # La stessa sessione e lo stesso bucket coprono reference,
                # traduzione iniziale e tutti i tentativi del Debugger.
                with (provider_runtime if provider_runtime is not None else nullcontext()):
                    if provider_runtime is not None:
                        report["provider"] = {"status": "active", **provider_runtime.provenance()}
                        write_json_atomic(report_path, report)
                    evaluated, saved_dir = await run_once(
                    task_id=task_id, model=model, debug_model=debug_model,
                    debug_attempts=debug_attempts, ollama_host=ollama_host,
                    model_timeout_seconds=None if model_timeout == -1 else model_timeout,
                    debug_timeout_seconds=debug_timeout,
                    ollama_think=True if ollama_think else (False if ollama_no_think else None),
                    translator_num_predict=translator_num_predict,
                    translator_retries=translator_retries,
                    translator_retry_no_think=translator_retry_no_think,
                    debug_strategy=debug_strategy,
                    prompt_translator=prompt_translator,
                    prompt_debugger=prompt_debugger,
                    prompt_feedback=prompt_feedback,
                    debugger_num_predict=debugger_num_predict,
                    ollama_num_ctx=ollama_num_ctx,
                    output_dir=task_dir / "translations",
                    environment_directory=directory / "environment" if prepare_environment else None,
                    environment_model=environment_model,
                    environment_max_repairs=environment_max_repairs,
                    provider_runtime=provider_runtime,
                    )
                if prepare_environment:
                    source_env = saved_dir / "report.json"
                    recorded = json.loads(source_env.read_text(encoding="utf-8"))
                    report["environment"] = recorded.get("environment")
                report["translation"] = {"status": evaluated.status,
                                         "run_directory": str(saved_dir.resolve()),
                                         "debug_attempts_limit": debug_attempts,
                                         "debug_strategy": debug_strategy,
                                         "prompt_translator": prompt_translator,
                                         "prompt_debugger": prompt_debugger,
                                         "prompt_feedback": prompt_feedback,
                                         "model": model, "debug_model": debug_model or model,
                                         "inference_options": {
                                             "ollama_think": True if ollama_think else (False if ollama_no_think else None),
                                             "translator_num_predict": translator_num_predict,
                                             "translator_retries": translator_retries,
                                             "translator_retry_no_think": translator_retry_no_think,
                                             "debugger_num_predict": debugger_num_predict,
                                             "ollama_num_ctx_requested": ollama_num_ctx,
                                             "translator_timeout_seconds": None if model_timeout == -1 else model_timeout,
                                             "debugger_timeout_seconds": None if debug_timeout == -1 else debug_timeout}}
                write_json_atomic(report_path, report)
                if not evaluated.passed:
                    raise RuntimeError("Traduzione non validata: " + evaluated.status)
            else:
                saved_dir = translation_run.expanduser().resolve()
                self.progress("Riutilizzo di un artefatto validato, nessuna chiamata LLM")
                report["translation"] = {"status": "reused", "run_directory": str(saved_dir)}
                write_json_atomic(report_path, report)

            go_path, provenance = verified_translation(
                saved_dir, task_id=task_id, candidate_id=candidate_id,
                source_sha256=suite.source_sha256, test_sha256=suite.test_sha256,
                total_cases=len(suite.cases),
            )
            report["translation"].update({"go_path": str(go_path),
                                          "go_sha256": sha256_file(go_path),
                                          "validated_status": provenance.get("final_status", provenance.get("status"))})
            if mode == "reuse" and prepare_environment:
                # Le traduzioni precedenti possono avere solo l'ambiente Go:
                # per la rivalidazione costruire una nuova baseline Docker
                # indipendente, prima di eseguire qualsiasi test Python.
                self.progress("FaaSDeployer MAF: preparazione Python prima della rivalidazione")
                python_environment = await prepare_python_for_profiling(
                    directory=directory / "environment", source_code=source.code,
                    entry=entry, model=environment_model,
                    ollama_host=ollama_host, max_repairs=environment_max_repairs,
                    pip_requirements=source.python_requirements_text,
                    **({"test_source": validation_test_source}
                       if validation_test_source is not None else {}))
                if (python_environment.get("code_sha256") != suite.source_sha256
                        or python_environment.get("status") != "completed"
                        or (source.python_requirements_text is not None and
                            python_environment.get("declared_requirements_sha256") !=
                            source.python_requirements_sha256)):
                    raise RuntimeError("Ambiente Python non corrispondente alla reference")
                self.progress("FaaSDeployer MAF: preparazione Go prima della rivalidazione")
                from agent_framework.ollama import OllamaChatClient
                from app.agents.FaaSDeployer import FaasDeployer
                from app.agents.EnvironmentPreparation import EnvironmentPreparation
                client = OllamaChatClient(model=environment_model, host=ollama_host)
                manager = EnvironmentPreparation(directory=directory / "environment")
                try:
                    agent = FaasDeployer(llm=environment_model, model_client=client)
                    go_environment = await agent.prepare_environment(
                        manager=manager, language="go",
                        source=go_path.read_text(encoding="utf-8"), entry=entry,
                        max_repairs=environment_max_repairs)
                finally:
                    close = getattr(client, "close", None)
                    if callable(close):
                        done = close()
                        if asyncio.iscoroutine(done):
                            await done
                report["environment"] = {
                    "status": "completed", "directory": str(manager.directory),
                    "python": python_environment, "go": go_environment,
                    "model": environment_model, "max_repairs": environment_max_repairs}
            if mode == "reuse":
                self.progress("Rivalidazione Python e Go su tutti i test originali")
                checked = await asyncio.to_thread(revalidate_existing, source, suite, go_path, entry,
                                                  report.get("environment"))
                report["validation"].update(checked)
            else:
                report["validation"].update({"status": "passed", "verified_in_translation_pipeline": True,
                                            "python_cases": len(suite.cases), "go_cases": len(suite.cases)})
            if prepare_environment:
                env_info = report.get("environment") or {}
                if (not isinstance(env_info, dict)
                        or (env_info.get("go") or {}).get("status") != "completed"
                        or (env_info.get("go") or {}).get("code_sha256") != sha256_file(go_path)):
                    raise RuntimeError("Ambiente Go non corrispondente al sorgente validato")
                if provider_runtime is None:
                    if ((env_info.get("python") or {}).get("status") != "completed"
                            or (env_info.get("python") or {}).get("code_sha256") != suite.source_sha256):
                        raise RuntimeError("Ambiente Python non corrispondente alla reference")
                else:
                    # Per le funzioni cloud la reference e' stata eseguita nel
                    # worker provider originale e attestata dai report della
                    # traduzione, non tramite l'immagine Python ordinaria.
                    reference_provenance = (provenance.get("provider_validation") or {})
                    if (suite.comparator != SeBSCaseExtractor.PROVIDER_COMPARATOR
                            or reference_provenance.get("wrapper_sha256") != provider_runtime.wrapper_sha256
                            or reference_provenance.get("reference_image_id") != provider_runtime.provenance().get("reference_image_id")
                            or reference_provenance.get("state_reset_between_candidates") is not True):
                        raise RuntimeError("Reference provider non attestata dal percorso originale")
                    sdk_environment = env_info["go"]
                    if (sdk_environment.get("type") != "provider_go_sdk"
                            or sdk_environment.get("test_sha256") != suite.test_sha256
                            or not reference_provenance.get("go_sdk_image_id")
                            or sdk_environment.get("build_image_id") != reference_provenance["go_sdk_image_id"]
                            or sdk_environment.get("helper_sha256") != provider_runtime.go_sdk.helper_sha256
                            or sdk_environment.get("helper_sha256") != reference_provenance.get("go_sdk_helper_sha256")
                            or sdk_environment.get("sdk_module") != reference_provenance.get("go_sdk_s3_module")):
                        raise RuntimeError("Ambiente SDK Go non attestato dal percorso provider")
                    report["provider"] = {"status": "validated", **reference_provenance,
                                          "automatic_detection": True, "profiling_supported": False}
                    env_info["python_reference"] = {
                        "type": "sebs_original_provider_reference", "source_sha256": suite.source_sha256,
                        "test_sha256": suite.test_sha256, "wrapper_sha256": provider_runtime.wrapper_sha256,
                        "reference_image_id": reference_provenance["reference_image_id"],
                    }
                env_info["status"] = "completed"
            write_json_atomic(report_path, report)

            if no_profile:
                report["status"] = "completed_translation_only"
                report["stage"] = "completed"
                self.progress("Traduzione e validazione terminate; profilazione disattivata")
            else:
                report["stage"] = "profiling"
                write_json_atomic(report_path, report)
                self.progress("Profilazione congiunta delle sole funzioni Python/Go")
                profiler = FunctionCombinedRunner(
                    rapl=RaplMeter(access=rapl_access),
                    scaphandre_url=scaphandre_url, timeout_seconds=timeout_docker,
                    sampling_interval_seconds=sample_interval,
                    go_gomaxprocs=go_gomaxprocs, progress=self.progress,
                    python_image=((report.get("environment") or {}).get("python") or {}).get(
                        "python_image") or "python:3.11-alpine",
                    build_image=((report.get("environment") or {}).get("go") or {}).get(
                        "build_image") or "golang:1.25-alpine",
                    go_image=((report.get("environment") or {}).get("go") or {}).get(
                        "go_image") or "alpine:3.22",
                    go_modules_enabled=((report.get("environment") or {}).get("go") or {}).get(
                        "go_modules_enabled", False),
                    go_cgo_enabled=((report.get("environment") or {}).get("go") or {}).get(
                        "go_cgo_enabled", False),
                )
                measured = await asyncio.to_thread(
                    profiler.run, python_source=python_path, go_source=go_path,
                    entry=entry, cases=cases,
                    validation_suite=(suite if input_metadata["kind"] == "validation_test_case" else None),
                    validation_case_id=input_metadata.get("case_id"),
                    invocations=invocations, max_invocations=max_invocations,
                    repetitions=repetitions, min_long_seconds=min_long_seconds,
                    calibration_mode=calibration_mode, timer_seconds=timer_seconds,
                    min_invocations_warning=min_invocations_warning,
                )
                if (measured.get("python_sha256") != suite.source_sha256 or
                        measured.get("go_sha256") != sha256_file(go_path) or
                        measured.get("input_sha256") != input_metadata["cases_sha256"] or
                        measured.get("entry_point") != entry):
                    raise RuntimeError("Il report di profilazione non corrisponde ai sorgenti/input validati")
                measurement_path = directory / "profiling.json"
                write_json_atomic(measurement_path, measured)
                # Stampa le statistiche dopo le finestre di misura: non
                # introduce I/O nelle acquisizioni RAPL o Scaphandre.
                for line in energy_variability_lines(measured):
                    self.progress(line)
                scaph_counts = {language: measured["summary"][language].get(
                    "scaphandre_attributed_j_per_invocation_resolved_repetitions", 0)
                    for language in ("python", "go")}
                warnings = list(measured.get("quality_warnings", []))
                if not measured.get("minimum_long_seconds_met", False):
                    warnings.append("Finestra minima della profilazione non raggiunta")
                for language, count in scaph_counts.items():
                    if count != repetitions:
                        warnings.append(f"Scaphandre {language}: {count}/{repetitions} ripetizioni valide")
                for language in ("python", "go"):
                    flagged = measured["summary"][language].get("rapl_idle_drift_flagged_repetitions", 0)
                    if flagged:
                        warnings.append(f"RAPL idle {language}: {flagged} ripetizioni instabili")
                report["profiling"].update({
                    "status": "completed_with_warnings" if warnings else "completed",
                    "path": str(measurement_path),
                    "actual_invocations": (
                        measured.get("invocations_by_language")
                        if calibration_mode in ("per-language", "timer")
                        else measured["invocations_at_long_point"]),
                    "calibration_mode": calibration_mode,
                    "timer_target_seconds": measured.get("timer_target_seconds"),
                    "scaphandre_valid_repetitions": scaph_counts,
                    "warnings": warnings,
                })
                report["status"] = "completed_with_warnings" if warnings else "completed"
                report["stage"] = "completed"
                self.progress("Profilazione terminata" + (" con avvisi" if warnings else ""))
            if deploy_language == "go":
                report["stage"] = "deployment"
                name = deploy_function_name or (
                    "spec2faas_" + re.sub(r"[^a-z0-9]+", "_", task_id.lower()).strip("_")[:35]
                    + "_" + uuid.uuid4().hex[:12])
                report["deployment"] = {
                    "status": "running", "language": "go", "name": name,
                    "runtime": "custom", "custom_image": deploy_go_image,
                    "model": deploy_model, "serverledge_url": serverledge_url,
                    "memory_mb": deploy_memory_mb, "cpu_demand": deploy_cpu_demand,
                    "go_sha256": sha256_file(go_path),
                    "test_sha256": suite.test_sha256,
                    "total_validated_cases": len(suite.cases),
                }
                write_json_atomic(report_path, report)
                self.progress("FaaSDeployer MAF: deployment Go su ServerlEdge")
                from agent_framework.ollama import OllamaChatClient
                from app.agents.FaaSDeployer import FaasDeployer
                client = OllamaChatClient(model=deploy_model, host=ollama_host)
                try:
                    agent = FaasDeployer(llm=deploy_model, model_client=client)
                    result = await agent.deploy_verified_go(
                        go_source=go_path, source_sha256=sha256_file(go_path),
                        entry=entry, suite=suite, directory=directory / "deployment",
                        serverledge_url=serverledge_url, image=deploy_go_image,
                        name=name, memory_mb=deploy_memory_mb, cpu_demand=deploy_cpu_demand,
                        go_build_environment=(report.get("environment") or {}).get("go"))
                finally:
                    close = getattr(client, "close", None)
                    if callable(close):
                        result_close = close()
                        if asyncio.iscoroutine(result_close):
                            await result_close
                report["deployment"].update(result)
                report["deployment"]["status"] = "completed"
                report["status"] = "completed_with_warnings" if report["profiling"].get("warnings") else "completed"
                report["stage"] = "completed"
                self.progress("ServerlEdge: funzione Go registrata e invocata correttamente: " + name)
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_json_atomic(report_path, report)
            return report, report_path
        except Exception as exc:
            if report.get("stage") == "deployment":
                report["deployment"]["status"] = "failed"
                report["deployment"]["error"] = {"type": type(exc).__name__, "message": str(exc)}
            report["status"] = "failed"
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
            report["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_json_atomic(report_path, report)
            raise
