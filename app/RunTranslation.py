"""Avvia una singola traduzione HumanEval+/ReFaaS Python → Go tramite Ollama e Docker.

Esecuzione dalla root del progetto: python -m app.RunTranslation ...
Non cambia il workflow originale di Spec2FaaS e non esegue debugging.
"""

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from app.agents.translation.Pipeline import TranslationEvaluation, TranslationPipeline
from app.agents.translation.Translator import Translator
from app.redataset import ReDatasetRepository
from app.validation.DockerGoRunner import DockerGoRunner
from app.validation.HumanevalCases import HumanEvalCaseExtractor
from app.validation.SeBSCases import SeBSCaseExtractor


# Nessun modello o funzione vengono scelti automaticamente dal sistema.
def select_candidate(repository: ReDatasetRepository, task_id: str) -> str:
    """Risolve un task HumanEval+ o ReFaaS in un solo sorgente del catalogo."""
    if task_id.startswith("ReFaaS/"):
        function = task_id.split("/", 1)[-1]
        if not re.fullmatch(r"f\d{2}", function):
            raise ValueError("Task ReFaaS invalido: esempio ReFaaS/f12")
        matches = [candidate.candidate_id for candidate in repository.list_candidates(
            source_id="predictive-refaas-main", language="python",
            role="refaas_original_function_source")
            if candidate.corpus_path == f"functions/refaas-original-14/{function}/main.py"]
    elif re.fullmatch(r"SeBS/(?:master|feature-workflows)/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", task_id):
        matches = [
            candidate.candidate_id
            for candidate in repository.list_candidates(language="python")
            if (candidate.source_id in SeBSCaseExtractor.SOURCE_IDS
                and candidate.source_task_id == task_id)
        ]
    else:
        matches = [
            candidate.candidate_id
            for candidate in repository.list_candidates(
                source_id=HumanEvalCaseExtractor.SOURCE_ID, language="python"
            )
            if candidate.source_task_id == task_id
        ]
    if len(matches) != 1:
        raise ValueError(
            f"Il task {task_id!r} non corrisponde a un unico sorgente supportato "
            f"(corrispondenze: {len(matches)})"
        )
    return matches[0]


def check_ollama(host: str, model: str, *, timeout_seconds: float = 5.0) -> None:
    """Controlla raggiungibilità e nome del modello senza consumare token."""
    if not host.startswith(("http://", "https://")):
        raise ValueError("Indirizzo Ollama non valido: usare http:// oppure https://")
    try:
        with urlopen(host.rstrip("/") + "/api/tags", timeout=timeout_seconds) as response:
            document = json.load(response)
    except (OSError, HTTPError, URLError, ValueError) as exc:
        raise RuntimeError(
            "Impossibile interrogare Ollama (/api/tags). "
            "Verificare che Ollama sia avviato e che --ollama-host sia corretto."
        ) from exc
    if not isinstance(document, dict) or not isinstance(document.get("models"), list):
        raise RuntimeError("Il server Ollama ha restituito un catalogo modelli non valido")

    names = {
        str(name)
        for item in document["models"]
        if isinstance(item, dict)
        for name in (item.get("name"), item.get("model"))
        if isinstance(name, str)
    }
    requested = {model, model + ":latest"} if ":" not in model else {model}
    if not names.intersection(requested):
        raise RuntimeError(
            f"Il modello {model!r} non risulta disponibile su Ollama. "
            "Controllare il nome con 'ollama list'."
        )


def build_report(result: TranslationEvaluation, *, task_id: str) -> dict:
    """Riepilogo riproducibile: nessuna modifica ai casi o ai sorgenti originali."""
    translation = result.translation
    execution = result.execution
    comparison = result.comparison
    reference = result.reference
    return {
        "schema_version": 1,
        "mode": "translate",
        "task_id": task_id,
        "candidate_id": result.candidate_id,
        "status": result.status,
        "reason": result.reason,
        "prepared_cases": result.prepared_cases,
        "reference": None if reference is None else {
            "status": reference.status,
            "source_sha256": reference.source_sha256,
            "test_sha256": reference.test_sha256,
            "total_cases": reference.total_cases,
            "checked_cases": reference.checked_cases,
            "failed_case_id": reference.failed_case_id,
            "error_type": reference.error_type,
            "error_message": reference.error_message,
        },
        "translation": None if translation is None else {
            "status": translation.status,
            "model": translation.model,
            "prompt_version": Translator.PROMPT_VERSION,
            "source_sha256": translation.source_sha256,
            "code_sha256": (
                hashlib.sha256(translation.code.encode("utf-8")).hexdigest()
                if translation.code is not None else None
            ),
            "generation_seconds": translation.generation_seconds,
            "total_tokens": translation.total_tokens,
            "error_reason": translation.error_reason,
            "diagnostics": translation.diagnostics,
        },
        "execution": None if execution is None else {
            "status": execution.status,
            "code_sha256": execution.code_sha256,
            "test_sha256": execution.test_sha256,
            "compile_seconds": execution.compile_seconds,
            "execution_seconds": execution.execution_seconds,
            "executed_cases": len(execution.cases),
            "error_message": execution.error_message,
        },
        "comparison": None if comparison is None else {
            "status": comparison.status,
            "passed_cases": comparison.passed_cases,
            "total_cases": comparison.total_cases,
            "failed_case_id": comparison.failed_case_id,
            "error_message": comparison.error_message,
        },
    }


def save_report(result: TranslationEvaluation, *, task_id: str, output_dir: Path) -> Path:
    """Salva un tentativo in una directory nuova, fuori dal dataset originale."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = output_dir / run_id
    run_dir.mkdir(mode=0o700, exist_ok=False)

    report = build_report(result, task_id=task_id)
    report["run_id"] = run_id
    report["created_utc"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if result.translation is not None:
        if result.translation.raw_response is not None:
            (run_dir / "model_response.txt").write_text(
                result.translation.raw_response, encoding="utf-8"
            )
        if result.translation.code is not None:
            (run_dir / "translation.go").write_text(
                result.translation.code, encoding="utf-8"
            )
    return run_dir


async def _close_client(client) -> None:
    """Rilascia la sessione HTTP MAF, quando il provider espone la chiusura."""
    close = getattr(client, "close", None)
    if not callable(close):
        inner = getattr(client, "client", None)
        close = getattr(inner, "aclose", None) or getattr(inner, "close", None)
        if not callable(close):
            close = getattr(getattr(inner, "_client", None), "aclose", None)
    if callable(close):
        done = close()
        if inspect.isawaitable(done):
            await done


async def run_once(
    *, task_id: str, model: str, ollama_host: str,
    output_dir: Path, model_timeout_seconds: float = 300.0,
) -> tuple[TranslationEvaluation, Path]:
    """Esegue una singola valutazione con modello Ollama reale e Docker isolato."""
    if model_timeout_seconds == -1:
        model_timeout_seconds = None
    if model_timeout_seconds is not None and (
        isinstance(model_timeout_seconds, bool)
        or not isinstance(model_timeout_seconds, (int, float))
        or not math.isfinite(model_timeout_seconds)
        or model_timeout_seconds <= 0
    ):
        raise ValueError("--model-timeout deve essere positivo, oppure -1 per nessun timeout")
    repository = ReDatasetRepository()
    output_path = output_dir.expanduser().resolve()
    if output_path == repository.root or output_path.is_relative_to(repository.root):
        raise ValueError("La directory dei risultati deve essere esterna a ReDataset")
    candidate_id = select_candidate(repository, task_id)
    runner = DockerGoRunner(enabled=True)
    problem = runner.preflight()
    if problem is not None:
        raise RuntimeError(f"Docker non pronto: {problem}")

    print(f"Task: {task_id} | Modello: {model} | Target: Go", flush=True)
    print("Docker pronto. Controllo Ollama e disponibilità del modello...", flush=True)
    await asyncio.to_thread(check_ollama, ollama_host, model)

    # La configurazione del client segue quella già utilizzata in app/main.py.
    from agent_framework.ollama import OllamaChatClient

    client = OllamaChatClient(model=model, host=ollama_host)
    try:
        translator = Translator(llm=model, model_client=client,
                                timeout_seconds=model_timeout_seconds)
        pipeline = TranslationPipeline(
            repository=repository, translator=translator, go_runner=runner
        )
        print(
            "Avvio: verifica della baseline Python, una traduzione LLM, "
            "compilazione Docker e confronto di TUTTI i casi.", flush=True
        )
        result = await pipeline.run(candidate_id)
    finally:
        await _close_client(client)

    report_dir = save_report(result, task_id=task_id, output_dir=output_dir)
    return result, report_dir


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Una traduzione HumanEval+ o ReFaaS Python → Go con Ollama e verifica Docker."
    )
    parser.add_argument("--task-id", required=True, help="Esempi: HumanEval/3 oppure ReFaaS/f12")
    parser.add_argument("--model", required=True, help="Nome esatto del modello su Ollama")
    parser.add_argument(
        "--ollama-host", default=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="URL Ollama; predefinito: OLLAMA_HOST oppure localhost:11434",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Cartella in cui conservare codice e risultati (esterna a ReDataset)",
    )
    parser.add_argument("--model-timeout", type=float, default=300.0,
                        help="Timeout Translator in secondi; -1 disattiva il timeout applicativo")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        result, directory = asyncio.run(run_once(
            task_id=args.task_id, model=args.model,
            ollama_host=args.ollama_host, output_dir=args.output_dir,
            model_timeout_seconds=args.model_timeout,
        ))
    except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
        print(f"ERRORE: {exc}", file=sys.stderr)
        return 2

    print(f"\nEsito finale: {result.status}")
    if result.reference is not None:
        print(f"Baseline Python: {result.reference.checked_cases}/{result.reference.total_cases}")
    if result.translation is not None:
        print(f"Traduzione: {result.translation.status} | "
              f"Tempo LLM: {result.translation.generation_seconds:.3f} s | "
              f"Token: {result.translation.total_tokens if result.translation.total_tokens is not None else 'non disponibili'}")
    if result.execution is not None:
        print(f"Go: {result.execution.status} | "
              f"Compilazione: {result.execution.compile_seconds:.3f} s | "
              f"Esecuzione suite: {result.execution.execution_seconds:.3f} s")
    if result.comparison is not None:
        print(f"Test Go: {result.comparison.passed_cases}/{result.comparison.total_cases} "
              f"({result.comparison.status})")
    if result.reason:
        print(f"Motivo: {result.reason}")
    print(f"Risultati salvati in: {directory}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
