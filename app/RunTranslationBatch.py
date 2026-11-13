"""Avvia in sequenza un manifest esplicito di traduzioni ReDataset (senza profilazione).

L'utente sceglie quali task includere; nessuna selezione automatica per
'facile' e nessun numero massimo arbitrario di task.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.orchestration.PipelineOrchestrator import PipelineOrchestrator, write_json_atomic
from app.redataset import ReDatasetRepository
from app.validation.SeBSCases import SeBSCaseExtractor


def load_manifest(path: Path) -> list[str]:
    """Richiede un elenco JSON esplicito di task; rifiuta duplicati e formati vaghi."""
    document = json.loads(path.expanduser().read_text(encoding='utf-8'))
    if not isinstance(document, dict) or not isinstance(document.get('tasks'), list):
        raise ValueError("Il manifest deve contenere {'tasks': ['ReFaaS/f03', ...]}")
    tasks = document['tasks']
    if not tasks:
        raise ValueError('Manifest vuoto: nessuna traduzione da eseguire')
    if any(type(task) is not str or not (
            re.fullmatch(r'(?:ReFaaS/f\d{2}|HumanEval/\d+)', task)
            or task in SeBSCaseExtractor.ENABLED_TASKS)
           for task in tasks):
        raise ValueError('Il manifest contiene task non supportati')
    if len(set(tasks)) != len(tasks):
        raise ValueError('Il manifest contiene task duplicati')
    return tasks


def arguments(argv=None):
    parser = argparse.ArgumentParser(
        description='Batch traduzione | Sequenza di task espliciti: traduzione + debugging + equivalenza')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--debug-model')
    parser.add_argument('--debug-attempts', type=int, choices=range(31), default=10)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ollama-host', default=os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434'))
    parser.add_argument('--model-timeout', type=float, default=300.0)
    parser.add_argument('--debug-timeout', type=float, default=300.0)
    parser.add_argument('--stop-on-error', action='store_true', help='Altrimenti prosegue con il task successivo')
    return parser.parse_args(argv)


async def execute_batch(cfg) -> tuple[dict, Path]:
    tasks = load_manifest(cfg.manifest)
    output_root = cfg.output_dir.expanduser().resolve()
    dataset_root = ReDatasetRepository().root
    if output_root == dataset_root or output_root.is_relative_to(dataset_root):
        raise ValueError("I report batch devono rimanere fuori dal ReDataset")
    directory = (output_root / '_batch' /
                 (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_'
                  + uuid.uuid4().hex[:8]))
    directory.mkdir(parents=True, exist_ok=False)
    report_path = directory / 'summary.json'
    report = {
        'schema_version': 1, 'kind': 'sequential_translation_batch',
        'manifest_path': str(cfg.manifest.expanduser().resolve()),
        'tasks': tasks, 'model': cfg.model,
        'debug_model': cfg.debug_model or cfg.model,
        'debug_attempts_limit': cfg.debug_attempts,
        'profiling': 'disabled_for_batch',
        'runs': [], 'status': 'running',
    }
    write_json_atomic(report_path, report)
    orchestrator = PipelineOrchestrator(
        progress=lambda message: print('[Batch traduzione] ' + message, flush=True))
    try:
        for index, task in enumerate(tasks, 1):
            print(f'\n[Batch traduzione] Task {index}/{len(tasks)}: {task}', flush=True)
            try:
                result, path = await orchestrator.run(
                    task_id=task, mode='translate', output_dir=cfg.output_dir,
                    model=cfg.model, debug_model=cfg.debug_model,
                    debug_attempts=cfg.debug_attempts, ollama_host=cfg.ollama_host,
                    model_timeout=cfg.model_timeout, debug_timeout=cfg.debug_timeout,
                    no_profile=True,
                )
                report['runs'].append({'task_id': task, 'status': result['status'],
                                       'report_path': str(path)})
            except (ValueError, RuntimeError, OSError, ImportError, KeyError, TypeError) as exc:
                report['runs'].append({'task_id': task, 'status': 'failed',
                                       'error': f'{type(exc).__name__}: {exc}'})
                print('[Batch traduzione] ERRORE:', task, exc, file=sys.stderr, flush=True)
                if cfg.stop_on_error:
                    break
            finally:
                write_json_atomic(report_path, report)
        failures = sum(row['status'] == 'failed' for row in report['runs'])
        successes = sum(row['status'] == 'completed_translation_only' for row in report['runs'])
        report['completed'] = successes
        report['failed'] = failures
        report['status'] = 'completed' if failures == 0 and len(report['runs']) == len(tasks) else 'completed_with_failures'
        write_json_atomic(report_path, report)
        return report, report_path
    except BaseException:
        report['status'] = 'interrupted'
        write_json_atomic(report_path, report)
        raise


def main(argv=None) -> int:
    cfg = arguments(argv)
    try:
        report, path = asyncio.run(execute_batch(cfg))
    except KeyboardInterrupt:
        print('Batch interrotto; riepilogo parziale salvato', file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, ImportError, TypeError) as exc:
        print('ERRORE:', exc, file=sys.stderr)
        return 2
    print(f"Completate: {report['completed']}; fallite: {report['failed']}")
    print('Riepilogo batch:', path)
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
