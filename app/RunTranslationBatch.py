"""Esegue un manifest ReDataset sequenzialmente usando processi RunPipeline isolati.

Non traduce, non valida e non profila direttamente: ogni task viene affidato
all'interfaccia ufficiale app.RunPipeline. Conserva un riepilogo atomico,
log separati e un meccanismo di ripresa con verifica dei report originali.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import signal
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


_TASK_PATTERN = re.compile(
    r"(?:ReFaaS/f\d{2}|HumanEval/\d+|"
    r"SeBS/(?:master|feature-workflows)/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)\Z"
)
# Queste opzioni sono di proprieta' del batch: non devono essere sovrascritte
# nei parametri inoltrati a RunPipeline.
_RESERVED = {
    '--task-id', '--mode', '--output-dir', '--translation-run',
    '--profile-case-id', '--profile-args-json', '--profile-args-file',
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path: Path, document: dict) -> None:
    """Sostituisce atomicamente il JSON, anche durante una campagna lunga."""
    temp = path.with_name(path.name + '.tmp')
    try:
        temp.write_text(json.dumps(document, indent=2, ensure_ascii=False,
                                   allow_nan=False) + '\n', encoding='utf-8')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def load_manifest(path: Path) -> list[str]:
    """Preserva l'ordine esplicito, senza selezioni basate sul CSV euristico."""
    document = json.loads(path.expanduser().read_text(encoding='utf-8'))
    if not isinstance(document, dict) or not isinstance(document.get('tasks'), list):
        raise ValueError("Il manifest deve contenere {'tasks': ['ReFaaS/f03', ...]}")
    tasks = document['tasks']
    if not tasks:
        raise ValueError('Manifest vuoto: nessuna funzione da eseguire')
    if any(type(task) is not str or not _TASK_PATTERN.fullmatch(task) for task in tasks):
        raise ValueError('Il manifest contiene task non supportati')
    if len(set(tasks)) != len(tasks):
        raise ValueError('Il manifest contiene task duplicati')
    return tasks


def _get_option(args: list[str], name: str) -> str | None:
    """Recupera un'opzione singola, scritta come --x y o --x=y."""
    found = []
    for index, arg in enumerate(args):
        if arg == name:
            if index + 1 >= len(args) or args[index + 1].startswith('--'):
                raise ValueError(f'{name} richiede un valore')
            found.append(args[index + 1])
        elif arg.startswith(name + '='):
            found.append(arg[len(name) + 1:])
    if len(found) > 1:
        raise ValueError(f'Opzione ripetuta: {name}')
    return found[0] if found else None


def _has_option(args: list[str], name: str) -> bool:
    return any(a == name or a.startswith(name + '=') for a in args)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Distingue le opzioni batch da quelle originali RunPipeline dopo '--'.

    Senza '--' rimane utilizzabile la vecchia CLI di sola traduzione.
    """
    raw = list(sys.argv[1:] if argv is None else argv)
    has_separator = '--' in raw
    if has_separator:
        position = raw.index('--')
        batch_args, pipeline_args = raw[:position], raw[position + 1:]
        if not pipeline_args:
            raise ValueError('Dopo -- specificare le opzioni di app.RunPipeline')
    else:
        batch_args, pipeline_args = raw, []
    parser = argparse.ArgumentParser(
        description='Campagna sequenziale ReDataset: subprocessi RunPipeline, log e ripresa',
        allow_abbrev=False,
    )
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume-summary', type=Path,
                        help='Riprendi il riepilogo della campagna, se configurazione e report coincidono')
    parser.add_argument('--profile-case-map', type=Path,
                        help='JSON {task_id: case_id}; selezione esplicita per funzione')
    parser.add_argument('--use-first-profile-case', action='store_true',
                        help='Demo esplorativa: consenti il primo test quando manca una selezione esplicita')
    parser.add_argument('--stop-on-error', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Mostra le invocazioni senza eseguirle')
    # Opzioni storiche: compatibilita' con le campagne translation-only precedenti.
    parser.add_argument('--model')
    parser.add_argument('--debug-model')
    parser.add_argument('--debug-attempts', type=int, choices=(-1, *range(31)), default=10)
    parser.add_argument('--ollama-host', default=os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434'))
    parser.add_argument('--model-timeout', type=float, default=300.0)
    parser.add_argument('--debug-timeout', type=float, default=300.0)
    cfg = parser.parse_args(batch_args)
    if has_separator:
        if cfg.model is not None or cfg.debug_model is not None:
            parser.error('Con -- specificare modello e debugger soltanto nelle opzioni RunPipeline')
        if any(a.split('=', 1)[0] in _RESERVED for a in pipeline_args):
            raise ValueError('Le opzioni --task-id, --mode, --output-dir, --translation-run '
                             'e --profile-*input sono gestite dal batch')
        if _get_option(pipeline_args, '--model') is None:
            raise ValueError('Le opzioni RunPipeline devono contenere --model')
    else:
        if cfg.model is None:
            parser.error('Per la vecchia modalità translation-only specificare --model')
        pipeline_args = ['--model', cfg.model, '--debug-attempts', str(cfg.debug_attempts),
                         '--ollama-host', cfg.ollama_host,
                         '--model-timeout', str(cfg.model_timeout),
                         '--debug-timeout', str(cfg.debug_timeout), '--no-profile']
        if cfg.debug_model:
            pipeline_args.extend(['--debug-model', cfg.debug_model])
    cfg.pipeline_args = pipeline_args
    cfg.no_profile = _has_option(pipeline_args, '--no-profile')
    cfg.deploy_language = _get_option(pipeline_args, '--deploy-language') or 'none'
    if cfg.deploy_language not in ('none', 'go'):
        raise ValueError('Deployment batch: none oppure go')
    if cfg.no_profile and (cfg.profile_case_map or cfg.use_first_profile_case):
        raise ValueError('Le opzioni di selezione della profilazione richiedono profilazione attiva')
    if cfg.resume_summary and cfg.dry_run:
        raise ValueError('--resume-summary e --dry-run sono mutuamente esclusivi')
    return cfg


def _profile_cases(cfg, tasks: list[str]) -> dict[str, str | None]:
    """Non presume che il primo test sia un benchmark scientificamente adatto."""
    selections = {}
    if cfg.profile_case_map:
        document = json.loads(cfg.profile_case_map.expanduser().read_text(encoding='utf-8'))
        if not isinstance(document, dict) or any(
            type(key) is not str or type(value) is not str or not value
            for key, value in document.items()
        ):
            raise ValueError('profile-case-map deve essere un oggetto JSON {task_id: case_id}')
        unknown = set(document) - set(tasks)
        if unknown:
            raise ValueError('profile-case-map contiene task estranei: ' + ', '.join(sorted(unknown)))
        selections = document
    if cfg.no_profile:
        return {task: None for task in tasks}
    missing = [task for task in tasks if task not in selections]
    if missing and not cfg.use_first_profile_case:
        raise ValueError('Manca un case_id di profilazione per: ' + ', '.join(missing) +
                         '. Fornire --profile-case-map oppure autorizzare --use-first-profile-case '
                         '(solo demo esplorativa)')
    return {task: selections.get(task) for task in tasks}


def _configuration(cfg, tasks: list[str], cases: dict) -> dict:
    return {
        'task_ids': tasks,
        'manifest_sha256': _sha256(cfg.manifest.expanduser()),
        'pipeline_arguments': cfg.pipeline_args,
        'profiling_case_ids': cases,
        'fallback_to_first_test': cfg.use_first_profile_case,
        'no_profile': cfg.no_profile,
        'deploy_language': cfg.deploy_language,
        'output_dir': str(cfg.output_dir.expanduser().resolve()),
    }


def _fingerprint(configuration: dict) -> str:
    data = json.dumps(configuration, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def _slug(task: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '-', task)


def _command(cfg, task: str, output_dir: Path, case_id: str | None) -> list[str]:
    command = [sys.executable, '-u', '-m', 'app.RunPipeline',
               '--task-id', task, '--mode', 'translate', '--output-dir', str(output_dir)]
    command.extend(cfg.pipeline_args)
    if case_id is not None:
        command.extend(['--profile-case-id', case_id])
    return command


def _find_child_report(task_dir: Path, task: str) -> tuple[Path | None, dict | None]:
    """Il subprocess puo' terminare prima di stampare il percorso del report."""
    paths = list(task_dir.glob(_slug(task) + '/orchestration/*/report.json'))
    if len(paths) != 1:
        return None, None
    path = paths[0]
    try:
        document = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError):
        return path, None
    return path, document if isinstance(document, dict) else None


def _verified_success(document: dict | None, task: str, cfg) -> bool:
    """Non si fida del solo exit code o della dicitura 'completed'."""
    if not isinstance(document, dict) or document.get('task_id') != task:
        return False
    if (document.get('validation') or {}).get('status') != 'passed':
        return False
    if _has_option(cfg.pipeline_args, '--prepare-environment'):
        if (document.get('environment') or {}).get('status') != 'completed':
            return False
    if cfg.no_profile:
        allowed = {'completed_translation_only', 'completed', 'completed_with_warnings'}
        if (document.get('profiling') or {}).get('status') != 'disabled':
            return False
    else:
        allowed = {'completed', 'completed_with_warnings'}
        if (document.get('profiling') or {}).get('status') not in ('completed', 'completed_with_warnings'):
            return False
        measurement = (document.get('profiling') or {}).get('path')
        if not isinstance(measurement, str) or not Path(measurement).is_file():
            return False
    if document.get('status') not in allowed:
        return False
    if cfg.deploy_language == 'go' and (document.get('deployment') or {}).get('status') != 'completed':
        return False
    if cfg.deploy_language == 'none' and (document.get('deployment') or {}).get('status') != 'disabled':
        return False
    return True


async def _stop_process_group(process) -> None:
    """Interrompe anche i processi figli locali in caso di Ctrl-C sul batch."""
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def run_pipeline_task(command: list[str], log_path: Path) -> int:
    """Esegue una singola pipeline; stampa e salva lo stream senza buffer illimitati."""
    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT, start_new_session=True)
    try:
        with log_path.open('wb') as log:
            while True:
                chunk = await process.stdout.read(65536)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()
                sys.stdout.write(chunk.decode('utf-8', errors='replace'))
                sys.stdout.flush()
        return await process.wait()
    except BaseException:
        await _stop_process_group(process)
        raise


def _eligible_to_skip(row: dict, directory: Path, task: str, cfg) -> bool:
    """Solo successi con report originale ancora presente e integro sono saltati."""
    if not isinstance(row, dict) or row.get('task_id') != task or not row.get('success'):
        return False
    try:
        path = Path(row['report_path']).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            return False
        if _sha256(path) != row.get('report_sha256'):
            return False
        content = json.loads(path.read_text(encoding='utf-8'))
        return _verified_success(content, task, cfg)
    except (KeyError, OSError, ValueError, TypeError):
        return False


def _counts(report: dict) -> None:
    rows = report['runs']
    report['completed'] = sum(row.get('success') is True for row in rows)
    report['failed'] = sum(row.get('status') == 'failed' for row in rows)
    report['with_warnings'] = sum(row.get('status') == 'completed_with_warnings' for row in rows)
    report['remaining'] = len(report['tasks']) - report['completed'] - report['failed']


async def execute_batch(cfg) -> tuple[dict, Path]:
    tasks = load_manifest(cfg.manifest)
    cases = _profile_cases(cfg, tasks)
    output_root = cfg.output_dir.expanduser().resolve()
    dataset_root = (Path(__file__).resolve().parents[1] / 'ReDataset').resolve()
    if output_root == dataset_root or output_root.is_relative_to(dataset_root):
        raise ValueError('I report batch devono rimanere fuori dal ReDataset')
    configuration = _configuration(cfg, tasks, cases)
    signature = _fingerprint(configuration)
    if cfg.resume_summary:
        summary_path = cfg.resume_summary.expanduser().resolve()
        directory = summary_path.parent
        report = json.loads(summary_path.read_text(encoding='utf-8'))
        if (report.get('schema_version') != 2 or report.get('kind') != 'sequential_pipeline_batch'
                or report.get('configuration_sha256') != signature
                or report.get('tasks') != tasks
                or not directory.is_relative_to(output_root / '_batch')):
            raise ValueError('Ripresa rifiutata: manifest, configurazione o directory non coincidono')
        if not isinstance(report.get('runs'), list):
            raise ValueError('Riepilogo da riprendere non valido')
    else:
        directory = (output_root / '_batch' /
                     (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_'
                      + uuid.uuid4().hex[:8]))
        if not cfg.dry_run:
            directory.mkdir(parents=True, exist_ok=False)
        summary_path = directory / 'summary.json'
        report = {
            'schema_version': 2, 'kind': 'sequential_pipeline_batch',
            'created_utc': _utc(),
            'manifest_path': str(cfg.manifest.expanduser().resolve()),
            'tasks': tasks, 'configuration': configuration,
            'configuration_sha256': signature,
            'runs': [], 'status': 'running',
        }
    if cfg.dry_run:
        for index, task in enumerate(tasks, 1):
            task_dir = directory / 'artifacts' / f'{index:03d}_{_slug(task)}'
            print(f'[{index}/{len(tasks)}] {task}:', flush=True)
            print(' '.join(__import__('shlex').quote(item) for item in
                           _command(cfg, task, task_dir, cases[task])), flush=True)
            if not cfg.no_profile and cases[task] is None:
                print('  [DEMO] Profilazione: primo caso originale, non validato come rappresentativo')
        report['status'] = 'dry_run'
        return report, summary_path

    (directory / 'logs').mkdir(parents=True, exist_ok=True)
    (directory / 'artifacts').mkdir(parents=True, exist_ok=True)
    previous = {row.get('task_id'): row for row in report['runs']
                if isinstance(row, dict) and row.get('task_id') in tasks}
    report['status'] = 'running'
    report['runs'] = [previous[task] for task in tasks if task in previous]
    _counts(report)
    _save(summary_path, report)
    try:
        for index, task in enumerate(tasks, 1):
            old = previous.get(task)
            if old is not None and _eligible_to_skip(old, directory, task, cfg):
                print(f'[Batch] {index}/{len(tasks)}: {task}: gia\' completato, salto', flush=True)
                continue
            print(f'\n[Batch] {index}/{len(tasks)}: {task}', flush=True)
            task_dir = directory / 'artifacts' / f'{index:03d}_{_slug(task)}'
            # Ogni tentativo usa una directory propria, senza sovrascrivere artefatti precedenti.
            attempt = int(old.get('attempt', 0)) + 1 if old else 1
            attempt_dir = task_dir / f'attempt_{attempt:02d}'
            attempt_dir.mkdir(parents=True, exist_ok=False)
            log_path = directory / 'logs' / f'{index:03d}_{_slug(task)}_attempt_{attempt:02d}.log'
            row = {
                'task_id': task, 'index': index, 'attempt': attempt,
                'status': 'running', 'success': False,
                'started_utc': _utc(), 'log_path': str(log_path),
                'artifact_root': str(attempt_dir),
                'profile_case_id': cases[task],
                'profile_selection': ('disabled' if cfg.no_profile else
                                      'explicit' if cases[task] else 'first_test_fallback'),
            }
            if old:
                row['previous_attempts'] = old.get('previous_attempts', []) + [
                    {key: old.get(key) for key in ('attempt', 'status', 'report_path', 'log_path',
                                                   'error', 'finished_utc')}]
            report['runs'] = [item for item in report['runs'] if item.get('task_id') != task]
            report['runs'].append(row)
            report['runs'].sort(key=lambda item: item['index'])
            _counts(report)
            _save(summary_path, report)
            command = _command(cfg, task, attempt_dir, cases[task])
            try:
                returncode = await run_pipeline_task(command, log_path)
                row['returncode'] = returncode
                child_path, child = _find_child_report(attempt_dir, task)
                if child_path:
                    row['report_path'] = str(child_path.resolve())
                    row['report_sha256'] = _sha256(child_path)
                if child:
                    row['pipeline_status'] = child.get('status')
                    row['stage'] = child.get('stage')
                    row['translation_status'] = (child.get('translation') or {}).get('status')
                    row['validation_status'] = (child.get('validation') or {}).get('status')
                    row['environment_status'] = (child.get('environment') or {}).get('status')
                    row['profiling_status'] = (child.get('profiling') or {}).get('status')
                    row['deployment_status'] = (child.get('deployment') or {}).get('status')
                    row['warnings'] = (child.get('profiling') or {}).get('warnings', [])
                row['success'] = returncode == 0 and _verified_success(child, task, cfg)
                row['status'] = child.get('status', 'failed') if row['success'] else 'failed'
                if not row['success']:
                    row['error'] = ((child or {}).get('error') or
                                    {'type': 'BatchTaskFailure',
                                     'message': f'RunPipeline returncode={returncode}; '
                                                'report assente o esito incompleto'})
            except Exception as exc:
                row['status'] = 'failed'
                row['error'] = {'type': type(exc).__name__, 'message': str(exc)}
                print(f'[Batch] ERRORE {task}: {exc}', file=sys.stderr, flush=True)
            finally:
                row['finished_utc'] = _utc()
                _counts(report)
                _save(summary_path, report)
            if not row['success'] and cfg.stop_on_error:
                break
        _counts(report)
        if report['completed'] == len(tasks):
            report['status'] = 'completed_with_warnings' if report['with_warnings'] else 'completed'
        else:
            report['status'] = 'completed_with_failures'
        report['finished_utc'] = _utc()
        _save(summary_path, report)
        return report, summary_path
    except BaseException:
        report['status'] = 'interrupted'
        report['finished_utc'] = _utc()
        _counts(report)
        _save(summary_path, report)
        raise


def main(argv: list[str] | None = None) -> int:
    try:
        cfg = arguments(argv)
        report, path = asyncio.run(execute_batch(cfg))
    except KeyboardInterrupt:
        print('Batch interrotto: riepilogo parziale conservato', file=sys.stderr)
        return 130
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print('ERRORE:', exc, file=sys.stderr)
        return 2
    if report['status'] == 'dry_run':
        return 0
    print(f"Completate: {report['completed']}; fallite: {report['failed']}; "
          f"restanti: {report['remaining']}; con avvisi: {report['with_warnings']}")
    print('Riepilogo batch:', path)
    return 0 if report['status'] in ('completed', 'completed_with_warnings') else 1


if __name__ == '__main__':
    raise SystemExit(main())
