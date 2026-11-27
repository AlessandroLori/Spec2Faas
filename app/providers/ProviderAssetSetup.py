"""Catalogued SeBS input assets and a replayable S3 setup plan.

The original generator is executed only in the isolated reference worker.
The host reads data from the existing asset archive; it never executes input.py.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

from app.validation.HumanevalCases import UnsupportedTestFormat


class ProviderAssetSetup:
    ARCHIVE = 'ReDataset_assets_v1.zip'
    MAX_FILES = 128
    MAX_FILE_BYTES = 8_000_000
    MAX_TOTAL_BYTES = 64_000_000
    # Deliberately bounded rollout: these are supported *operations/dependencies*,
    # not a catalogue of hand-written events or per-benchmark adapters.
    STORAGE_OPERATIONS = frozenset({'get_instance', 'download', 'download_stream',
                                    'download_directory', 'upload', 'upload_stream'})
    EXTRA_IMPORTS = frozenset({'PIL', 'sebs.utils'})
    REQUIREMENTS = ('Pillow==10.3.0', 'click==8.3.1', 'docker==7.1.0')

    @staticmethod
    def _path(value: str) -> PurePosixPath:
        if type(value) is not str or '\\' in value or '\x00' in value:
            raise ValueError('Invalid asset path')
        path = PurePosixPath(value)
        if (not path.parts or path.is_absolute() or path.as_posix() != value
                or any(p.startswith('.') for p in path.parts)):
            raise ValueError('Invalid asset path: ' + value)
        return path

    @classmethod
    def capability(cls, source) -> tuple[int, int] | None:
        """Conservative, static admission of the first asset-backed workload class."""
        if (source.candidate.source_id != 'sebs-master'
                or source.candidate.role != 'benchmark_handler_source'
                or len(source.tests) != 1):
            return None
        try:
            tree = ast.parse(source.tests[0].content)
            code = ast.parse(source.code)
            functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
            gen, val, count = (functions[n] for n in ('generate_input', 'validate_output', 'buckets_count'))
            if [a.arg for a in gen.args.args] != ['data_dir', 'size', 'benchmarks_bucket',
                    'input_paths', 'output_paths', 'upload_func', 'nosql_func']:
                return None
            if gen.args.vararg or gen.args.kwarg or gen.args.kwonlyargs or gen.args.posonlyargs:
                return None
            if [a.arg for a in val.args.args] != ['data_dir','input_config','output','language','storage']:
                return None
            if len(count.body) != 1 or not isinstance(count.body[0], ast.Return):
                return None
            counts = ast.literal_eval(count.body[0].value)
            if (type(counts) is not tuple or len(counts) != 2 or
                    any(type(n) is not int for n in counts) or counts != (1, 1)):
                return None
            if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                       and n.func.id == 'upload_func' for n in ast.walk(tree)):
                return None
            if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == 'nosql_func' for n in ast.walk(tree)):
                return None
            if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                       and isinstance(n.func.value, ast.Name) and n.func.value.id == 'storage'
                       and n.func.attr == 'download' for n in ast.walk(val)):
                return None
            # No arbitrary libraries or provider calls are enabled by this profile.
            for module in (tree, code):
                for n in ast.walk(module):
                    names = ([a.name for a in n.names] if isinstance(n, ast.Import) else
                             [n.module or ''] if isinstance(n, ast.ImportFrom) and not n.level else [])
                    if any(name.split('.')[0] not in sys.stdlib_module_names
                           and name not in cls.EXTRA_IMPORTS for name in names):
                        return None
            for n in ast.walk(code):
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and isinstance(n.func.value, ast.Name) and n.func.value.id == 'client'
                        and n.func.attr not in cls.STORAGE_OPERATIONS):
                    return None
            declared = [line.split('#',1)[0].strip() for line in (source.python_requirements_text or '').splitlines()]
            if any(line and line.lower() != 'pillow==10.3.0' for line in declared):
                return None
            return counts
        except (KeyError, SyntaxError, TypeError, ValueError, AttributeError):
            return None

    @classmethod
    def archive_path(cls, repository) -> Path:
        return repository.root.parent / cls.ARCHIVE

    def __init__(self, repository, source, *, bucket: str):
        counts = self.capability(source)
        if counts is None:
            raise UnsupportedTestFormat('provider_asset_capability_not_supported')
        self.bucket = bucket
        path = self._path(source.candidate.source_path)
        if path.parts[0] != 'benchmarks' or path.parts[-2:] != ('python', 'function.py'):
            raise UnsupportedTestFormat('provider_asset_source_path_not_supported')
        self.benchmark_key = PurePosixPath(*path.parts[1:-2]).as_posix()
        self.data_dir = 'sebs/benchmarks-data/' + self.benchmark_key
        self.bindings = {'benchmarks_bucket': bucket,
                         'input_buckets': [f'input-{i}' for i in range(counts[0])],
                         'output_buckets': [f'output-{i}' for i in range(counts[1])]}
        archive = self.archive_path(repository)
        if not archive.is_file():
            raise UnsupportedTestFormat('provider_asset_archive_missing: ' + str(archive))
        self.assets: dict[str, bytes] = {}
        self.manifest: dict[str, str] = {}
        self.files: dict[str, bytes] = {}
        try:
            with (repository.root / 'catalog/asset_occurrences.csv').open(newline='', encoding='utf-8') as stream:
                rows = [r for r in csv.DictReader(stream) if r['source_id'] == 'sebs-data-master'
                        and r['benchmark_key'] == self.benchmark_key]
            if not rows or len(rows) > self.MAX_FILES:
                raise ValueError('Missing or excessive catalogued assets')
            total = 0
            with zipfile.ZipFile(archive) as z:
                for row in rows:
                    relative = self._path(row['source_path']).relative_to(self._path(self.benchmark_key)).as_posix()
                    self._path(relative)
                    size = int(row['size_bytes'])
                    sha = row['sha256']
                    if (relative in self.assets or not re.fullmatch('[a-f0-9]{64}', sha)
                            or not 0 <= size <= self.MAX_FILE_BYTES):
                        raise ValueError('Invalid asset metadata: ' + relative)
                    total += size
                    if total > self.MAX_TOTAL_BYTES:
                        raise ValueError('Asset size limit exceeded')
                    member = self._path(row['asset_zip_member']).as_posix()
                    info = z.getinfo(member)
                    if sum(1 for entry in z.infolist() if entry.filename == member) != 1:
                        raise ValueError('Duplicate ZIP member: ' + member)
                    if info.file_size != size or info.is_dir():
                        raise ValueError('Asset size differs from catalogue: ' + relative)
                    with z.open(info) as f:
                        data = f.read(size + 1)
                    if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
                        raise ValueError('Asset SHA-256 differs from catalogue: ' + relative)
                    self.assets[relative] = data
                    self.manifest[relative] = sha
                    self.files[f'{self.data_dir}/{relative}'] = data
            # The original compression oracle imports sebs.utils.get_project_root.
            # Stage the exact module in a namespace package, without importing
            # the full SeBS deployment framework via sebs/__init__.py.
            utils = (repository.root / 'sources/sebs-master/sebs/utils.py').resolve()
            if not utils.is_relative_to(repository.root):
                raise ValueError('Invalid SeBS utility path')
            self.files['sebs/utils.py'] = utils.read_bytes()
        except (OSError, KeyError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
            raise UnsupportedTestFormat('provider_original_assets_not_ready: ' + str(exc)) from exc
        self.uploads: list[dict] = []
        self.event: dict | None = None

    def payload(self) -> dict:
        return {'asset_data_dir': self.data_dir, 'asset_manifest': self.manifest,
                'asset_bindings': self.bindings}

    def accept_plan(self, answer: dict) -> None:
        if answer.get('status') != 'planned':
            raise ValueError('Original generator failed: ' + str(answer.get('error_message')))
        event, uploads = answer.get('event'), answer.get('uploads')
        if (type(event) is not dict or len(json.dumps(event, allow_nan=False)) > 100_000
                or type(event.get('bucket')) is not dict
                or event['bucket'].get('bucket') != self.bucket
                or event['bucket'].get('input') not in self.bindings['input_buckets']
                or event['bucket'].get('output') not in self.bindings['output_buckets']
                or type(uploads) is not list or not 0 < len(uploads) <= self.MAX_FILES):
            raise ValueError('Invalid S3 setup plan')
        seen = set()
        for item in uploads:
            if type(item) is not dict or set(item) != {'bucket_index','key','asset'}:
                raise ValueError('Invalid setup entry')
            index, key, asset = item['bucket_index'], item['key'], item['asset']
            self._path(key)
            if (type(index) is not int or not 0 <= index < len(self.bindings['input_buckets'])
                    or type(asset) is not str or asset not in self.assets or (index,key) in seen):
                raise ValueError('Invalid/duplicate asset upload')
            seen.add((index,key))
        self.event = event
        self.uploads = sorted(uploads, key=lambda x:(x['bucket_index'],x['key'],x['asset']))

    def seed(self, client) -> None:
        if self.event is None:
            raise ValueError('S3 setup plan not prepared')
        for item in self.uploads:
            key = self.bindings['input_buckets'][item['bucket_index']] + '/' + item['key']
            data = self.assets[item['asset']]
            client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def provenance(self) -> dict:
        return {'asset_archive': self.ARCHIVE, 'asset_manifest': self.manifest,
                'asset_manifest_sha256': hashlib.sha256(json.dumps(self.manifest, sort_keys=True,
                    separators=(',', ':')).encode()).hexdigest(), 'asset_uploads': self.uploads,
                'asset_count': len(self.assets), 'asset_total_bytes': sum(map(len,self.assets.values())),
                'original_sebs_utils_sha256': hashlib.sha256(self.files['sebs/utils.py']).hexdigest()}
