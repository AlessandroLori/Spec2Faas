"""Runtime provider della validazione Python -> Go, indipendente dal task ID.

La prima implementazione attiva AWS/S3 per sorgenti Python che utilizzano
il wrapper storage originale SeBS. Le altre categorie di provider non sono
implicitamente dichiarate supportate. Nessun sorgente originale e' modificato.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import uuid
from pathlib import Path

from app.providers.ProviderEnvironment import ProviderEnvironment
from app.providers.ProviderGoSdk import ProviderGoSdk
from app.providers.ProviderAssetSetup import ProviderAssetSetup
from app.providers.FlociAwsS3 import s3_client
from app.providers.ProviderHttpFixture import (ProviderHttpFixture, fixture_metadata,
                                                  verified_fixture, FixtureError)
from app.validation.HumanevalCases import UnsupportedTestFormat
from app.validation.SeBSCases import SeBSCaseExtractor
from app.validation.ProviderPythonReference import ProviderPythonReferenceRunner
from app.validation.PythonReference import ReferenceRunResult
from app.validation.OutputComparator import ComparisonResult, OutputComparator


class ProviderTestError(RuntimeError):
    """Errore di configurazione, fixture o infrastruttura provider."""


class ProviderAwareComparator(OutputComparator):
    def __init__(self, runtime: 'ProviderTestRuntime') -> None:
        self.runtime = runtime

    def compare(self, suite, result):
        if suite.comparator == self.runtime.COMPARATOR:
            return self.runtime.validate_go(suite, result)
        return super().compare(suite, result)


class ProviderTestRuntime:
    """Contratto condiviso fra orchestratore, reference, Go e Debugger.

    L'attivazione deriva dal resolver, senza flag provider manuale.
    La sorgente non determina mai un endpoint pubblico.
    Un nuovo servizio aggiungera' un proprio adattatore per setup/reset,
    lasciando invariati orchestratore e strumenti LLM.
    """

    COMPARATOR = SeBSCaseExtractor.PROVIDER_COMPARATOR
    REFERENCE_IMAGE = "spec2faas-sebs-aws-python-reference:py311-boto3-1.44.11"

    def __init__(self, *, repository, source, provider: str = "aws",
                 fixture_path: Path | None = None, docker: str = "docker") -> None:
        self.repository = repository
        self.source = source
        self.provider = provider
        self.docker = docker
        self.fixture_path = fixture_path
        self.bucket = "spec2faas-provider-" + uuid.uuid4().hex[:24]
        if provider != "aws":
            raise UnsupportedTestFormat("provider_sdk_not_yet_implemented")
        candidate = source.candidate
        if (candidate.source_id not in SeBSCaseExtractor.SOURCE_IDS
                or candidate.language != "python" or candidate.role != "benchmark_handler_source"
                or len(source.tests) != 1):
            raise UnsupportedTestFormat("provider_original_python_suite_not_available")
        # Il routing si basa sui moduli effettivamente importati dal sorgente,
        # non sul nome 120.uploader o su un ID di funzione predefinito.
        tree = ast.parse(source.code)
        if not any(isinstance(node, ast.ImportFrom) and node.level == 1
                   and any(alias.name == "storage" for alias in node.names)
                   for node in ast.walk(tree)):
            raise UnsupportedTestFormat("provider_source_has_no_supported_storage_dependency")
        wrapper_path = (repository.root / "sources" / candidate.source_id /
                        "benchmarks/wrappers/aws/python/storage.py").resolve()
        if not wrapper_path.is_relative_to(repository.root) or not wrapper_path.is_file():
            raise UnsupportedTestFormat("provider_original_wrapper_missing")
        self.wrapper = wrapper_path.read_bytes()
        self.wrapper_sha256 = hashlib.sha256(self.wrapper).hexdigest()
        self.go_sdk = ProviderGoSdk(docker=docker)
        self.asset_setup: ProviderAssetSetup | None = None
        self.bindings = {
            "benchmarks_bucket": self.bucket,
            "input_buckets": [], "output_buckets": ["output"],
        }
        self.session: ProviderEnvironment | None = None
        self.client = None
        self.http_proxy: ProviderHttpFixture | None = None
        self.reference_runner = ProviderPythonReferenceRunner(docker=docker)
        self.comparator = ProviderAwareComparator(self)
        self.files = {
            "function/__init__.py": b"",
            "function/function.py": self.source.code.encode("utf-8"),
            "function/storage.py": self.wrapper,
            "benchmark_input.py": self.source.tests[0].content.encode("utf-8"),
            "aws_config.ini": b"[default]\ns3 =\n    addressing_style = path\n",
        }
        self._image_id: str | None = None
        self.reference_image = self.REFERENCE_IMAGE
        self.extractor = SeBSCaseExtractor(generator_bindings=self.bindings)
        if ProviderAssetSetup.capability(source) is not None:
            if fixture_path is not None:
                raise ProviderTestError("--provider-fixture only applies to attested HTTP downloads")
            self.asset_setup = ProviderAssetSetup(repository, source, bucket=self.bucket)
            self.bindings = self.asset_setup.bindings
            self.files.update(self.asset_setup.files)
            self.reference_image += "-assets-v1"
            self._image_id = self._ensure_image()
            answer = self.reference_runner.plan(image=self.reference_image, files=self.files,
                payload={"protocol": "sebs_provider_reference_v1", "mode": "plan",
                         **self.asset_setup.payload()})
            self.asset_setup.accept_plan(answer)
            raw_suite = self.extractor.prepare_from_provider_event(source, self.asset_setup.event)
        else:
            raw_suite = self.extractor.prepare(source)
        from dataclasses import replace
        self.suite = replace(raw_suite, comparator=self.COMPARATOR)
        if len(self.suite.cases) != 1:
            raise UnsupportedTestFormat("provider_multiple_cases_not_yet_supported")
        event = self.suite.cases[0].args[0]
        metadata = fixture_metadata(source.tests[0].content, event)
        self.fixture_url = metadata[0] if metadata else None
        self.fixture_md5 = metadata[1] if metadata else None
        if metadata is None and self.asset_setup is None:
            raise UnsupportedTestFormat("provider_original_test_setup_not_supported")
        self.fixture: bytes | None = None
        self.fixture_sha256: str | None = None

    def go_prompt_context(self) -> str:
        return self.go_sdk.prompt_context(original_wrapper=self.wrapper.decode("utf-8"),
                                          asset_workload=self.asset_setup is not None)

    def prepare(self, source):
        if (source.candidate.candidate_id != self.source.candidate.candidate_id
                or source.candidate.source_sha256 != self.suite.source_sha256):
            raise ProviderTestError("Suite provider riferita ad altro sorgente")
        return self.suite

    @staticmethod
    def _generator_args(text: str, bindings: dict) -> list:
        tree = ast.parse(text)
        matches = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "generate_input"]
        if len(matches) != 1 or matches[0].args.kwarg or matches[0].args.vararg:
            raise UnsupportedTestFormat("provider_sebs_generator_signature_unsupported")
        available = {
            "data_dir": None, "size": "test", "benchmarks_bucket": bindings["benchmarks_bucket"],
            "input_buckets": bindings["input_buckets"],
            "output_buckets": bindings["output_buckets"],
            "upload_func": None, "nosql_func": None,
        }
        params = [arg.arg for arg in matches[0].args.args]
        if any(name not in available for name in params):
            raise UnsupportedTestFormat("provider_sebs_generator_signature_unsupported")
        return [available[name] for name in params]

    def _ensure_image(self) -> str:
        checked = subprocess.run([self.docker, "image", "inspect", "--format", "{{.Id}}",
                                  self.reference_image], capture_output=True, text=True,
                                 timeout=20, check=False)
        if checked.returncode == 0:
            return checked.stdout.strip()
        dockerfile = ("FROM python:3.11-alpine\n"
                      "RUN python -m pip install --no-cache-dir 'boto3==1.44.11'\n")
        if self.asset_setup is not None:
            dockerfile += "RUN python -m pip install --no-cache-dir " + " ".join(
                "'" + requirement + "'" for requirement in ProviderAssetSetup.REQUIREMENTS) + "\n"
        result = subprocess.run([self.docker, "build", "--pull=false", "--tag",
                                 self.reference_image, "-"], input=dockerfile,
                                capture_output=True, text=True, timeout=240, check=False)
        if result.returncode:
            raise ProviderTestError("Build immagine reference provider fallita: "
                                    + (result.stderr or result.stdout)[-600:])
        inspected = subprocess.run([self.docker, "image", "inspect", "--format", "{{.Id}}",
                                    self.reference_image], capture_output=True, text=True,
                                   timeout=20, check=False)
        if inspected.returncode:
            raise ProviderTestError("Immagine reference non disponibile dopo il build")
        return inspected.stdout.strip()

    def __enter__(self) -> 'ProviderTestRuntime':
        try:
            if self.fixture_url is not None:
                self.fixture = verified_fixture(self.fixture_url, self.fixture_md5,
                                                path=self.fixture_path)
                self.fixture_sha256 = hashlib.sha256(self.fixture).hexdigest()
            elif self.fixture_path is not None:
                raise ProviderTestError("Fixture fornita, ma il test non dichiara URL e checksum")
            if self._image_id is None:
                self._image_id = self._ensure_image()
            self.session = ProviderEnvironment(provider=self.provider, docker=self.docker)
            self.session.start()
            self.client = s3_client(self.session.host_endpoint)
            self.client.create_bucket(Bucket=self.bucket)
            if self.fixture is not None:
                self.files["http_fixture.bin"] = self.fixture
                self.http_proxy = ProviderHttpFixture(session=self.session,
                                                       url=self.fixture_url, data=self.fixture,
                                                       docker=self.docker)
                self.http_proxy.start()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc, tb):
        if self.http_proxy:
            self.http_proxy.close()
            self.http_proxy = None
        if self.client:
            try:
                self.clear_state()
                self.client.delete_bucket(Bucket=self.bucket)
            except Exception:
                pass
            self.client = None
        if self.session:
            self.session.close()
            self.session = None

    def clear_state(self) -> None:
        """Ripristina i bucket allocati prima di Python e di ogni tentativo Go."""
        if self.client is None:
            raise ProviderTestError("Floci non avviato")
        # Re-list the first page after deleting it, so even a truncated listing
        # cannot leave state from an earlier candidate behind.
        for _ in range(100):
            page = self.client.list_objects_v2(Bucket=self.bucket)
            contents = page.get("Contents", [])
            if not contents:
                if page.get("KeyCount", 0) or page.get("IsTruncated"):
                    raise ProviderTestError("Ripristino S3 non riuscito")
                return
            for item in contents:
                self.client.delete_object(Bucket=self.bucket, Key=item["Key"])
        raise ProviderTestError("Ripristino S3 non riuscito: limite pagine")

    def reset_inputs(self) -> None:
        self.clear_state()
        if self.asset_setup is not None:
            self.asset_setup.seed(self.client)

    def _payload(self, *, mode: str, output: dict | None = None) -> dict:
        payload = {
            "protocol": "sebs_provider_reference_v1", "mode": mode,
            "handler_module": "function.function", "entry_point": "handler",
            "storage_module": "function.storage",
            "expected_event": self.suite.cases[0].args[0],
        }
        if self.asset_setup is not None:
            payload.update(self.asset_setup.payload())
            payload["expected_uploads"] = self.asset_setup.uploads
        else:
            payload["generator_args"] = self._generator_args(self.source.tests[0].content, self.bindings)
        if self.fixture is not None:
            payload.update({"fixture_file": "http_fixture.bin",
                            "fixture_sha256": self.fixture_sha256,
                            "fixture_url": self.fixture_url})
        if output is not None:
            payload["output"] = output
        return payload

    def verify(self, source, suite, *, python_environment=None) -> ReferenceRunResult:
        if source.candidate.candidate_id != self.source.candidate.candidate_id or suite != self.suite:
            raise ProviderTestError("Suite provider non coerente")
        if not self.session:
            raise ProviderTestError("Sessione provider non avviata")
        self.reset_inputs()
        answer = self.reference_runner.run(session=self.session, image=self.reference_image,
                                           files=self.files, payload=self._payload(mode="reference"))
        if answer.get("storage_checked") is not True and answer.get("status") == "passed":
            raise ProviderTestError("Reference dichiarata valida senza controllo del provider")
        return ReferenceRunResult(
            candidate_id=suite.candidate_id, task_id=suite.task_id,
            source_sha256=suite.source_sha256, test_sha256=suite.test_sha256,
            status=answer["status"], total_cases=len(suite.cases),
            checked_cases=answer["checked_cases"],
            failed_case_id=None if answer["status"] == "passed" else suite.cases[0].case_id,
            error_type=answer.get("error_type"), error_message=answer.get("error_message"),
        )

    def before_go_execution(self):
        self.reset_inputs()

    def validate_go(self, suite, execution) -> ComparisonResult:
        if suite.comparator != self.COMPARATOR or suite.candidate_id != execution.candidate_id or suite.test_sha256 != execution.test_sha256:
            raise ValueError("Validazione Go con suite provider non coerente")
        if not execution.completed or len(execution.cases) != len(suite.cases):
            return ComparisonResult(suite.candidate_id, "not_executed", len(suite.cases), 0,
                                    error_message=f"Esecuzione Go non completata: {execution.status}")
        # Il validatore legge S3 DOPO l'invocazione Go; non viene mai chiamato
        # con storage=None, ne' sostituito da una comparazione JSON inventata.
        for case in execution.cases:
            if case.status != "returned" or not isinstance(case.value, dict):
                return ComparisonResult(suite.candidate_id, "execution_error", len(suite.cases), 0,
                                        case.case_id, case.error_message or "Output Go non valido")
            answer = self.reference_runner.run(
                session=self.session, image=self.reference_image, files=self.files,
                payload=self._payload(mode="validate", output=case.value),
            )
            if answer["status"] == "setup_error":
                raise ProviderTestError("Errore infrastrutturale del validatore provider: "
                                        + str(answer.get("error_message")))
            if answer["status"] == "passed" and (answer.get("storage_checked") is not True
                                                  or answer.get("checked_cases") != 1):
                raise ProviderTestError("Go dichiarato valido senza controllo del provider")
            if answer["status"] != "passed":
                return ComparisonResult(suite.candidate_id, "failed", len(suite.cases), 0,
                                        case.case_id, answer.get("error_message") or
                                        answer.get("error_type") or "Validatore originale SeBS: fallito")
        return ComparisonResult(suite.candidate_id, "passed", len(suite.cases), len(suite.cases))

    def runtime_docker_args(self) -> list[str]:
        if not self.session:
            raise ProviderTestError("Sessione provider non avviata")
        args = self.session.workload_docker_args()
        if self.http_proxy:
            args += self.http_proxy.workload_docker_args()
        return args

    def validated_go_environment(self, execution, comparison, *, runner) -> dict:
        """Attest the SDK environment of the actually validated Go revision.

        The ordinary EnvironmentPreparation is deliberately bypassed for
        provider workloads; never invent a successful generic build manifest.
        """
        suite = self.suite
        if (execution is None or comparison is None or not execution.completed
                or execution.language != "go" or execution.candidate_id != suite.candidate_id
                or execution.test_sha256 != suite.test_sha256
                or not re.fullmatch(r"[0-9a-f]{64}", execution.code_sha256)
                or tuple(c.case_id for c in execution.cases) != tuple(c.case_id for c in suite.cases)
                or not comparison.passed or comparison.candidate_id != suite.candidate_id
                or comparison.total_cases != len(suite.cases)):
            raise ProviderTestError("Revisione Go provider non attestata dalla suite")
        if (not self.go_sdk.image_id or runner.provider_runtime is not self
                or runner.build_image != self.go_sdk.image or not runner.go_modules_enabled):
            raise ProviderTestError("Ambiente SDK Go provider non attestato")
        return {
            "status": "completed", "type": "provider_go_sdk",
            "code_sha256": execution.code_sha256, "test_sha256": suite.test_sha256,
            "build_image": runner.build_image, "build_image_id": self.go_sdk.image_id,
            "go_image": runner.runtime_image, "go_modules_enabled": True,
            "go_cgo_enabled": runner.go_cgo_enabled,
            "helper_sha256": self.go_sdk.helper_sha256,
            "sdk_module": self.go_sdk.S3_MODULE + "@" + self.go_sdk.S3_VERSION,
        }

    def provenance(self) -> dict:
        return {"provider": self.provider, "service": "s3", "emulator_image_id":
                self.session.image_id if self.session else None,
                "reference_image_id": self._image_id,
                "wrapper_sha256": self.wrapper_sha256,
                "go_sdk_image_id": self.go_sdk.image_id,
                "go_sdk_helper_sha256": self.go_sdk.helper_sha256,
                "go_sdk_s3_module": self.go_sdk.S3_MODULE + "@" + self.go_sdk.S3_VERSION,
                "case_event_sha256": hashlib.sha256(json.dumps(
                    self.suite.cases[0].args[0], sort_keys=True,
                    separators=(",", ":"), allow_nan=False
                ).encode("utf-8")).hexdigest(),
                "fixture_sha256": self.fixture_sha256,
                "fixture_md5": self.fixture_md5,
                "fixture_transport": "verified_https_local_proxy_not_timing_equivalent"
                if self.fixture else "none",
                "state_reset_between_candidates": True,
                "profiling_equivalent": False,
                **(self.asset_setup.provenance() if self.asset_setup is not None else {})}
