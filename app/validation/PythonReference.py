import hashlib
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.redataset.Models import FunctionSource

from .HumanevalCases import PreparedSuite
from .ReFaaSCases import ReFaaSCaseExtractor
from .SeBSCases import SeBSCaseExtractor


@dataclass(frozen=True, slots=True)
class ReferenceRunResult:
    """Esito della verifica dei casi sulla funzione Python originale."""

    candidate_id: str
    task_id: str
    source_sha256: str
    test_sha256: str
    status: str
    total_cases: int
    checked_cases: int
    failed_case_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def passed(self) -> bool:
        """Indica che tutti i casi previsti sono stati verificati."""
        return self.status == "passed" and self.checked_cases == self.total_cases


class PythonReferenceRunner:
    """Verifica i casi HumanEval+ in un processo Python separato."""

    SOURCE_ID = "spec2faas-humanevalplus-164"
    COMPARATOR = "humaneval_plus_assertion"

    def __init__(self, *, timeout_seconds: float = 60.0, docker: str = "docker") -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Il timeout deve essere positivo e finito")
        if not docker:
            raise ValueError("Eseguibile Docker non valido")
        self.timeout_seconds = timeout_seconds
        self.docker = docker
        self._worker_path = Path(__file__).with_name("PythonReferenceWorker.py")

    def verify(self, source: FunctionSource, suite: PreparedSuite,
               *, python_environment: dict | None = None) -> ReferenceRunResult:
        """Verifica la reference sull'host o nel Docker Python gia' preparato.

        Il Docker usa la stessa immagine risolta per la profilazione. L'immagine
        e' scelta dal manifest del preparatore, mai dal sorgente della funzione.
        """
        candidate = source.candidate
        is_refaas = (suite.comparator == ReFaaSCaseExtractor.COMPARATOR
                     and candidate.source_id == ReFaaSCaseExtractor.SOURCE_ID)
        is_sebs = (suite.comparator == SeBSCaseExtractor.COMPARATOR
                   and candidate.source_id in SeBSCaseExtractor.SOURCE_IDS)
        is_sebs_derived = (suite.comparator == SeBSCaseExtractor.DERIVED_COMPARATOR
                           and candidate.source_id in SeBSCaseExtractor.SOURCE_IDS)
        if not (is_refaas or is_sebs or is_sebs_derived) and (candidate.source_id != self.SOURCE_ID
                                           or suite.comparator != self.COMPARATOR):
            raise ValueError("Comparatore o origine dei test non supportati")
        entry = suite.entry_point or candidate.entry_point
        if not entry:
            raise ValueError("Entry point della funzione non disponibile")
        if (suite.candidate_id != candidate.candidate_id
                or (candidate.source_task_id and suite.task_id != candidate.source_task_id)
                or suite.source_sha256 != candidate.source_sha256):
            raise ValueError("I casi non appartengono alla funzione selezionata")
        if not suite.cases:
            raise ValueError("La suite non contiene casi")

        source_hash = hashlib.sha256(source.code.encode("utf-8")).hexdigest()
        if source_hash != candidate.source_sha256:
            raise ValueError("Il codice non corrisponde all'hash registrato")

        paths = suite.test_paths or (suite.test_path,)
        tests = [test for path in paths for test in source.tests if test.path == path]
        if len(tests) != len(paths) or len(set(paths)) != len(paths):
            raise ValueError("I test della suite non sono disponibili")
        test_hash = (ReFaaSCaseExtractor.tests_sha256(tests) if is_refaas
                     else hashlib.sha256(tests[0].content.encode("utf-8")).hexdigest())
        if test_hash != suite.test_sha256:
            raise ValueError("I test non corrispondono all'hash della suite")

        payload = {
            "code": source.code,
            "tests": tests[0].content if not is_refaas else None,
            "comparator": suite.comparator,
            "entry_point": entry,
            "cases": [
                {"case_id": case.case_id, "args": case.args,
                 "expected": case.expected, "atol": case.atol}
                for case in suite.cases
            ],
        }
        # Il worker riceve codice e test solo da stdin: il sorgente non viene
        # copiato/modificato e non viene installato nulla nell'host.
        # -I ignora PYTHONPATH. Docker disabilita rete, privilegi e rootfs RW.
        command = [sys.executable, "-I", str(self._worker_path)]
        if python_environment is not None:
            if (not isinstance(python_environment, dict)
                    or python_environment.get("status") != "completed"
                    or python_environment.get("language") != "python"
                    or python_environment.get("code_sha256") != source_hash
                    or not isinstance(python_environment.get("python_image"), str)
                    or not python_environment["python_image"]):
                raise ValueError("Manifest Python non coerente con la reference")
            command = [
                self.docker, "run", "--rm", "--pull=never", "--network=none",
                "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                "--cpus=1", "--memory=1g", "--pids-limit=64", "--user=65534:65534",
                "--tmpfs=/tmp:rw,nosuid,noexec,size=64m", "-i",
                "--mount", f"type=bind,src={self._worker_path.resolve()},dst=/opt/reliq-reference-worker.py,readonly",
                python_environment["python_image"], "python", "-I",
                "/opt/reliq-reference-worker.py",
            ]
        with tempfile.TemporaryDirectory(prefix="redataset_reference_") as workdir:
            try:
                completed = subprocess.run(
                    command, input=json.dumps(payload, ensure_ascii=False, allow_nan=False),
                    capture_output=True, text=True, cwd=workdir,
                    timeout=self.timeout_seconds, check=False,
                )
            except subprocess.TimeoutExpired:
                return self._result(suite, "timeout", 0, error_type="TimeoutExpired")
            except OSError as exc:
                return self._result(suite, "worker_error", 0,
                                    error_type=type(exc).__name__, error_message=str(exc)[:400])

        if completed.returncode != 0:
            return self._result(
                suite, "worker_error", 0, error_type="WorkerExited",
                error_message=(f"Codice di uscita {completed.returncode}: "
                               + (completed.stderr or "")[-800:]),
            )
        try:
            reply = json.loads(completed.stdout)
        except (TypeError, ValueError):
            return self._result(
                suite, "worker_error", 0, error_type="InvalidWorkerResponse"
            )
        if not isinstance(reply, dict):
            return self._result(
                suite, "worker_error", 0, error_type="InvalidWorkerResponse"
            )
        status = reply.get("status")
        count = reply.get("checked_cases")
        if (status not in ("passed", "failed", "setup_error")
                or type(count) is not int or not 0 <= count <= len(suite.cases)
                or (status == "passed" and count != len(suite.cases))):
            return self._result(
                suite, "worker_error", 0, error_type="InvalidWorkerResponse"
            )
        return self._result(
            suite, status, count,
            failed_case_id=reply.get("failed_case_id"),
            error_type=reply.get("error_type"),
            error_message=reply.get("error_message"),
        )

    @staticmethod
    def _result(
        suite: PreparedSuite, status: str, checked_cases: int,
        *, failed_case_id: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> ReferenceRunResult:
        return ReferenceRunResult(
            candidate_id=suite.candidate_id,
            task_id=suite.task_id,
            source_sha256=suite.source_sha256,
            test_sha256=suite.test_sha256,
            status=status,
            total_cases=len(suite.cases),
            checked_cases=checked_cases,
            failed_case_id=failed_case_id,
            error_type=error_type,
            error_message=error_message,
        )
