"""Esecuzione di una reference Python in Docker con un provider emulato.

Il runner non conosce SeBS: riceve file immutabili, un payload e un ambiente
provider già avviato. Il worker applica l'oracolo della suite originale.
L'eventuale fixture HTTP deve essere preparata e verificata dal chiamante.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Mapping

from app.providers.ProviderEnvironment import ProviderEnvironment, ProviderEnvironmentError


class ProviderReferenceError(RuntimeError):
    """Errore infrastrutturale o di preparazione, distinto da un test fallito."""


class ProviderPythonReferenceRunner:
    """Lancia una reference su rete Docker isolata e con credenziali locali fittizie."""

    MAX_FILE_BYTES = 8_000_000
    MAX_TOTAL_BYTES = 12_000_000
    IMAGE_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._/:-]{1,180}\Z")

    def __init__(self, *, docker: str = "docker", timeout_seconds: float = 90) -> None:
        if not docker or not isinstance(docker, str):
            raise ValueError("Comando Docker non valido")
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 3600:
            raise ValueError("Timeout della reference non valido")
        self.docker = docker
        self.timeout_seconds = timeout_seconds
        self.worker = Path(__file__).with_name("ProviderPythonReferenceWorker.py").resolve()

    @classmethod
    def _safe_files(cls, files: Mapping[str, bytes]) -> None:
        if not files or len(files) > 64:
            raise ValueError("Elenco dei file Python non valido")
        size = 0
        for name, value in files.items():
            path = PurePosixPath(name)
            if (type(name) is not str or path.is_absolute() or not path.parts
                    or ".." in path.parts or any(not p or p.startswith(".") for p in path.parts)
                    or type(value) is not bytes or len(value) > cls.MAX_FILE_BYTES):
                raise ValueError(f"File della reference non ammesso: {name!r}")
            size += len(value)
        if size > cls.MAX_TOTAL_BYTES:
            raise ValueError("Dimensione complessiva della reference eccessiva")

    def run(self, *, session: ProviderEnvironment, image: str,
            files: Mapping[str, bytes], payload: dict) -> dict:
        if not isinstance(session, ProviderEnvironment) or session.host_endpoint is None:
            raise ProviderEnvironmentError("Sessione provider non avviata")
        if type(image) is not str or not self.IMAGE_PATTERN.fullmatch(image):
            raise ValueError("Immagine Docker Python non valida")
        self._safe_files(files)
        if not isinstance(payload, dict):
            raise ValueError("Payload reference non valido")
        if not self.worker.is_file():
            raise ProviderReferenceError("Worker Python provider mancante")
        with tempfile.TemporaryDirectory(prefix="reliq_provider_reference_") as temp:
            # Il worker distribuito può avere permessi 0600 (ad es. da ZIP):
            # il container usa UID 65534 e non potrebbe leggere il bind mount.
            # Copiamo in una posizione privata e rendiamo leggibile solo la
            # copia temporanea, senza cambiare i permessi del progetto.
            staged_worker = Path(temp) / "worker.py"
            shutil.copyfile(self.worker, staged_worker)
            staged_worker.chmod(0o644)
            directory = Path(temp) / "source"
            directory.mkdir(mode=0o755)
            for relative, content in files.items():
                destination = directory.joinpath(*PurePosixPath(relative).parts)
                destination.parent.mkdir(parents=True, mode=0o755, exist_ok=True)
                destination.write_bytes(content)
                destination.chmod(0o644)
            command = [
                self.docker, "run", "--rm", "--pull=never",
                *session.workload_docker_args(),
                "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                "--cpus=1", "--memory=1g", "--pids-limit=64", "--user=65534:65534",
                "--tmpfs=/tmp:rw,nosuid,noexec,size=64m", "-i",
                "--mount", f"type=bind,src={directory.resolve()},dst=/opt/reliq-source,readonly",
                "--mount", f"type=bind,src={staged_worker},dst=/opt/reliq-worker.py,readonly",
                "--workdir", "/opt/reliq-source", image, "python", "-I", "/opt/reliq-worker.py",
            ]
            try:
                outcome = subprocess.run(
                    command, input=json.dumps(payload, ensure_ascii=False, allow_nan=False),
                    text=True, capture_output=True, timeout=self.timeout_seconds, check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise ProviderReferenceError(f"Reference Docker non eseguibile: {exc}") from exc
        if outcome.returncode:
            raise ProviderReferenceError(
                f"Reference Docker uscita {outcome.returncode}: "
                f"{(outcome.stderr or outcome.stdout)[-1000:]}"
            )
        try:
            answer = json.loads(outcome.stdout)
        except (ValueError, TypeError) as exc:
            raise ProviderReferenceError(
                f"Risposta della reference non JSON: {outcome.stdout[-400:]}"
            ) from exc
        if (not isinstance(answer, dict) or answer.get("status") not in
                ("passed", "failed", "setup_error") or type(answer.get("checked_cases")) is not int):
            raise ProviderReferenceError("Risposta strutturata della reference non valida")
        return answer
