"""Fixture HTTPS attestata per test provider in rete Docker senza Internet.

L'URL del benchmark non cambia. Un proxy CONNECT consente ai client Go che
rispettano HTTPS_PROXY di leggere unicamente i byte originali con checksum
verificato. Il certificato e' effimero e fidato solo dal container di test.
Non usare questo percorso per profilazione o deployment.
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from app.providers.ProviderEnvironment import ProviderEnvironment, ProviderEnvironmentError


class FixtureError(RuntimeError):
    """Fixture non attestabile oppure proxy Docker non avviabile."""


def fixture_metadata(test_code: str, event: dict) -> tuple[str, str] | None:
    """Individua URL e checksum *letterali* originali, senza eseguire input.py."""
    url = event.get("object", {}).get("url") if isinstance(event, dict) else None
    size = event.get("object", {}).get("size") if isinstance(event, dict) else None
    if not isinstance(url, str) or not isinstance(size, str):
        return None
    tree = ast.parse(test_code)
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "expected_checksums"):
            try:
                values = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                return None
            expected = values.get(size) if type(values) is dict else None
            if type(expected) is str and re.fullmatch(r"[a-f0-9]{32}", expected):
                parsed = urllib.parse.urlsplit(url)
                if (parsed.scheme == "https" and parsed.port in (None, 443)
                        and parsed.hostname and parsed.hostname.endswith((".wikimedia.org", ".github.com", ".pytorch.org"))
                        and parsed.username is None and parsed.password is None
                        and not parsed.fragment):
                    return url, expected
    return None


def verified_fixture(url: str, md5: str, *, path: Path | None = None,
                     max_bytes: int = 2_000_000) -> bytes:
    """Non modifica un oracolo SeBS: il byte stream deve avere MD5 originale."""
    if type(md5) is not str or not re.fullmatch(r"[a-f0-9]{32}", md5):
        raise FixtureError("Checksum MD5 originale assente")
    if path is not None:
        local = Path(path).expanduser().resolve()
        if not local.is_file() or local.stat().st_size > max_bytes:
            raise FixtureError("Fixture locale mancante o troppo grande")
        data = local.read_bytes()
    else:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "SeBS/1.2"})
            with urllib.request.urlopen(request, timeout=25) as response:
                if urllib.parse.urlsplit(response.url).hostname != urllib.parse.urlsplit(url).hostname:
                    raise FixtureError("Redirect HTTP verso host diverso")
                data = response.read(max_bytes + 1)
        except (OSError, TimeoutError) as exc:
            raise FixtureError("Download della fixture originale fallito") from exc
    if not data or len(data) > max_bytes or hashlib.md5(data).hexdigest() != md5:
        raise FixtureError("Fixture diversa dal checksum MD5 del test SeBS originale")
    return data


class ProviderHttpFixture:
    """Proxy HTTPS effimero raggiungibile soltanto dalla rete workload Floci."""

    def __init__(self, *, session: ProviderEnvironment, url: str, data: bytes,
                 docker: str = "docker") -> None:
        if not session.host_endpoint:
            raise ProviderEnvironmentError("Sessione provider non avviata")
        if not isinstance(data, bytes) or not data:
            raise ValueError("Fixture vuota")
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.port not in (None, 443):
            raise ValueError("Solo URL HTTPS validi sono ammessi")
        self.session = session
        self.url = url
        self.data = data
        self.docker = docker
        self.container = "reliq-fixture-" + uuid.uuid4().hex[:16]
        self.alias = self.container
        self._temp: tempfile.TemporaryDirectory | None = None
        self.root: Path | None = None
        self._started = False

    def start(self) -> "ProviderHttpFixture":
        self._temp = tempfile.TemporaryDirectory(prefix="reliq_http_fixture_")
        self.root = Path(self._temp.name)
        self.root.chmod(0o755)  # montaggio leggibile dal processo container UID 65534
        server_script = Path(__file__).with_name("ProviderHttpFixtureServer.py")
        for filename, content in (("proxy.py", server_script.read_bytes()),
                                  ("fixture.bin", self.data), ("url.txt", self.url.encode())):
            target = self.root / filename
            target.write_bytes(content)
            target.chmod(0o644)
        hostname = urllib.parse.urlsplit(self.url).hostname
        result = subprocess.run([
            "openssl", "req", "-x509", "-nodes", "-newkey", "rsa:2048", "-days", "1",
            "-keyout", str(self.root / "key.pem"), "-out", str(self.root / "cert.pem"),
            "-subj", f"/CN={hostname}", "-addext", f"subjectAltName=DNS:{hostname}"
        ], text=True, capture_output=True, timeout=30, check=False)
        if result.returncode:
            self.close()
            raise FixtureError("Certificato temporaneo per fixture non generabile")
        for name in ("key.pem", "cert.pem"):
            (self.root / name).chmod(0o644)  # UID 65534, solo per la durata del test
        cmd = [
            self.docker, "run", "-d", "--rm", "--pull=never", "--name", self.container,
            "--network", self.session.network_name, "--network-alias", self.alias,
            "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--user=65534:65534", "--pids-limit=32", "--memory=128m", "--cpus=0.5",
            "--mount", f"type=bind,src={self.root},dst=/opt/fixture,readonly",
            "python:3.11-alpine", "python", "-I", "/opt/fixture/proxy.py",
        ]
        try:
            launched = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
            if launched.returncode:
                raise FixtureError("Docker non ha avviato il proxy: " + launched.stderr[-500:])
            self._started = True
            # Controllo d'avvio nel container, senza pubblicare una porta host.
            for _ in range(25):
                ready = subprocess.run([
                    self.docker, "exec", self.container, "python", "-c",
                    "import socket;s=socket.create_connection(('127.0.0.1',8181),1);s.close()"
                ], capture_output=True, timeout=5)
                if ready.returncode == 0:
                    return self
                time.sleep(.1)
            raise FixtureError("Proxy HTTPS non in ascolto nel container")
        except BaseException:
            self.close()
            raise

    def workload_docker_args(self) -> list[str]:
        if not self._started or self.root is None:
            raise FixtureError("Proxy HTTP non avviato")
        return [
            "--env", f"HTTPS_PROXY=http://{self.alias}:8181",
            "--env", f"https_proxy=http://{self.alias}:8181",
            "--env", "SSL_CERT_FILE=/opt/reliq-fixture-cert.pem",
            "--mount", (f"type=bind,src={self.root / 'cert.pem'},"
                        "dst=/opt/reliq-fixture-cert.pem,readonly"),
        ]

    def close(self) -> None:
        if self._started:
            subprocess.run([self.docker, "rm", "-f", self.container],
                           capture_output=True, timeout=15, check=False)
            self._started = False
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None
            self.root = None

    def __enter__(self) -> "ProviderHttpFixture":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
