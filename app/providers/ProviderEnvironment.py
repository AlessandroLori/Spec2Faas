"""Ambienti cloud emulati e isolati, indipendenti dal dataset e dalla lingua.

Questa componente gestisce soltanto il ciclo di vita dell'emulatore. Non
esegue codice del corpus e non cambia gli oracoli dei test. L'SDK per ciascun
provider resta di competenza di un adattatore separato.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderSpecification:
    """Configurazione esplicita di un emulatore, senza input shell arbitrari."""

    provider: str
    image: str
    port: int
    health_path: str
    alias: str
    storage_environment: tuple[str, str]

    def __post_init__(self) -> None:
        if (not re.fullmatch(r"[a-z][a-z0-9-]{0,30}", self.provider)
                or not re.fullmatch(r"[a-z][a-z0-9-]{0,30}", self.alias)
                or not re.fullmatch(r"[a-z0-9][a-z0-9./:_-]{1,150}", self.image)
                or not 1024 <= self.port <= 65535
                or not re.fullmatch(r"/[A-Za-z0-9/_-]{1,100}", self.health_path)):
            raise ValueError("Configurazione dell'emulatore non valida")


# La rete/lifecycle e' condivisa; il supporto agli SDK verra' aggiunto per
# servizio. Le voci non garantiscono ancora l'eseguibilita' di benchmark Azure,
# GCP od OCI e non forzano l'avvio di emulatori non richiesti.
FLOCI_PROVIDERS = {
    "aws": ProviderSpecification("aws", "floci/floci:latest", 4566,
                                 "/_floci/health", "floci-aws",
                                 ("FLOCI_STORAGE_MODE", "memory")),
    "azure": ProviderSpecification("azure", "floci/floci-az:latest", 4577,
                                   "/_floci/health", "floci-azure",
                                   ("FLOCI_AZ_STORAGE_MODE", "memory")),
    "gcp": ProviderSpecification("gcp", "floci/floci-gcp:latest", 4588,
                                 "/_floci-gcp/health", "floci-gcp",
                                 ("FLOCI_GCP_STORAGE_MODE", "memory")),
    "oci": ProviderSpecification("oci", "floci/floci-oci:latest", 4599,
                                 "/_floci-oci/health", "floci-oci",
                                 ("FLOCI_OCI_STORAGE_MODE", "memory")),
}


class ProviderEnvironmentError(RuntimeError):
    """Errore infrastrutturale: non attribuibile a Python/Go o al Debugger."""


class ProviderEnvironment:
    """Gestisce un emulatore per sessione con rete Docker e stato indipendenti.

    La rete workload e' interna (senza accesso Internet); la rete di controllo
    separata consente di pubblicare l'endpoint host soltanto su 127.0.0.1.
    Soltanto Floci e' collegato a entrambe. I futuri container Python e Go
    utilizzano esclusivamente la rete workload e il *container_endpoint*.
    Le risorse create vengono eliminate anche in caso di avvio incompleto.
    """

    def __init__(self, *, provider: str = "aws", docker: str = "docker",
                 health_timeout_seconds: float = 30.0) -> None:
        if provider not in FLOCI_PROVIDERS:
            raise ValueError(f"Provider non supportato: {provider}")
        if not docker or not isinstance(docker, str):
            raise ValueError("Eseguibile Docker non valido")
        if not 0 < health_timeout_seconds <= 300:
            raise ValueError("Timeout health non valido")
        self.spec = FLOCI_PROVIDERS[provider]
        self.docker = docker
        self.health_timeout_seconds = health_timeout_seconds
        suffix = uuid.uuid4().hex[:16]
        self.network_name = f"spec2faas-provider-net-{suffix}"
        self.control_network_name = f"spec2faas-provider-control-{suffix}"
        self.container_name = f"spec2faas-provider-{self.spec.provider}-{suffix}"
        self.host_endpoint: str | None = None
        self.image_id: str | None = None
        self._network_created = False
        self._control_network_created = False
        self._container_created = False

    @property
    def container_endpoint(self) -> str:
        return f"http://{self.spec.alias}:{self.spec.port}"

    def workload_docker_args(self) -> list[str]:
        """Opzioni da applicare ai futuri runner Python e Go, non al builder.

        Non viene mai fornito un endpoint AWS pubblico. Le opzioni sono
        disponibili solo durante la sessione attiva.
        """
        if not self._network_created or not self._container_created:
            raise ProviderEnvironmentError("Emulatore provider non avviato")
        args = ["--network", self.network_name]
        if self.spec.provider == "aws":
            variables = {
                "AWS_ENDPOINT_URL": self.container_endpoint,
                "AWS_ENDPOINT_URL_S3": self.container_endpoint,
                "AWS_DEFAULT_REGION": "us-east-1",
                "AWS_REGION": "us-east-1",
                "AWS_ACCESS_KEY_ID": "test",
                "AWS_SECRET_ACCESS_KEY": "test",
                "AWS_EC2_METADATA_DISABLED": "true",
                "NO_PROXY": self.spec.alias + ",localhost,127.0.0.1",
            }
            for name, value in variables.items():
                args.extend(["--env", f"{name}={value}"])
        return args

    def _docker(self, *args: str, timeout: float = 30) -> str:
        try:
            completed = subprocess.run(
                [self.docker, *args], capture_output=True, text=True,
                timeout=timeout, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProviderEnvironmentError(f"Docker non disponibile: {exc}") from exc
        if completed.returncode:
            raise ProviderEnvironmentError(
                f"Docker {' '.join(args[:2])} fallito: "
                f"{(completed.stderr or completed.stdout)[-600:]}"
            )
        return completed.stdout.strip()

    def _host_port(self) -> int:
        output = self._docker("port", self.container_name, f"{self.spec.port}/tcp")
        # Con -p 127.0.0.1::4566 il daemon assegna una porta libera.
        matches = re.findall(r"(?m)^127\.0\.0\.1:(\d+)\s*$", output)
        if len(matches) != 1:
            raise ProviderEnvironmentError("Mapping su loopback Docker non riconosciuto")
        return int(matches[0])

    def _assert_container_running(self) -> None:
        """Fornisce la causa reale di un exit prematuro, prima di leggere -p.

        `docker run -d` può restituire 0 anche se l'entrypoint termina subito:
        in quel caso Docker non assegna una porta effettiva al container.
        """
        status = self._docker("inspect", "--format", "{{.State.Status}}",
                              self.container_name)
        if status == "running":
            return
        try:
            logs = self._docker("logs", "--tail", "20", self.container_name)
        except ProviderEnvironmentError as exc:
            logs = f"log non disponibile: {exc}"
        raise ProviderEnvironmentError(
            f"Emulatore {self.spec.provider} non avviato: "
            f"stato Docker={status!r}; log={logs[-800:]}"
        )

    def start(self) -> "ProviderEnvironment":
        if self._network_created or self._control_network_created or self._container_created:
            raise ProviderEnvironmentError("Sessione provider gia' avviata")
        try:
            self.image_id = self._docker("image", "inspect", "--format", "{{.Id}}",
                                         self.spec.image)
            # I workload si collegano solo alla rete interna: non ottengono
            # una route verso Internet, neppure attraverso la rete di controllo.
            self._docker("network", "create", "--internal", "--driver", "bridge",
                         self.network_name)
            self._network_created = True
            # Docker non rende disponibile -p per container collegati soltanto
            # a un bridge --internal. Pubblicare su una seconda rete bridge,
            # dedicata all'emulatore e mai assegnata ai workload.
            self._docker("network", "create", "--driver", "bridge",
                         self.control_network_name)
            self._control_network_created = True
            self._docker(
                "run", "--detach", "--pull=never", "--name", self.container_name,
                "--network", self.control_network_name,
                "--publish", f"127.0.0.1::{self.spec.port}",
                # Floci usa `chroot --userspec` nell'entrypoint per
                # abbandonare root: servono chroot, setuid e setgid.
                # Non ripristinare tutte le capability e non usare --privileged.
                "--cap-drop=ALL", "--cap-add=SYS_CHROOT",
                "--cap-add=SETUID", "--cap-add=SETGID",
                "--security-opt=no-new-privileges",
                "--memory=1g", "--pids-limit=256",
                "--env", f"{self.spec.storage_environment[0]}={self.spec.storage_environment[1]}",
                self.spec.image,
            )
            self._container_created = True
            self._docker("network", "connect", "--alias", self.spec.alias,
                         self.network_name, self.container_name)
            self._assert_container_running()
            self.host_endpoint = f"http://127.0.0.1:{self._host_port()}"
            self._wait_healthy()
            return self
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _health_payload_ready(data: object, provider: str) -> bool:
        """Accetta le due varianti documentate del contratto health Floci.

        Floci AWS 2.5.0 puo' restituire `version` e `services`, senza
        `status`/`cloud`. Se presenti, i due campi vanno comunque verificati.
        Non assumiamo che ogni ambiente provider utilizzi S3: i servizi
        richiesti dal benchmark vengono verificati nei rispettivi adattatori.
        """
        if not isinstance(data, dict):
            return False
        services = data.get("services")
        if not isinstance(services, dict) or not any(
            value == "running" for value in services.values()
        ):
            return False
        status = data.get("status")
        if status is not None and status not in ("UP", "up", "ok", "healthy", "ready", "running"):
            return False
        cloud = data.get("cloud")
        if cloud is not None and cloud != provider:
            return False
        if status is None or cloud is None:
            version = data.get("version")
            return isinstance(version, str) and bool(version.strip())
        return True

    def _wait_healthy(self) -> None:
        assert self.host_endpoint is not None
        deadline = time.monotonic() + self.health_timeout_seconds
        last_error = "nessuna risposta"
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    self.host_endpoint + self.spec.health_path, timeout=2,
                ) as response:
                    if response.status == 200:
                        data = json.loads(response.read(16_384))
                        if self._health_payload_ready(data, self.spec.provider):
                            return
                        services = data.get("services") if isinstance(data, dict) else None
                        running = (
                            [name for name, value in services.items() if value == "running"]
                            if isinstance(services, dict) else []
                        )
                        last_error = (
                            f"health JSON non pronto per {self.spec.provider}: "
                            f"status={data.get('status')!r}, "
                            f"cloud={data.get('cloud')!r}, "
                            f"servizi_attivi={running[:5]!r}"
                        ) if isinstance(data, dict) else "health JSON non valido"
                    else:
                        last_error = f"HTTP {response.status}"
            except (OSError, ValueError, urllib.error.URLError) as exc:
                last_error = str(exc)
            time.sleep(0.2)
        raise ProviderEnvironmentError(f"Health di Floci non raggiungibile: {last_error}")

    def close(self) -> None:
        if self._container_created:
            try:
                self._docker("rm", "-f", self.container_name, timeout=15)
            except ProviderEnvironmentError:
                pass
            self._container_created = False
        if self._network_created:
            try:
                self._docker("network", "rm", self.network_name, timeout=15)
            except ProviderEnvironmentError:
                pass
            self._network_created = False
        if self._control_network_created:
            try:
                self._docker("network", "rm", self.control_network_name,
                             timeout=15)
            except ProviderEnvironmentError:
                pass
            self._control_network_created = False
        self.host_endpoint = None
        self.image_id = None

    def __enter__(self) -> "ProviderEnvironment":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
