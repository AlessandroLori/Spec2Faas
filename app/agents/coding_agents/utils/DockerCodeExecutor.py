
import asyncio
import re
import os
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import docker
from docker.errors import ImageNotFound, NotFound

from .Code_Extractors import CodeBlock


@dataclass(frozen=True)
class CodeExecutionResult:
    """
    Rappresenta il risultato dell'esecuzione Docker.

    exit_code: codice di uscita dell'ultimo comando eseguito.
    output: stdout e stderr raccolti dai comandi.
    code_file: percorso del primo file di codice creato.
    """
    exit_code: int
    output: str
    code_file: str | None = None


class DockerCodeExecutor:
    """
    Esegue blocchi di codice dentro un container Docker.

    La classe mantiene un container attivo fra le chiamate,
    consentendo ai blocchi di condividere file e librerie.
    """

    def __init__(
        self,
        image: str = "python:3-slim",
        timeout: int = 60,
        work_dir: str | Path | None = None,
    ) -> None:

        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        self.image = image
        self.timeout = timeout

        # Directory usata per salvare i file da eseguire.
        self.work_dir = (
            Path(work_dir).resolve()
            if work_dir is not None
            else None
        )

        self._temporary_dir = None
        self._client = None
        self._container = None

        # Evita esecuzioni simultanee sullo stesso container.
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """
        Prepara il workspace e avvia Docker.
        """

        if self._container is not None:
            raise RuntimeError("Docker executor already started")

        # Se non viene fornita una directory,
        # ne creiamo una temporanea dedicata.
        if self.work_dir is None:
            self._temporary_dir = tempfile.TemporaryDirectory(
                prefix="spec2faas-docker-"
            )

            self.work_dir = Path(self._temporary_dir.name)

            # Il processo nel container deve poter leggere
            # la directory montata, anche senza capability
            # Docker aggiuntive.
            # Permetti al proprietario e al gruppo di leggere,
            # scrivere e attraversare la directory temporanea.
            self.work_dir.chmod(0o770)

        else:
            self.work_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        try:
            # Connessione al daemon Docker di Ubuntu.
            self._client = await asyncio.to_thread(
                docker.from_env
            )

            await asyncio.to_thread(self._client.ping)

            # Verifichiamo la presenza dell'immagine.
            # Se manca, Docker la scarica.
            try:
                await asyncio.to_thread(
                    self._client.images.get,
                    self.image,
                )
            except ImageNotFound:
                await asyncio.to_thread(
                    self._client.images.pull,
                    self.image,
                )

            # Creiamo un container persistente.
            # Il comando shell lo mantiene attivo.
            self._container = await asyncio.to_thread(
                self._client.containers.run,
                self.image,
                ["sh", "-c", "while :; do sleep 3600; done"],
                detach=True,
                auto_remove=True,
                user=f"0:{os.getgid()}",
                volumes={
                    str(self.work_dir): {
                        "bind": "/workspace",
                        "mode": "rw",
                    }
                },
                working_dir="/workspace",
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
            )

        except Exception:
            # Pulizia in caso di inizializzazione fallita.
            await self.stop()
            raise

    async def execute_code_blocks(
        self,
        code_blocks: list[CodeBlock],
    ) -> CodeExecutionResult:
        """
        Esegue i blocchi nell'ordine ricevuto.

        Ogni blocco viene salvato in un file e avviato
        all'interno del container.

        Se un comando fallisce, i successivi non vengono
        eseguiti.
        """

        if not code_blocks:
            raise ValueError("No code blocks to execute")

        async with self._lock:

            if self._container is None or self.work_dir is None:
                raise RuntimeError("Docker executor is not started")

            outputs = []
            exit_code = 0
            first_file = None

            # Linguaggi supportati.
            commands = {
                "python": "python",
                "sh": "sh",
                "shell": "sh",
                "bash": "bash",
            }

            for block in code_blocks:

                language = block.language.lower().strip()

                if language not in commands:
                    return CodeExecutionResult(
                        exit_code=1,
                        output="".join(outputs)
                        + f"unknown language {language}",
                        code_file=first_file,
                    )

                code = block.code

                # Manteniamo la riduzione della verbosità
                # dei comandi pip install.
                if language in ("sh", "shell", "bash"):
                    code = re.sub(
                        r"(?m)^(\s*)(pip\s+install)\b",
                        r"\1\2 -qqq",
                        code,
                    )

                extension = (
                    "py" if language == "python" else language
                )

                # Nome deterministico basato sul codice.
                filename = (
                    "tmp_code_"
                    + sha256(code.encode("utf-8")).hexdigest()
                    + "."
                    + extension
                )

                code_path = self.work_dir / filename
                code_path.write_text(
                    code,
                    encoding="utf-8",
                )

                # Rende il file leggibile al container,
                # anche con le capability rimosse.
                code_path.chmod(0o644)

                if first_file is None:
                    first_file = str(code_path)

                # Avvio del programma DENTRO Docker,
                # non direttamente sul computer Ubuntu.
                result = await asyncio.to_thread(
                    self._container.exec_run,
                    [
                        "timeout",
                        str(self.timeout),
                        commands[language],
                        filename,
                    ],
                    workdir="/workspace",
                )

                exit_code = result.exit_code

                output = result.output.decode(
                    "utf-8",
                    errors="replace",
                )

                # Codice 124: timeout del comando.
                if exit_code == 124:
                    output += "\n Timeout"

                outputs.append(output)

                # Interrompiamo la sequenza in caso di errore.
                if exit_code != 0:
                    break

            return CodeExecutionResult(
                exit_code=exit_code,
                output="".join(outputs),
                code_file=first_file,
            )


    async def stop(self) -> None:
        """
        Ferma il container e libera le risorse temporanee.

        Prima di eliminare il workspace temporaneo, ripristina
        i permessi di gruppo sulle directory create dal
        processo root all'interno del container.

        Questo permette all'utente Ubuntu di eliminare
        correttamente i file prodotti durante l'esecuzione.
        """

        async with self._lock:

            container = self._container
            self._container = None

            try:
                if container is not None:
                    try:
                        if self._temporary_dir is not None:

                            # Il container esegue comandi come root.
                            # Alcune sottodirectory possono quindi
                            # appartenere a UID 0 ed essere non
                            # scrivibili dall'utente Ubuntu.
                            #
                            # Ripristiniamo i permessi SOLO sulle
                            # directory possedute dal processo Docker,
                            # senza modificare i file dell'utente host
                            # e senza seguire link simbolici.
                            permission_script = (
                                "import os\n"
                                "import stat\n"
                                "for root, dirs, _ in os.walk('/workspace'):\n"
                                "    for name in dirs:\n"
                                "        path = os.path.join(root, name)\n"
                                "        if os.path.islink(path):\n"
                                "            continue\n"
                                "        info = os.stat(path)\n"
                                "        if info.st_uid == os.geteuid() "
                                "and info.st_gid == os.getegid():\n"
                                "            mode = stat.S_IMODE(info.st_mode)\n"
                                "            os.chmod(path, mode | stat.S_IRWXG)\n"
                            )

                            result = await asyncio.to_thread(
                                container.exec_run,
                                ["python", "-c", permission_script],
                                workdir="/workspace",
                            )

                            if result.exit_code != 0:
                                raise RuntimeError(
                                    "Unable to restore Docker workspace "
                                    "permissions: "
                                    + result.output.decode(
                                        "utf-8",
                                        errors="replace",
                                    )
                                )

                    finally:
                        # Il container viene fermato anche se
                        # il ripristino dei permessi fallisce.
                        try:
                            await asyncio.to_thread(
                                container.stop,
                                timeout=1,
                            )
                        except NotFound:
                            pass

            finally:
                try:
                    if self._client is not None:
                        await asyncio.to_thread(
                            self._client.close
                        )
                        self._client = None

                finally:
                    if self._temporary_dir is not None:
                        self._temporary_dir.cleanup()
                        self._temporary_dir = None
                        self.work_dir = None

