
import unittest
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.agents.coding_agents.utils.Code_Extractors import CodeBlock
from app.agents.coding_agents.utils.DockerCodeExecutor import DockerCodeExecutor


class FakeContainer:
    """
    Simula un container e registra i comandi ricevuti.
    """

    def __init__(self, work_dir):
        self.work_dir = Path(work_dir)
        self.commands = []
        self.stopped = False


    def exec_run(self, command, workdir):
        self.commands.append(list(command))

        # Simulazione del ripristino dei permessi
        # eseguito durante la chiusura del container.
        if command[:2] == ["python", "-c"]:
            return SimpleNamespace(
                exit_code=0,
                output=b"",
            )

        # Simulazione delle normali esecuzioni di codice.
        source = (
            self.work_dir / command[-1]
        ).read_text(encoding="utf-8")

        if "raise RuntimeError" in source:
            return SimpleNamespace(
                exit_code=1,
                output=b"RuntimeError: simulated\n",
            )

        return SimpleNamespace(
            exit_code=0,
            output=b"OK\n",
        )


    def stop(self, timeout):
        self.stopped = True


class FakeClient:
    """
    Simula le operazioni necessarie del Docker SDK.
    """

    def __init__(self):
        self.images = SimpleNamespace(
            get=lambda image: None
        )

        self.containers = SimpleNamespace(
            run=self._run
        )

        self.container = None
        self.options = None
        self.closed = False

    def ping(self):
        return True

    def _run(self, image, command, **kwargs):
        self.options = kwargs

        self.container = FakeContainer(
            next(iter(kwargs["volumes"]))
        )

        return self.container

    def close(self):
        self.closed = True


class TestDockerCodeExecutor(unittest.IsolatedAsyncioTestCase):

    async def test_start_execute_stop(self):
        client = FakeClient()

        with patch(
            "app.agents.coding_agents.utils."
            "DockerCodeExecutor.docker.from_env",
            return_value=client,
        ):
            executor = DockerCodeExecutor()

            try:
                await executor.start()

                path = executor.work_dir

                # La directory temporanea deve essere scrivibile dal gruppo.
                self.assertEqual(
                    stat.S_IMODE(path.stat().st_mode),
                    0o770,
                )

                # Il container deve utilizzare il gruppo del processo Python.
                self.assertEqual(
                    client.options["user"],
                    f"0:{os.getgid()}",
                )

                result = await executor.execute_code_blocks([
                    CodeBlock("print('OK')", "python"),
                    CodeBlock("echo OK", "sh"),
                ])

                self.assertEqual(result.exit_code, 0)
                self.assertEqual(result.output, "OK\nOK\n")
                self.assertEqual(len(client.container.commands), 2)

                # Verifichiamo che Python venga lanciato
                # con il timeout corretto.
                self.assertEqual(
                    client.container.commands[0][:3],
                    ["timeout", "60", "python"],
                )

                # Verifichiamo il comando shell.
                self.assertEqual(
                    client.container.commands[1][:3],
                    ["timeout", "60", "sh"],
                )

                self.assertEqual(
                    client.options["working_dir"],
                    "/workspace",
                )

                self.assertTrue(path.exists())

            finally:
                await executor.stop()

            self.assertEqual(
                client.container.commands[-1][:2],
                ["python", "-c"],
            )

            self.assertTrue(client.container.stopped)
            self.assertTrue(client.closed)
            self.assertFalse(path.exists())

    async def test_error_stops_next_block(self):
        client = FakeClient()

        with patch(
            "app.agents.coding_agents.utils."
            "DockerCodeExecutor.docker.from_env",
            return_value=client,
        ):
            executor = DockerCodeExecutor()

            try:
                await executor.start()

                result = await executor.execute_code_blocks([
                    CodeBlock(
                        "raise RuntimeError('simulated')",
                        "python",
                    ),
                    CodeBlock(
                        "print('this must not run')",
                        "python",
                    ),
                ])

                self.assertEqual(result.exit_code, 1)
                self.assertIn("RuntimeError", result.output)
                self.assertEqual(len(client.container.commands), 1)

            finally:
                await executor.stop()

    async def test_requires_start(self):
        executor = DockerCodeExecutor()

        with self.assertRaises(RuntimeError):
            await executor.execute_code_blocks([
                CodeBlock("print(1)", "python")
            ])


if __name__ == "__main__":
    unittest.main()
