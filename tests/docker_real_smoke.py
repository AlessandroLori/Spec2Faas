
import unittest
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

from app.agents.coding_agents.TestExecutor import TestExecutor
from app.agents.coding_agents.messages.MessagesTypes import (
    DebugMessage,
    ExecuteMessage,
    Message,
)
from app.agents.coding_agents.utils.Code_Extractors import CodeBlock
from app.agents.coding_agents.utils.DockerCodeExecutor import DockerCodeExecutor


# Questi test usano Docker, senza API LLM esterne.
# Il file non entra nella suite veloce ordinaria.
CODE = "```python\ndef add(a, b):\n    return a + b\n```"
TESTS = "```python\nassert add(2, 3) == 5\n```"


class FixedResponseModel:
    """Simula la risposta LLM, lasciando reale l'esecuzione Docker."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    async def get_response(self, messages):
        self.calls.append(list(messages))
        return SimpleNamespace(text=self.response)


class RecordingContext:
    """Registra i messaggi emessi dal Test Executor."""

    def __init__(self):
        self.sent = []

    async def send_message(self, message, target_id=None):
        self.sent.append((target_id, message))


class DockerRealSmoke(unittest.IsolatedAsyncioTestCase):
    """Test reali dell'adattatore e del Test Executor."""

    async def asyncSetUp(self):
        # Ogni test usa un proprio container per evitare interferenze.
        self.executor = DockerCodeExecutor()
        await self.executor.start()

    async def asyncTearDown(self):
        # Il container viene fermato anche quando il test fallisce.
        await self.executor.stop()

    async def test_installazione_pip_offline(self):
        # Costruiamo un piccolo pacchetto wheel installabile da pip.
        filename = "spec2faas_smoke-0.1.0-py3-none-any.whl"
        dist_info = "spec2faas_smoke-0.1.0.dist-info"

        files = {
            "spec2faas_smoke/__init__.py": (
                "def answer():\n    return 42\n"
            ),
            f"{dist_info}/METADATA": (
                "Metadata-Version: 2.1\n"
                "Name: spec2faas-smoke\n"
                "Version: 0.1.0\n"
            ),
            f"{dist_info}/WHEEL": (
                "Wheel-Version: 1.0\n"
                "Generator: spec2faas-smoke-test\n"
                "Root-Is-Purelib: true\n"
                "Tag: py3-none-any\n"
            ),
        }

        # Il file RECORD elenca i contenuti del pacchetto wheel.
        files[f"{dist_info}/RECORD"] = "".join(
            f"{path},,\n"
            for path in [*files, f"{dist_info}/RECORD"]
        )

        wheel_path = self.executor.work_dir / filename

        with ZipFile(wheel_path, "w", ZIP_DEFLATED) as wheel:
            for path, content in files.items():
                wheel.writestr(path, content)

        wheel_path.chmod(0o644)

        # Primo blocco: installazione pip nel container.
        # Secondo blocco: import del pacchetto appena installato.
        result = await self.executor.execute_code_blocks([
            CodeBlock(
                "pip install --no-index --no-deps "
                f"--target /workspace/vendor /workspace/{filename}\n",
                "sh",
            ),
            CodeBlock(
                "import sys\n"
                "sys.path.insert(0, '/workspace/vendor')\n"
                "from spec2faas_smoke import answer\n"
                "assert answer() == 42\n"
                "print('PIP_INSTALL_OK')\n",
                "python",
            ),
        ])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("PIP_INSTALL_OK", result.output)

    async def test_timeout(self):
        self.executor.timeout = 1

        result = await self.executor.execute_code_blocks([
            CodeBlock(
                "import time\ntime.sleep(4)\n",
                "python",
            )
        ])

        self.assertEqual(result.exit_code, 124, result.output)
        self.assertIn("Timeout", result.output)

    async def test_exit_code_senza_output(self):
        # Un programma può fallire senza stampare "Error"
        # e senza produrre alcun output.
        result = await self.executor.execute_code_blocks([
            CodeBlock(
                "import sys\nsys.exit(7)\n",
                "python",
            ),
            CodeBlock(
                "print('NON_DEVE_COMPARIRE')\n",
                "python",
            ),
        ])

        self.assertEqual(result.exit_code, 7)
        self.assertEqual(result.output, "")

    async def test_test_executor_successo_reale(self):
        # Il modello LLM è simulato: restituisce codice e test
        # già uniti in un unico blocco Python eseguibile.
        model = FixedResponseModel(
            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "assert add(2, 3) == 5\n"
            "```"
        )

        # Il Test Executor è quello realmente migrato a MAF.
        # L'esecutore Docker è reale.
        agent = TestExecutor("fake-llm", model, self.executor)
        ctx = RecordingContext()

        # Prima arriva soltanto il codice dal Coder.
        await agent.handle_execute_code_message(
            ExecuteMessage(
                "Somma due numeri",
                CODE,
                "",
                "coder",
            ),
            ctx,
        )

        self.assertEqual(
            ctx.sent[-1],
            (
                "entry_point",
                Message("Still waiting", "test_executor_response"),
            ),
        )

        # Successivamente arrivano i test dal Test Designer.
        await agent.handle_execute_code_message(
            ExecuteMessage(
                "Somma due numeri",
                "",
                TESTS,
                "test_designer",
            ),
            ctx,
        )

        # Il Test Executor deve avere eseguito il programma
        # e restituito il codice all'Entry Point.
        self.assertEqual(
            ctx.sent[-1],
            (
                "entry_point",
                Message(CODE, "test_executor_response"),
            ),
        )

        self.assertEqual(len(model.calls), 1)

    async def test_test_executor_errore_reale(self):
        # Questa volta il codice contiene un'asserzione falsa.
        model = FixedResponseModel(
            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "assert add(2, 3) == 99\n"
            "```"
        )

        agent = TestExecutor("fake-llm", model, self.executor)
        ctx = RecordingContext()

        await agent.handle_execute_code_message(
            ExecuteMessage(
                "Somma due numeri",
                CODE,
                "",
                "coder",
            ),
            ctx,
        )

        await agent.handle_execute_code_message(
            ExecuteMessage(
                "Somma due numeri",
                "",
                TESTS,
                "test_designer",
            ),
            ctx,
        )

        # Docker deve restituire un traceback con AssertionError.
        # Il Test Executor deve inoltrarlo al Debugger.
        target, debug_request = ctx.sent[-1]

        self.assertEqual(target, "debugger")
        self.assertIsInstance(debug_request, DebugMessage)
        self.assertIn(
            "AssertionError",
            debug_request.error_message,
        )
        self.assertEqual(debug_request.code, CODE)


if __name__ == "__main__":
    unittest.main()
