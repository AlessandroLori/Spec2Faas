import base64
import importlib
import io
import json
import os
import tarfile
import unittest
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from agent_framework import Content, Message as LLMMessage

from app.agents.FaaSDeployer import create_json_serverledge
from app.agents.coding_agents.TestExecutor import TestExecutor
from app.agents.coding_agents.messages.MessagesTypes import ExecuteMessage, Message
from app.server import Server
from experiments.MessageTypesTest import TestSystemMessage
from experiments.maf_evaluator_common import (
    ROLES,
    result_from_workflow,
    system_workflow,
)
from experiments.system.System_evaluator import initial_metrics

"""Verifiche finali pre-Serverledge per Spec2FaaS/Microsoft Agent Framework.

Non richiede Ollama, GPU, Docker, ChromaDB reale, Gemini o Serverledge.
Non modifica i risultati Parquet e non apre porte di rete. È ancora simulativo
"""


CODE = "```python\ndef add(a, b):\n    return a + b\n```"
TESTS = "```python\nassert add(2, 3) == 5\n```"
MERGED = "```python\ndef add(a, b):\n    return a + b\nassert add(2, 3) == 5\n```"
TOOL_ARGS = {
    "name": "pre_server_fake",
    "code": (
        "def handler(params, context):\n"
        "    return add(params['a'], params['b'])\n"
        "def add(a, b): return a + b\n"
    ),
    "runtime": "python310",
    "memoryMB": 1024,
    "CPUDemand": 1,
}


class FakeModel:
    """LLM deterministico: risposte note e usage_details nel formato Ollama."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def get_response(self, messages, options=None):
        self.calls.append((list(messages), options))
        if not self.responses:
            raise AssertionError("Chiamata LLM non prevista: controllare il routing")

        item = self.responses.pop(0)
        if not isinstance(item, str):
            return item

        return SimpleNamespace(
            text=item,
            messages=[LLMMessage(role="assistant", contents=[item])],
            usage_details={"input_token_count": 12, "output_token_count": 8},
        )


class FakeDocker:
    """Non crea container: conserva i blocchi che TestExecutor voleva eseguire."""

    def __init__(self, exit_code=0, output=""):
        self.exit_code = exit_code
        self.output = output
        self.calls = []

    async def execute_code_blocks(self, blocks):
        self.calls.append(list(blocks))
        return SimpleNamespace(exit_code=self.exit_code, output=self.output)


class FakeContext:
    def __init__(self):
        self.sent = []
        self.outputs = []

    async def send_message(self, message, target_id=None):
        self.sent.append((target_id, message))

    async def yield_output(self, value):
        self.outputs.append(value)


def models_for_system(*, real_deployer=False):
    models = {role: FakeModel() for role in ROLES}
    models["assistant"] = FakeModel("Write a function adding two numbers")
    models["entry_point"] = FakeModel("def add(a, b):")
    models["coder"] = FakeModel(CODE)
    models["test_designer"] = FakeModel(TESTS)
    models["test_executor"] = FakeModel(MERGED)

    if real_deployer:
        tool_call = Content.from_function_call(
            call_id="pre_server_call_1",
            name="create_json_serverledge",
            arguments=json.dumps(TOOL_ARGS),
        )
        models["faas_deployer"] = FakeModel(
            SimpleNamespace(
                text="",
                messages=[LLMMessage(role="assistant", contents=[tool_call])],
                usage_details={"input_token_count": 12, "output_token_count": 8},
            ),
            "Deployment completed",
        )

    return models


class ATestExecutorReadiness(unittest.IsolatedAsyncioTestCase):
    """A: codice/test inoltrati a Docker ed exit code interpretato correttamente."""

    def setUp(self):
        ui = patch.dict(os.environ, {"UI": "False", "MAX_DEBUG_ATTEMPTS": "2"})
        ui.start()
        self.addCleanup(ui.stop)

    async def test_normal_wait_merges_exactly_and_succeeds(self):
        docker = FakeDocker()
        model = FakeModel(MERGED)
        agent = TestExecutor("fake", model, docker)
        ctx = FakeContext()

        await agent.handle_execute_code_message(
            ExecuteMessage("add integers", CODE, "", "coder"), ctx
        )
        self.assertEqual(ctx.sent[-1], ("entry_point", Message("Still waiting", "test_executor_response")))
        self.assertEqual(docker.calls, [])

        await agent.handle_execute_code_message(
            ExecuteMessage("add integers", "", TESTS, "test_designer"), ctx
        )
        self.assertEqual(len(docker.calls), 1)
        self.assertEqual(docker.calls[0][0].code, MERGED.removeprefix("```python\n").removesuffix("```"))
        self.assertEqual(ctx.sent[-1], ("entry_point", Message(CODE, "test_executor_response")))
        self.assertIn(CODE, model.calls[0][0][-1].text)
        self.assertIn(TESTS, model.calls[0][0][-1].text)

    async def test_timeout_not_accepted_as_success(self):
        docker = FakeDocker(exit_code=124, output="\n Timeout")
        agent = TestExecutor("fake", FakeModel(MERGED), docker)
        ctx = FakeContext()
        await agent.handle_execute_code_message(
            ExecuteMessage("add integers", CODE, "", "coder"), ctx
        )
        await agent.handle_execute_code_message(
            ExecuteMessage("add integers", "", TESTS, "test_designer"), ctx
        )
        self.assertEqual(ctx.sent[-1][0], "debugger")
        self.assertIn("Timeout", ctx.sent[-1][1].error_message)

    async def test_system_success_preserves_metrics_and_code(self):
        docker = FakeDocker()
        agent = TestExecutor("fake", FakeModel(MERGED), docker)
        ctx = FakeContext()
        tokens, times = initial_metrics()

        await agent.handle_test_system_execute_code_message(
            TestSystemMessage(
                tokens=tokens, time=times, messages=4,
                prompt="add integers", signature="def add(a, b):",
                original_func="def add(a, b): return a + b",
                code=CODE, new_chat=True, sender="coder",
            ), ctx
        )
        self.assertEqual(ctx.sent[-1][0], "entry_point")
        self.assertEqual(docker.calls, [])

        await agent.handle_test_system_execute_code_message(
            TestSystemMessage(
                tokens=tokens, time=times, messages=5,
                prompt="add integers", signature="def add(a, b):",
                original_func="def add(a, b): return a + b",
                code=CODE, tests=TESTS, tests_str="assert add(2, 3) == 5",
                sender="test_designer",
            ), ctx
        )
        target, result = ctx.sent[-1]
        self.assertEqual(target, "entry_point")
        self.assertTrue(result.generated)
        self.assertEqual(result.attempts, 0)
        self.assertEqual(result.final_func.strip(), "def add(a, b):\n    return a + b")
        self.assertEqual(result.tokens["test_executor"], 20)
        self.assertGreaterEqual(result.time["test_executor"], 0)
        self.assertEqual(len(docker.calls), 1)


class BEndToEndDeploymentReadiness(unittest.IsolatedAsyncioTestCase):
    """B: percorre davvero i sette executor con risultati LLM fissati."""

    def setUp(self):
        ui = patch.dict(os.environ, {"UI": "False", "MAX_DEBUG_ATTEMPTS": "2"})
        ui.start()
        self.addCleanup(ui.stop)
        self.llm = {role: "fake" for role in ROLES}

    async def test_offline_success_never_contacts_serverledge(self):
        models = models_for_system()
        docker = FakeDocker()
        flow = system_workflow(self.llm, models, docker, allow_deployment=False)
        tokens, times = initial_metrics()
        with patch(
            "app.agents.FaaSDeployer.create_json_serverledge",
            new_callable=AsyncMock,
        ) as http_call:
            result = result_from_workflow(await flow.run(
                TestSystemMessage(tokens=tokens, time=times, prompt="add integers")
            ))
            http_call.assert_not_awaited()

        self.assertTrue(result.generated)
        self.assertFalse(result.deployed)
        self.assertEqual(result.result_deployment, "FAIL")
        self.assertEqual(result.tokens["faas_deployer"], 0)
        self.assertEqual(len(models["faas_deployer"].calls), 0)
        self.assertEqual(len(docker.calls), 1)
        self.assertGreater(result.messages, 5)

    async def test_full_experimental_success_with_mocked_http_tool(self):
        models = models_for_system(real_deployer=True)
        docker = FakeDocker()
        flow = system_workflow(self.llm, models, docker, allow_deployment=True)
        tokens, times = initial_metrics()
        with patch(
            "app.agents.FaaSDeployer.create_json_serverledge",
            new_callable=AsyncMock,
            return_value={"name": "pre_server_fake"},
        ) as http_call:
            result = result_from_workflow(await flow.run(
                TestSystemMessage(tokens=tokens, time=times, prompt="add integers")
            ))
            http_call.assert_awaited_once_with(**TOOL_ARGS)

        self.assertTrue(result.generated)
        self.assertTrue(result.deployed)
        self.assertEqual(result.result_deployment, "pre_server_fake")
        self.assertIn("def handler", result.deployed_function)
        self.assertEqual(result.attempts, 0)
        self.assertEqual(result.tokens["assistant"], 20)
        self.assertEqual(result.tokens["test_executor"], 20)
        self.assertEqual(result.tokens["faas_deployer"], 40)
        self.assertEqual(result.tokens["debugger"], 0)
        self.assertGreater(result.time["system"], 0)
        self.assertGreater(result.messages, 6)
        self.assertEqual(len(models["faas_deployer"].calls), 2)
        self.assertEqual(len(models["debugger"].calls), 0)
        self.assertEqual(len(docker.calls), 1)

    async def test_http_error_not_marked_as_deployed(self):
        models = models_for_system(real_deployer=True)
        flow = system_workflow(self.llm, models, FakeDocker(), allow_deployment=True)
        tokens, times = initial_metrics()
        with patch(
            "app.agents.FaaSDeployer.create_json_serverledge",
            new_callable=AsyncMock,
            return_value={"status": 409, "text": "Function already exists"},
        ) as http_call:
            result = result_from_workflow(await flow.run(
                TestSystemMessage(tokens=tokens, time=times, prompt="add integers")
            ))
            http_call.assert_awaited_once()

        self.assertTrue(result.generated)
        self.assertFalse(result.deployed)
        self.assertEqual(result.result_deployment, "FAIL")
        self.assertEqual(result.tokens["faas_deployer"], 40)


class BHttpPayloadReadiness(unittest.IsolatedAsyncioTestCase):
    """B: payload Serverledge costruito correttamente senza alcuna richiesta HTTP."""

    async def test_serverledge_payload_tar_and_handler_without_network(self):
        response = MagicMock()
        response.status = 200
        response.ok = True
        response.text = AsyncMock(return_value='{"name":"pre_server_fake"}')
        response.json = AsyncMock(return_value={"name": "pre_server_fake"})

        post_context = MagicMock()
        post_context.__aenter__ = AsyncMock(return_value=response)
        post_context.__aexit__ = AsyncMock(return_value=None)

        session = MagicMock()
        session.post.return_value = post_context
        session_context = MagicMock()
        session_context.__aenter__ = AsyncMock(return_value=session)
        session_context.__aexit__ = AsyncMock(return_value=None)

        with (
            patch.dict(os.environ, {"SERVERLEDGE_URL": "http://127.0.0.1:9999"}),
            patch("app.agents.FaaSDeployer.aiohttp.ClientSession", return_value=session_context),
        ):
            result = await create_json_serverledge(**TOOL_ARGS)

        self.assertEqual(result, {"name": "pre_server_fake"})
        session.post.assert_called_once()
        url = session.post.call_args.args[0]
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(url, "http://127.0.0.1:9999/create")
        self.assertEqual(payload["Name"], TOOL_ARGS["name"])
        self.assertEqual(payload["Runtime"], TOOL_ARGS["runtime"])
        self.assertEqual(payload["MemoryMB"], TOOL_ARGS["memoryMB"])
        self.assertEqual(payload["CPUDemand"], TOOL_ARGS["CPUDemand"])
        self.assertEqual(payload["Handler"], "pre_server_fake.handler")

        data = base64.b64decode(payload["TarFunctionCode"])
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            file = archive.extractfile("pre_server_fake.py")
            self.assertIsNotNone(file)
            self.assertEqual(file.read().decode("utf-8"), TOOL_ARGS["code"])



class CApplicationReadiness(unittest.IsolatedAsyncioTestCase):
    """C: API Flask e invocazione dei main senza rete né Docker reali."""

    async def test_standard_main_ui_dispatch_and_cleanup(self):
        main_mod = importlib.import_module("app.main")
        models = {role: object() for role in ROLES}
        executor = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
        workflow = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(
            get_outputs=lambda: [Message("finished", "deployment")]
        )))
        with (
            patch.dict(os.environ, {"UI": "True"}),
            patch.object(main_mod, "create_model_clients", return_value=models),
            patch.object(main_mod, "DockerCodeExecutor", return_value=executor),
            patch.object(main_mod, "build_workflow", return_value=workflow),
            patch.object(main_mod, "_close_model_clients", new_callable=AsyncMock) as close,
        ):
            await main_mod.main({role: "fake" for role in ROLES}, None, "hello")

        workflow.run.assert_awaited_once_with(Message("hello", "request"))
        executor.start.assert_awaited_once()
        executor.stop.assert_awaited_once()
        close.assert_awaited_once_with(models)

    async def test_rag_main_ui_dispatch_memory_and_cleanup(self):
        rag_mod = importlib.import_module("app.main_RAG")
        models = {role: object() for role in ROLES}
        executor = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
        memory = SimpleNamespace(close=AsyncMock())
        workflow = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(
            get_outputs=lambda: [Message("finished", "deployment")]
        )))
        with (
            patch.dict(os.environ, {"UI": "True"}),
            patch.object(rag_mod, "create_model_clients", return_value=models),
            patch.object(rag_mod, "DockerCodeExecutor", return_value=executor),
            patch.object(rag_mod, "ChromaRAGMemory", return_value=memory),
            patch.object(rag_mod, "build_workflow_rag", return_value=workflow),
            patch.object(rag_mod, "_close_model_clients", new_callable=AsyncMock) as close,
        ):
            await rag_mod.main({role: "fake" for role in ROLES}, None, "hello rag")

        workflow.run.assert_awaited_once_with(Message("hello rag", "request"))
        executor.start.assert_awaited_once()
        executor.stop.assert_awaited_once()
        memory.close.assert_awaited_once()
        close.assert_awaited_once_with(models)

    async def test_background_thread_calls_rag_main(self):
        server = Server()
        fake_llm = {role: "fake" for role in ROLES}

        with (
            patch("app.main_RAG.main", new_callable=AsyncMock) as run_rag,
            patch("app.server.logging.shutdown"),
        ):
            # Eseguiamo il metodo in un thread separato,
            # proprio come avviene nell'applicazione Flask.
            await asyncio.to_thread(
                server.run_main_in_background,
                fake_llm,
                "check function",
            )

            # Verifichiamo che il main RAG sia stato
            # effettivamente eseguito con i parametri attesi.
            run_rag.assert_awaited_once_with(
                fake_llm,
                server,
                "check function",
            )

class CFlaskReadiness(unittest.TestCase):
    def setUp(self):
        self.server = Server()
        self.client = self.server.app.test_client()

    def test_flask_root_and_rejected_bad_request(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        response = self.client.post("/start", json={"other": "missing text"})
        self.assertEqual(response.status_code, 400)

    def test_flask_post_starts_only_mocked_thread(self):
        llm = {role: "fake" for role in ROLES}
        with (
            patch("app.server.load_env_variables"),
            patch("app.server.get_config_data", return_value=llm),
            patch("app.server.threading.Thread") as thread_class,
            patch("app.server.set_logging_config"),
        ):
            response = self.client.post("/start", json={"text": "write f"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["received"], "write f")
            thread_class.assert_called_once()
            kwargs = thread_class.call_args.kwargs
            self.assertEqual(kwargs["args"], (llm, "write f"))
            self.assertEqual(kwargs["target"], self.server.run_main_in_background)
            thread_class.return_value.start.assert_called_once()

    def test_websocket_chunk_structure_without_socket(self):
        class FakeSocket:
            def __init__(self):
                self.sent = []

            def send(self, data):
                self.sent.append(json.loads(data))

        ws = FakeSocket()
        self.server.connected_clients.add(ws)
        self.server.send_chunk("sample", "coder", "text")
        self.assertEqual(ws.sent, [{
            "type": "text", "text": "sample", "sender": "coder",
        }])


if __name__ == "__main__":
    unittest.main()
