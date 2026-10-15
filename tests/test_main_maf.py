
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent_framework import Content, Message as LLMMessage

from app.main import build_workflow
from app.agents.coding_agents.messages.MessagesTypes import Message


# Tutti i ruoli presenti nel config.yaml originale.
ROLES = (
    "assistant",
    "entry_point",
    "coder",
    "test_designer",
    "test_executor",
    "debugger",
    "faas_deployer",
)


class ScriptedModel:
    """
    Simula un client LLM e registra i messaggi ricevuti.
    """

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def get_response(self, messages, options=None):
        self.calls.append(list(messages))

        if not self.responses:
            raise AssertionError(
                "Chiamata LLM aggiuntiva non prevista"
            )

        item = self.responses.pop(0)

        # Alcune risposte contengono già tool call MAF.
        if isinstance(item, SimpleNamespace):
            return item

        # Risposta testuale normale.
        return SimpleNamespace(
            text=item,
            messages=[
                LLMMessage(
                    role="assistant",
                    contents=[item],
                )
            ],
            usage_details=SimpleNamespace(
                input_token_count=5,
                output_token_count=3,
            ),
        )


class FakeDocker:
    """
    Simula l'esecuzione del codice senza avviare Docker.
    """

    def __init__(self, output=""):
        self.calls = []
        self.output = output

    async def execute_code_blocks(self, blocks):
        self.calls.append(list(blocks))

        return SimpleNamespace(
            output=self.output,
            exit_code=0,
        )


def deployment_models(assistant_answer):
    """
    Prepara i modelli simulati e una chiamata
    allo strumento di deployment.
    """

    models = {
        name: ScriptedModel()
        for name in ROLES
    }

    models["assistant"] = ScriptedModel(
        assistant_answer
    )

    arguments = {
        "code": (
            "def handler(params, context): "
            "return 42\n"
        ),
        "name": "smoke_main",
        "runtime": "python310",
        "memoryMB": 1024,
        "CPUDemand": 1,
    }

    tool_call = Content.from_function_call(
        call_id="call_main_1",
        name="create_json_serverledge",
        arguments=json.dumps(arguments),
    )

    models["faas_deployer"] = ScriptedModel(
        SimpleNamespace(
            text="",
            messages=[
                LLMMessage(
                    role="assistant",
                    contents=[tool_call],
                )
            ],
        ),
        "Deployment completed",
    )

    return models


class TestMainMAF(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Disabilita l'interfaccia UI nei test.
        patcher = patch.dict(
            os.environ,
            {"UI": "False"},
        )

        patcher.start()
        self.addCleanup(patcher.stop)

        self.llm = {
            name: "fake"
            for name in ROLES
        }

    async def test_deployment_diretto_attraverso_main(self):

        # L'Assistant riconosce codice già pronto.
        models = deployment_models(
            "deployment:def handler(params, context): return 42"
        )

        workflow = build_workflow(
            self.llm,
            models,
            FakeDocker(),
        )

        # Il deployment HTTP non viene realmente eseguito.
        with patch(
            "app.agents.FaaSDeployer.create_json_serverledge",
            new_callable=AsyncMock,
        ) as deployment:

            deployment.return_value = {
                "name": "smoke_main"
            }

            result = await workflow.run(
                Message(
                    "def handler(params, context): return 42",
                    "request",
                )
            )

            deployment.assert_awaited_once()

        self.assertEqual(
            result.get_outputs(),
            [
                Message(
                    "Deployment completed",
                    "deployment",
                )
            ],
        )

        self.assertEqual(
            len(models["assistant"].calls),
            1,
        )

        self.assertEqual(
            len(models["faas_deployer"].calls),
            2,
        )

    async def test_traduzione_generazione_test_e_deployment(self):

        # Questa volta percorriamo tutta la pipeline di coding.
        models = deployment_models(
            "translation: Add two integers"
        )

        models["entry_point"] = ScriptedModel(
            "def add(a, b):"
        )

        models["coder"] = ScriptedModel(
            "```python\n"
            "def add(a, b): return a + b\n"
            "```"
        )

        models["test_designer"] = ScriptedModel(
            "```python\n"
            "assert add(2, 3) == 5\n"
            "```"
        )

        models["test_executor"] = ScriptedModel(
            "```python\n"
            "def add(a, b): return a + b\n"
            "assert add(2, 3) == 5\n"
            "```"
        )

        docker = FakeDocker()

        workflow = build_workflow(
            self.llm,
            models,
            docker,
        )

        with patch(
            "app.agents.FaaSDeployer.create_json_serverledge",
            new_callable=AsyncMock,
        ) as deployment:

            deployment.return_value = {
                "name": "smoke_main"
            }

            result = await workflow.run(
                Message(
                    "Somma due numeri",
                    "request",
                )
            )

            deployment.assert_awaited_once()

        self.assertEqual(
            result.get_outputs(),
            [
                Message(
                    "Deployment completed",
                    "deployment",
                )
            ],
        )

        # Ogni agente del coding deve aver lavorato.
        self.assertEqual(len(models["entry_point"].calls), 1)
        self.assertEqual(len(models["coder"].calls), 1)
        self.assertEqual(len(models["test_designer"].calls), 1)
        self.assertEqual(len(models["test_executor"].calls), 1)

        # Il Test Executor deve aver richiesto l'esecuzione.
        self.assertEqual(len(docker.calls), 1)

        # Non abbiamo generato errori: il Debugger non interviene.
        self.assertEqual(len(models["debugger"].calls), 0)


if __name__ == "__main__":
    unittest.main()
