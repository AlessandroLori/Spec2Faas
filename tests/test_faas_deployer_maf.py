
import hashlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent_framework import Content, Message as LLMMessage, WorkflowBuilder

from app.agents.FaaSDeployer import FaasDeployer
from app.agents.Assistant import Assistant
from app.agents.coding_agents.messages.MessagesTypes import (
    DeployMessage,
    FaasMessage,
    Message,
)
from experiments.MessageTypesTest import (
    TestDeployMessage,
    TestDeployResult,
)


ARGS = {
    "name": "sum_deployed",
    "code": (
        "def handler(params, context): "
        "return add(params['a'], params['b'])\n"
        "def add(a, b): return a + b\n"
    ),
    "runtime": "python310",
    "memoryMB": 1024,
    "CPUDemand": 1,
}


def call_response(arguments=None, *, call_id="call_1"):
    # Simula un messaggio MAF con una vera chiamata
    # alla funzione create_json_serverledge.
    arguments = ARGS if arguments is None else arguments

    content = Content.from_function_call(
        call_id=call_id,
        name="create_json_serverledge",
        arguments=json.dumps(arguments),
    )

    return SimpleNamespace(
        text="",
        messages=[
            LLMMessage(
                role="assistant",
                contents=[content],
            )
        ],
        usage_details=SimpleNamespace(
            input_token_count=7,
            output_token_count=3,
        ),
    )


def text_response(text="Deployment completed"):
    # Simula una risposta finale senza tool call.
    return SimpleNamespace(
        text=text,
        messages=[
            LLMMessage(
                role="assistant",
                contents=[text],
            )
        ],
        usage_details=SimpleNamespace(
            input_token_count=7,
            output_token_count=3,
        ),
    )


class FakeModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def get_response(self, messages, options=None):
        self.calls.append(
            (list(messages), options)
        )

        return self.responses.pop(0)


class FakeContext:
    def __init__(self):
        self.sent = []

    async def send_message(self, message, target_id=None):
        self.sent.append(
            (target_id, message)
        )


class TestFaasDeployerMAF(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.ui = patch.dict(
            os.environ,
            {"UI": "False"},
        )
        self.ui.start()
        self.addCleanup(self.ui.stop)

    def test_prompt_immutato(self):
        agent = FaasDeployer(
            "fake",
            FakeModel(),
        )

        self.assertEqual(
            hashlib.sha256(
                agent.SYSTEM_PROMPT.encode()
            ).hexdigest(),
            "1f56e632a7989d8efcc0ef51961d3487499cf9da66d76f6f05a8f6b79181e8ea",
        )

        self.assertEqual(
            agent._tools[0].name,
            "create_json_serverledge",
        )

    async def test_normale_deployment(self):
        model = FakeModel(
            call_response(),
            text_response(),
        )

        agent = FaasDeployer("fake", model)
        ctx = FakeContext()

        # Simuliamo Serverledge senza effettuare richieste HTTP.
        with patch(
            "app.agents.FaaSDeployer.create_json_serverledge",
            new_callable=AsyncMock,
        ) as deployment:
            deployment.return_value = {
                "name": "sum_deployed"
            }

            await agent.handle_deploy_message(
                DeployMessage(
                    "def add(a,b):return a+b"
                ),
                ctx,
            )

            deployment.assert_awaited_once_with(**ARGS)

        target, result = ctx.sent[0]

        self.assertEqual(target, "assistant")
        self.assertIsInstance(result, FaasMessage)
        self.assertEqual(result.content, "Deployment completed")
        self.assertEqual(result.arguments, ARGS)
        self.assertEqual(result.error_message, "")

        # Prima chiamata: system prompt e codice.
        # Seconda chiamata: presente anche il risultato del tool.
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(
            model.calls[0][0][1].text,
            "This is the code: def add(a,b):return a+b",
        )

        self.assertEqual(
            model.calls[1][0][-1].role,
            "tool",
        )

        self.assertEqual(
            model.calls[1][0][-1].contents[0].type,
            "function_result",
        )

    async def test_normale_errore_http(self):
        agent = FaasDeployer(
            "fake",
            FakeModel(
                call_response(),
                text_response(),
            ),
        )

        ctx = FakeContext()

        with patch(
            "app.agents.FaaSDeployer.create_json_serverledge",
            new_callable=AsyncMock,
        ) as deployment:
            deployment.return_value = {
                "status": 409,
                "text": "Function already exists",
            }

            await agent.handle_deploy_message(
                DeployMessage("code"),
                ctx,
            )

        self.assertIn(
            "409",
            ctx.sent[0][1].error_message,
        )

    async def test_normale_senza_tool(self):
        agent = FaasDeployer(
            "fake",
            FakeModel(
                text_response("Done")
            ),
        )

        ctx = FakeContext()

        await agent.handle_deploy_message(
            DeployMessage("code"),
            ctx,
        )

        self.assertEqual(ctx.sent[0][0], "assistant")
        self.assertNotEqual(
            ctx.sent[0][1].error_message,
            "",
        )

    async def test_sperimentale_due_tentativi(self):
        changed = {
            **ARGS,
            "name": "sum_deployed_v2",
        }

        model = FakeModel(
            call_response(),
            call_response(
                changed,
                call_id="call_2",
            ),
            text_response(),
        )

        agent = FaasDeployer("fake", model)
        ctx = FakeContext()

        with patch(
            "app.agents.FaaSDeployer.create_json_serverledge",
            new_callable=AsyncMock,
        ) as deployment:
            deployment.side_effect = [
                {
                    "status": 409,
                    "text": "Function already exists",
                },
                {
                    "name": "sum_deployed_v2"
                },
            ]

            await agent.handle_test_deploy_message(
                TestDeployMessage("code"),
                ctx,
            )

            self.assertEqual(
                deployment.await_count,
                2,
            )

        target, result = ctx.sent[0]

        self.assertEqual(target, "assistant")
        self.assertIsInstance(result, TestDeployResult)
        self.assertEqual(result.result, "sum_deployed_v2")

        # 3 chiamate LLM: 10 token ciascuna.
        self.assertEqual(result.tokens, 30)

        # 2 batch di chiamate a Serverledge.
        self.assertEqual(
            result.invocation_attempts,
            2,
        )

        self.assertEqual(
            result.deployed_function,
            ARGS["code"],
        )

        self.assertIsNone(result.ctx)

        # La terza richiesta LLM contiene tutta la cronologia.
        self.assertEqual(
            len(model.calls[-1][0]),
            6,
        )

    async def test_sperimentale_errore_tool(self):
        agent = FaasDeployer(
            "fake",
            FakeModel(
                call_response(),
                text_response(),
            ),
        )

        ctx = FakeContext()

        with patch(
            "app.agents.FaaSDeployer.create_json_serverledge",
            new_callable=AsyncMock,
        ) as deployment:
            deployment.side_effect = RuntimeError(
                "server offline"
            )

            await agent.handle_test_deploy_message(
                TestDeployMessage("code"),
                ctx,
            )

        self.assertEqual(
            ctx.sent[0][1].result,
            "FAIL",
        )

        self.assertEqual(
            ctx.sent[0][1].invocation_attempts,
            1,
        )

    async def test_qwen_risposta_json(self):
        # Qwen 2.5 restituisce la chiamata come testo JSON.
        raw = json.dumps({
            "name": "create_json_serverledge",
            "arguments": ARGS,
        })

        agent = FaasDeployer(
            "qwen2.5-coder",
            FakeModel(
                text_response(raw),
                text_response("OK"),
            ),
        )

        ctx = FakeContext()

        with patch(
            "app.agents.FaaSDeployer.create_json_serverledge",
            new_callable=AsyncMock,
        ) as deployment:
            deployment.return_value = {
                "name": "sum_deployed"
            }

            await agent.handle_test_deploy_message(
                TestDeployMessage("code"),
                ctx,
            )

            deployment.assert_awaited_once_with(**ARGS)

        self.assertEqual(
            ctx.sent[0][1].result,
            "sum_deployed",
        )

    async def test_qwen_json_malformato(self):
        text = (
            '{"name": "create_json_serverledge", '
            '"arguments": NOT_JSON}'
        )

        agent = FaasDeployer(
            "qwen2.5-coder",
            FakeModel(
                text_response(text)
            ),
        )

        ctx = FakeContext()

        await agent.handle_test_deploy_message(
            TestDeployMessage("code"),
            ctx,
        )

        self.assertEqual(
            ctx.sent[0][1].result,
            "FAIL",
        )

        self.assertEqual(
            ctx.sent[0][1].invocation_attempts,
            0,
        )

    async def test_workflow_assistant_e_deployer(self):
        # Qui utilizziamo un vero WorkflowBuilder.
        #
        # Assistant e FaasDeployer sono entrambi
        # agenti migrati a MAF.
        assistant = Assistant(
            "fake-assistant",
            FakeModel(
                text_response(
                    "deployment:def f():return 1"
                )
            ),
        )

        deployer = FaasDeployer(
            "fake-deployer",
            FakeModel(
                call_response(),
                text_response("Serverledge response"),
            ),
        )

        workflow = (
            WorkflowBuilder(
                start_executor=assistant,
                max_iterations=20,
            )
            .add_edge(assistant, deployer)
            .add_edge(deployer, assistant)
            .build()
        )

        # Il vero tool HTTP è simulato.
        with patch(
            "app.agents.FaaSDeployer.create_json_serverledge",
            new_callable=AsyncMock,
        ) as deployment:
            deployment.return_value = {
                "name": "sum_deployed"
            }

            result = await workflow.run(
                Message(
                    "def f():return 1",
                    "request",
                )
            )

        self.assertEqual(
            result.get_outputs(),
            [
                Message(
                    "Serverledge response",
                    "deployment",
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
