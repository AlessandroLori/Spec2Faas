
import hashlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler

from app.agents.Assistant import Assistant
from app.agents.coding_agents.messages.MessagesTypes import (
    DeployMessage,
    FaasMessage,
    Message,
)
from experiments.MessageTypesTest import (
    TestDeployMessage,
    TestDeployResult,
    TestMessage,
    TestSystemMessage,
)


class FakeModel:
    # Modello deterministico che registra i prompt ricevuti.
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    async def get_response(self, messages):
        self.calls.append(list(messages))

        return SimpleNamespace(
            text=self.answers.pop(0),
            usage_details=SimpleNamespace(
                input_token_count=8,
                output_token_count=3,
            ),
        )


class FakeContext:
    # Registra comunicazioni e risultati senza eseguire un workflow.
    def __init__(self):
        self.sent = []
        self.outputs = []

    async def send_message(self, message, target_id=None):
        self.sent.append((target_id, message))

    async def yield_output(self, message):
        self.outputs.append(message)


class EntryPointStub(Executor):
    # Sostituto deterministico dell'Entry Point.
    def __init__(self):
        super().__init__(id="entry_point")
        self.received = []

    @handler
    async def handle_request(
        self,
        message: Message,
        ctx: WorkflowContext[Message],
    ) -> None:

        self.received.append(message)

        await ctx.send_message(
            Message(
                "```python\ndef f(): return 42\n```",
                "test_executor_response",
            ),
            target_id="assistant",
        )


class DeployerStub(Executor):
    # Sostituto deterministico del FaaS Deployer.
    def __init__(self):
        super().__init__(id="faas_deployer")
        self.received = []

    @handler
    async def handle_deploy(
        self,
        message: DeployMessage,
        ctx: WorkflowContext[FaasMessage],
    ) -> None:

        self.received.append(message)

        await ctx.send_message(
            FaasMessage(
                "Deployment completed",
                {},
                "",
            ),
            target_id="assistant",
        )


class TestAssistantMAF(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        # Evitiamo di attivare accidentalmente il server UI nei test.
        self.ui = patch.dict(os.environ, {"UI": "False"})
        self.ui.start()
        self.addCleanup(self.ui.stop)

    def test_prompt_identici(self):
        agent = Assistant("fake", FakeModel())

        self.assertEqual(
            hashlib.sha256(
                agent.SYSTEM_PROMPT.encode()
            ).hexdigest(),
            "7c7ac211eb54461b4fbe7c3c0dea3ca7e293a97c720c490d9907b001d58fdcc7",
        )

        self.assertEqual(
            hashlib.sha256(
                agent.SYSTEM_PROMPT_TEST.encode()
            ).hexdigest(),
            "2bb57e4834e69d7bf13cd6695710d707969f705311a26b4479cc0207eaf25ef6",
        )

    async def test_traduzione_e_deployment(self):
        model = FakeModel("translation: Write a sum function")
        agent = Assistant("fake", model)
        ctx = FakeContext()

        # Richiesta iniziale.
        await agent.handle_user_message(
            Message("Somma", "request"),
            ctx,
        )

        self.assertEqual(
            ctx.sent[-1],
            (
                "entry_point",
                Message(" Write a sum function", "request"),
            ),
        )

        self.assertEqual(ctx.outputs, [])

        # Risposta dell'Entry Point.
        code = "```python\ndef f(): return 42\n```"

        await agent.handle_user_message(
            Message(code, "test_executor_response"),
            ctx,
        )

        self.assertEqual(
            ctx.sent[-1],
            (
                "faas_deployer",
                DeployMessage(code),
            ),
        )

        # Risposta del Deployer.
        await agent.handle_faas_message(
            FaasMessage("Deployed", {}, ""),
            ctx,
        )

        self.assertEqual(
            ctx.outputs,
            [Message("Deployed", "deployment")],
        )

        self.assertIsNone(agent._normal_phase)
        self.assertEqual(model.calls[0][0].role, "system")
        self.assertEqual(model.calls[0][1].text, "Somma")

    async def test_traduzione_generazione_fallita(self):
        agent = Assistant(
            "fake",
            FakeModel("translation: sum"),
        )

        ctx = FakeContext()

        await agent.handle_user_message(
            Message("Somma", "request"),
            ctx,
        )

        await agent.handle_user_message(
            Message("FAIL", "test_executor_response"),
            ctx,
        )

        self.assertEqual(
            ctx.outputs,
            [
                Message(
                    "We couldn't generate a correct function "
                    "given the specification.",
                    "failure",
                )
            ],
        )

        # Nessun deployment dopo la generazione fallita.
        self.assertEqual(len(ctx.sent), 1)
        self.assertIsNone(agent._normal_phase)

    async def test_deployment_diretto_e_errore(self):
        agent = Assistant(
            "fake",
            FakeModel("deployment:def f(): return 1"),
        )

        ctx = FakeContext()

        await agent.handle_user_message(
            Message("def f(): return 1", "request"),
            ctx,
        )

        self.assertEqual(
            ctx.sent,
            [
                (
                    "faas_deployer",
                    DeployMessage("def f(): return 1"),
                )
            ],
        )

        # Simuliamo un problema della piattaforma FaaS.
        await agent.handle_faas_message(
            FaasMessage(
                "FAILED",
                {},
                "connection error",
            ),
            ctx,
        )

        self.assertEqual(
            ctx.outputs,
            [Message("FAILED", "deployment")],
        )

    async def test_chiarimento(self):
        agent = Assistant(
            "fake",
            FakeModel("Please clarify the function."),
        )

        ctx = FakeContext()

        await agent.handle_user_message(
            Message("hello", "request"),
            ctx,
        )

        self.assertEqual(
            ctx.outputs,
            [
                Message(
                    "Please clarify the function.",
                    "request",
                )
            ],
        )

        self.assertEqual(ctx.sent, [])

    async def test_valutazione_isolata(self):
        model = FakeModel(
            "translation: sort items",
            "deployment:def f(): return 1",
            "clarify",
        )

        agent = Assistant("fake", model)
        ctx = FakeContext()

        for _ in range(3):
            await agent.handle_test_user_message(
                TestMessage("input"),
                ctx,
            )

        self.assertEqual(
            [x.type for x in ctx.outputs],
            ["translation", "deployment", "other"],
        )

        self.assertEqual(
            [x.tokens for x in ctx.outputs],
            [11, 11, 11],
        )

        self.assertTrue(
            all(x.time >= 0 for x in ctx.outputs)
        )

        # La valutazione isolata usa il prompt normale.
        self.assertEqual(
            model.calls[0][0].text,
            agent.SYSTEM_PROMPT,
        )

    async def test_sistema_generazione_fallita(self):
        agent = Assistant(
            "fake",
            FakeModel("english translated task"),
        )

        ctx = FakeContext()

        initial = TestSystemMessage(
            tokens={"previous": 1},
            time={},
            messages=4,
            prompt="Italiano",
        )

        await agent.handle_test_system_user_message(
            initial,
            ctx,
        )

        destination, sent = ctx.sent[-1]

        self.assertEqual(destination, "entry_point")
        self.assertEqual(sent.prompt, "english translated task")
        self.assertEqual(sent.messages, 5)
        self.assertEqual(sent.tokens["assistant"], 11)

        # In questo percorso deve usare il prompt sperimentale.
        self.assertEqual(
            agent._model_client.calls[0][0].text,
            agent.SYSTEM_PROMPT_TEST,
        )

        # Il coding team non riesce a generare la funzione.
        await agent.handle_test_system_user_message(
            TestSystemMessage(
                tokens=sent.tokens,
                time=sent.time,
                type="test_executor_response",
                generated=False,
            ),
            ctx,
        )

        self.assertEqual(len(ctx.outputs), 1)
        self.assertFalse(ctx.outputs[0].generated)
        self.assertIn("system", ctx.outputs[0].time)
        self.assertIsNone(agent._system_phase)

    async def test_sistema_deployment_e_metriche(self):
        agent = Assistant(
            "fake",
            FakeModel("english translated task"),
        )

        ctx = FakeContext()

        await agent.handle_test_system_user_message(
            TestSystemMessage(
                tokens={},
                time={},
                messages=1,
                prompt="Somma",
            ),
            ctx,
        )

        sent = ctx.sent[-1][1]

        completed = TestSystemMessage(
            tokens=sent.tokens,
            time=sent.time,
            messages=12,
            prompt="english translated task",
            signature="def f():",
            original_func="def f(): return 42",
            tests_str="assert f()==42",
            code_final_func="```python\ndef f(): return 42\n```",
            final_func="def f(): return 42",
            attempts=2,
            generated=True,
            type="test_executor_response",
        )

        # Il coding team ha terminato con successo.
        await agent.handle_test_system_user_message(
            completed,
            ctx,
        )

        self.assertEqual(
            ctx.sent[-1],
            (
                "faas_deployer",
                TestDeployMessage(completed.code_final_func),
            ),
        )

        # Risposta del Deployer con metriche.
        await agent.handle_test_deploy_result(
            TestDeployResult(
                "func_name",
                0.6,
                19,
                None,
                "function deployed",
                1,
            ),
            ctx,
        )

        result = ctx.outputs[-1]

        self.assertEqual(result.messages, 13)
        self.assertEqual(result.tokens["faas_deployer"], 19)
        self.assertEqual(result.time["faas_deployer"], 0.6)
        self.assertTrue(result.deployed)
        self.assertEqual(
            result.deployed_function,
            "function deployed",
        )
        self.assertEqual(
            result.final_func,
            "def f(): return 42",
        )
        self.assertEqual(result.attempts, 2)

    async def test_sistema_deployment_fallito(self):
        agent = Assistant(
            "fake",
            FakeModel("translated"),
        )

        ctx = FakeContext()

        await agent.handle_test_system_user_message(
            TestSystemMessage(
                tokens={},
                time={},
                prompt="x",
            ),
            ctx,
        )

        await agent.handle_test_system_user_message(
            TestSystemMessage(
                tokens={},
                time={},
                generated=True,
                code_final_func="code",
                type="test_executor_response",
            ),
            ctx,
        )

        await agent.handle_test_deploy_result(
            TestDeployResult(
                "FAIL",
                0.1,
                2,
                None,
            ),
            ctx,
        )

        self.assertFalse(ctx.outputs[-1].deployed)
        self.assertEqual(
            ctx.outputs[-1].result_deployment,
            "FAIL",
        )

    async def test_workflow_normale_maf(self):
        # Assistant reale, Entry Point e Deployer simulati.
        assistant = Assistant(
            "fake",
            FakeModel("translation: a function"),
        )

        entry_point = EntryPointStub()
        deployer = DeployerStub()

        # Vero grafo MAF con messaggi di ritorno.
        workflow = (
            WorkflowBuilder(
                start_executor=assistant,
                max_iterations=20,
            )
            .add_edge(assistant, entry_point)
            .add_edge(entry_point, assistant)
            .add_edge(assistant, deployer)
            .add_edge(deployer, assistant)
            .build()
        )

        result = await workflow.run(
            Message("a function", "request")
        )

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
            entry_point.received,
            [
                Message(
                    " a function",
                    "request",
                )
            ],
        )

        self.assertEqual(
            deployer.received,
            [
                DeployMessage(
                    "```python\ndef f(): return 42\n```"
                )
            ],
        )

        self.assertIsNone(assistant._normal_phase)


if __name__ == "__main__":
    unittest.main()
