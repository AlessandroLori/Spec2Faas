
import hashlib
import os
import unittest
from types import SimpleNamespace
from typing import Never
from unittest.mock import patch

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler

from app.agents.coding_agents.TestDesigner import TestDesigner
from app.agents.coding_agents.messages.MessagesTypes import CodeMessage, ExecuteMessage
from experiments.MessageTypesTest import TestCodeMessage, TestCodeResult, TestSystemMessage


ORIGINAL_SYSTEM_PROMPT_SHA256 = "20945a0ef5b13943e0967713aa1b2146a6ecaac13717746948e96b2b8f8bb0e6"

ASSERTION = "assert add(2, 3) == 5, 'Test Case 1: basic addition'"
GENERATED_TESTS = f"```python\n{ASSERTION}\n```"

USER_PROMPT = (
    "Function specification: Add two numbers"
    "\nFunction signature: def add(a, b):"
    "\nPlease generate the tests:"
)


class FakeModelClient:
    def __init__(self):
        self.response_text = GENERATED_TESTS
        self.calls = []

    async def get_response(self, messages):
        self.calls.append(list(messages))

        return SimpleNamespace(
            text=self.response_text,
            usage_details=SimpleNamespace(
                input_token_count=12,
                output_token_count=8,
            ),
        )


class FakeContext:
    def __init__(self):
        self.sent = []
        self.outputs = []

    async def send_message(self, message):
        self.sent.append(message)

    async def yield_output(self, message):
        self.outputs.append(message)


class ResultCollector(Executor):
    @handler
    async def collect(
        self,
        message: ExecuteMessage,
        ctx: WorkflowContext[Never, ExecuteMessage],
    ) -> None:
        await ctx.yield_output(message)


class TestTestDesignerMAF(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.model = FakeModelClient()
        self.designer = TestDesigner(
            llm="fake-model",
            model_client=self.model,
        )
        self.ctx = FakeContext()

    def assert_request_correct(self):
        self.assertEqual(len(self.model.calls), 1)

        messages = self.model.calls[0]
        self.assertEqual(len(messages), 2)

        self.assertEqual(
            [getattr(m.role, "value", m.role) for m in messages],
            ["system", "user"],
        )

        self.assertEqual(
            messages[0].text,
            TestDesigner.SYSTEM_PROMPT,
        )
        self.assertEqual(messages[1].text, USER_PROMPT)

    def test_system_prompt_immutato(self):
        self.assertEqual(
            hashlib.sha256(
                TestDesigner.SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            ORIGINAL_SYSTEM_PROMPT_SHA256,
        )

    async def test_generazione_normale(self):
        with patch.dict(os.environ, {"UI": "False"}):
            await self.designer.handle_generate_code_message(
                CodeMessage(
                    "Add two numbers",
                    "def add(a, b):",
                    "not used",
                ),
                self.ctx,
            )

        self.assert_request_correct()
        self.assertEqual(len(self.ctx.sent), 1)
        self.assertEqual(len(self.ctx.outputs), 0)

        result = self.ctx.sent[0]

        self.assertIsInstance(result, ExecuteMessage)

        self.assertEqual(
            (
                result.specification,
                result.code,
                result.tests,
                result.sender,
            ),
            (
                "Add two numbers",
                "",
                GENERATED_TESTS,
                "test_designer",
            ),
        )

    async def test_valutazione_isolata(self):
        for flag in (True, False):
            with self.subTest(prompt_flag=flag):
                self.model.calls.clear()
                self.ctx.outputs.clear()

                await self.designer.handle_generate_test_code_message(
                    TestCodeMessage(
                        "Add two numbers",
                        "def add(a, b):",
                        flag,
                    ),
                    self.ctx,
                )

                # Il TestDesigner originale usa SEMPRE il system prompt.
                self.assert_request_correct()

                self.assertEqual(len(self.ctx.outputs), 1)
                self.assertEqual(len(self.ctx.sent), 0)

                result = self.ctx.outputs[0]

                self.assertIsInstance(result, TestCodeResult)
                self.assertEqual(result.content, GENERATED_TESTS)
                self.assertEqual(result.tokens, 20)
                self.assertGreaterEqual(result.time, 0)
                self.assertIsNone(result.ctx)

    async def test_sistema_con_markdown(self):
        before = TestSystemMessage(
            tokens={"entry_point": 14, "coder": 18},
            time={"entry_point": 0.1, "coder": 0.2},
            messages=6,
            prompt="Add two numbers",
            signature="def add(a, b):",
            original_func="def add(a,b): return a+b",
            code="```python\ndef add(a,b): return a+b\n```",
        )

        await self.designer.handle_test_system_generate_code_message(
            before,
            self.ctx,
        )

        self.assert_request_correct()
        self.assertEqual(len(self.ctx.sent), 1)

        out = self.ctx.sent[0]

        self.assertIsInstance(out, TestSystemMessage)
        self.assertEqual(out.messages, 7)
        self.assertEqual(out.sender, "test_designer")

        # Risposta LLM completa, inclusi i delimitatori Markdown.
        self.assertEqual(out.tests, GENERATED_TESTS)

        # Codice estratto dal blocco Markdown.
        self.assertEqual(out.tests_str, ASSERTION + "\n")

        self.assertEqual(out.original_func, before.original_func)
        self.assertEqual(out.code, before.code)

        self.assertEqual(
            out.tokens,
            {
                "entry_point": 14,
                "coder": 18,
                "test_designer": 20,
            },
        )

        self.assertEqual(out.time["coder"], 0.2)
        self.assertGreaterEqual(out.time["test_designer"], 0)

    async def test_sistema_senza_markdown(self):
        self.model.response_text = ASSERTION

        before = TestSystemMessage(
            tokens={},
            time={},
            messages=3,
            prompt="Add two numbers",
            signature="def add(a, b):",
        )

        await self.designer.handle_test_system_generate_code_message(
            before,
            self.ctx,
        )

        self.assert_request_correct()
        self.assertEqual(len(self.ctx.sent), 1)

        self.assertEqual(self.ctx.sent[0].tests, ASSERTION)
        self.assertEqual(self.ctx.sent[0].tests_str, ASSERTION)

    async def test_workflow_maf(self):
        collector = ResultCollector(id="collector")

        workflow = (
            WorkflowBuilder(start_executor=self.designer)
            .add_edge(self.designer, collector)
            .build()
        )

        with patch.dict(os.environ, {"UI": "False"}):
            result = await workflow.run(
                CodeMessage("Add two numbers", "def add(a, b):")
            )

        self.assertEqual(len(self.model.calls), 1)

        outputs = result.get_outputs()

        self.assertEqual(len(outputs), 1)
        self.assertIsInstance(outputs[0], ExecuteMessage)
        self.assertEqual(outputs[0].tests, GENERATED_TESTS)


if __name__ == "__main__":
    unittest.main()
