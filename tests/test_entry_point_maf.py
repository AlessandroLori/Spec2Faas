
import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.agents.coding_agents.EntryPoint import EntryPoint
from app.agents.coding_agents.messages.MessagesTypes import (
    CodeMessage,
    Message,
)
from experiments.MessageTypesTest import TestSystemMessage


ORIGINAL_PROMPT_SHA256 = (
    "8167c871a76019b7f71d2ddae137321daaf3bb899e80359b451e674a34584ac5"
)


class FakeModelClient:
    def __init__(self):
        self.requests = []

    async def get_response(self, messages):
        self.requests.append(list(messages))

        return SimpleNamespace(
            text="def add(a, b):",
            usage_details=SimpleNamespace(
                input_token_count=10,
                output_token_count=4,
            ),
        )


class FakeContext:
    def __init__(self):
        self.sent = []
        self.outputs = []

    async def send_message(self, message, target_id=None):
        self.sent.append((target_id, message))

    async def yield_output(self, message):
        self.outputs.append(message)


class TestEntryPointMAF(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.model = FakeModelClient()
        self.entry = EntryPoint(
            llm="fake-model",
            model_client=self.model,
        )
        self.ctx = FakeContext()

    def check_prompt(self, expected_user):
        self.assertEqual(len(self.model.requests), 1)

        messages = self.model.requests[0]

        self.assertEqual(len(messages), 2)

        roles = [
            getattr(m.role, "value", m.role)
            for m in messages
        ]

        self.assertEqual(roles, ["system", "user"])
        self.assertEqual(
            messages[0].text,
            EntryPoint.SYSTEM_PROMPT,
        )
        self.assertEqual(messages[1].text, expected_user)

    def test_system_prompt_invariato(self):
        digest = hashlib.sha256(
            EntryPoint.SYSTEM_PROMPT.encode()
        ).hexdigest()

        self.assertEqual(digest, ORIGINAL_PROMPT_SHA256)

    async def test_ordine_normale(self):

        with patch(
            "app.agents.coding_agents.EntryPoint.dialogue"
        ):
            await self.entry.handle_assistant_message(
                Message("Add two numbers", "request"),
                self.ctx,
            )

        self.check_prompt("Add two numbers")

        # La prima destinazione deve essere il Coder.
        self.assertEqual(len(self.ctx.sent), 1)

        target, coder_msg = self.ctx.sent[0]

        self.assertEqual(target, "coder")
        self.assertIsInstance(coder_msg, CodeMessage)
        self.assertEqual(
            coder_msg.specification,
            "Add two numbers",
        )
        self.assertEqual(
            coder_msg.function_signature,
            "def add(a, b):",
        )

        # Il Test Executor comunica che mancano i test.
        await self.entry.handle_assistant_message(
            Message("Still waiting", "test_executor_response"),
            self.ctx,
        )

        # Ora viene attivato il Test Designer.
        self.assertEqual(len(self.ctx.sent), 2)
        self.assertEqual(
            self.ctx.sent[1][0],
            "test_designer",
        )
        self.assertEqual(self.ctx.sent[1][1], coder_msg)

        # Il risultato finale viene restituito al chiamante.
        final = Message(
            "def add(a, b): return a + b",
            "test_executor_response",
        )

        await self.entry.handle_assistant_message(
            final,
            self.ctx,
        )

        self.assertEqual(self.ctx.outputs, [final])

        # La firma deve essere generata una sola volta.
        self.assertEqual(len(self.model.requests), 1)

    async def test_ordine_sperimentale_e_metriche(self):

        initial = TestSystemMessage(
            tokens={"assistant": 9},
            time={"assistant": 0.5},
            messages=4,
            prompt="Add two numbers",
        )

        await self.entry.handle_test_system_assistant_message(
            initial,
            self.ctx,
        )

        self.check_prompt("Add two numbers")

        self.assertEqual(len(self.ctx.sent), 1)

        target, to_coder = self.ctx.sent[0]

        self.assertEqual(target, "coder")
        self.assertIsInstance(to_coder, TestSystemMessage)
        self.assertEqual(to_coder.messages, 5)
        self.assertEqual(to_coder.signature, "def add(a, b):")

        # 10 token in input + 4 token in output.
        self.assertEqual(
            to_coder.tokens,
            {"assistant": 9, "entry_point": 14},
        )

        self.assertEqual(
            to_coder.time["assistant"], 0.5
        )
        self.assertGreaterEqual(
            to_coder.time["entry_point"], 0
        )

        # Simuliamo la risposta dopo la generazione del codice.
        after_coder = TestSystemMessage(
            tokens={
                "assistant": 9,
                "entry_point": 14,
                "coder": 20,
            },
            time={
                "assistant": 0.5,
                "entry_point": 0.1,
                "coder": 0.2,
            },
            messages=6,
            prompt="Add two numbers",
            signature="def add(a, b):",
            code="```python\ndef add(a, b): return a + b\n```",
            original_func="def add(a, b): return a + b",
        )

        await self.entry.handle_test_system_assistant_message(
            after_coder,
            self.ctx,
        )

        # Ora deve essere attivato il Test Designer.
        self.assertEqual(len(self.ctx.sent), 2)

        target, to_designer = self.ctx.sent[1]

        self.assertEqual(target, "test_designer")
        self.assertEqual(to_designer.messages, 7)
        self.assertEqual(to_designer.tokens["coder"], 20)
        self.assertEqual(to_designer.code, after_coder.code)
        self.assertEqual(
            to_designer.original_func,
            after_coder.original_func,
        )

        # Simuliamo il risultato finale del Test Executor.
        final = TestSystemMessage(
            tokens=to_designer.tokens,
            time=to_designer.time,
            messages=8,
            type="test_executor_response",
            generated=True,
        )

        await self.entry.handle_test_system_assistant_message(
            final,
            self.ctx,
        )

        self.assertEqual(self.ctx.outputs, [final])
        self.assertEqual(len(self.model.requests), 1)


if __name__ == "__main__":
    unittest.main()
