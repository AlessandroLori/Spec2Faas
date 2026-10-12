
"""Test di regressione deterministici per il Coder MAF."""

import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.agents.coding_agents.Coder import Coder
from app.agents.coding_agents.messages.MessagesTypes import (
    CodeMessage,
    ExecuteMessage,
)

from experiments.MessageTypesTest import (
    TestCodeMessage,
    TestCodeResult,
    TestSystemMessage,
)


# SHA-256 del system prompt del Coder AutoGen originale.
ORIGINAL_SYSTEM_PROMPT_SHA256 = (
    "a60a2908dfdb814d4e80487f3deb3d4fa1c533e5b6641195faea47901eba93e3"
)

GENERATED_CODE = (
    "```python\n"
    "def add(a, b):\n"
    "    return a + b\n"
    "```"
)


class FakeModelClient:
    """
    Simula un LLM e registra ogni richiesta.
    """

    def __init__(self, text=GENERATED_CODE):
        self.text = text
        self.calls = []

    async def get_response(self, messages):
        self.calls.append(list(messages))

        return SimpleNamespace(
            text=self.text,
            usage_details=SimpleNamespace(
                input_token_count=11,
                output_token_count=7,
            ),
        )


class FakeContext:
    """
    Registra i messaggi inviati dal Coder
    e gli output restituiti al workflow.
    """

    def __init__(self):
        self.sent = []
        self.outputs = []

    async def send_message(self, message):
        self.sent.append(message)

    async def yield_output(self, message):
        self.outputs.append(message)


class TestCoderMAF(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.model = FakeModelClient()

        self.coder = Coder(
            llm="fake-model",
            model_client=self.model,
        )

        self.ctx = FakeContext()

    def check_messages(self, expected_user, with_system):
        self.assertEqual(len(self.model.calls), 1)

        msgs = self.model.calls[0]

        self.assertEqual(
            len(msgs),
            2 if with_system else 1,
        )

        roles = [
            getattr(msg.role, "value", msg.role)
            for msg in msgs
        ]

        self.assertEqual(
            roles,
            ["system", "user"] if with_system else ["user"],
        )

        if with_system:
            self.assertEqual(
                msgs[0].text,
                Coder.SYSTEM_PROMPT,
            )

        self.assertEqual(
            msgs[-1].text,
            expected_user,
        )

    def test_system_prompt_identico_a_autogen(self):
        digest = hashlib.sha256(
            Coder.SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest()

        self.assertEqual(
            digest,
            ORIGINAL_SYSTEM_PROMPT_SHA256,
        )

    async def test_normale(self):
        with patch.dict("os.environ", {"UI": "False"}):
            await self.coder.handle_generate_code_message(
                CodeMessage(
                    "Add two numbers",
                    "def add(a, b):",
                    "ignored",
                ),
                self.ctx,
            )

        self.check_messages(
            "Function specification: Add two numbers"
            "\nFunction signature: def add(a, b):",
            with_system=True,
        )

        self.assertEqual(len(self.ctx.sent), 1)

        msg = self.ctx.sent[0]

        self.assertIsInstance(msg, ExecuteMessage)

        self.assertEqual(
            (msg.specification, msg.code, msg.tests, msg.sender),
            ("Add two numbers", GENERATED_CODE, "", "coder"),
        )

    async def test_valutazione_isolata_con_e_senza_system_prompt(self):

        for with_system in (True, False):
            with self.subTest(with_system=with_system):

                self.model.calls.clear()
                self.ctx.outputs.clear()

                await self.coder.handle_generate_test_code_message(
                    TestCodeMessage(
                        "Specification",
                        "def f():",
                        with_system,
                    ),
                    self.ctx,
                )

                self.check_messages(
                    "Function specification: Specification"
                    "\nFunction signature: def f():",
                    with_system=with_system,
                )

                self.assertEqual(len(self.ctx.outputs), 1)

                msg = self.ctx.outputs[0]

                self.assertIsInstance(msg, TestCodeResult)
                self.assertEqual(msg.content, GENERATED_CODE)
                self.assertEqual(msg.tokens, 18)
                self.assertGreaterEqual(msg.time, 0)

                # Cancellazione Docker da migrare successivamente.
                self.assertIsNone(msg.ctx)

    async def test_valutazione_end_to_end(self):

        start = TestSystemMessage(
            tokens={"entry_point": 12},
            time={"entry_point": 1.5},
            messages=4,
            prompt="Add two numbers",
            signature="def add(a, b):",
        )

        await self.coder.handle_test_system_generate_code_message(
            start,
            self.ctx,
        )

        self.check_messages(
            "Function specification: Add two numbers"
            "\nFunction signature: def add(a, b):",
            with_system=True,
        )

        self.assertEqual(len(self.ctx.sent), 1)

        msg = self.ctx.sent[0]

        self.assertIsInstance(msg, TestSystemMessage)

        self.assertEqual(msg.messages, 5)
        self.assertTrue(msg.new_chat)
        self.assertEqual(msg.sender, "coder")

        self.assertEqual(msg.code, GENERATED_CODE)

        self.assertEqual(
            msg.original_func,
            "def add(a, b):\n    return a + b\n",
        )

        self.assertEqual(
            msg.tokens,
            {"entry_point": 12, "coder": 18},
        )

        self.assertEqual(msg.time["entry_point"], 1.5)
        self.assertGreaterEqual(msg.time["coder"], 0)


if __name__ == "__main__":
    unittest.main()
