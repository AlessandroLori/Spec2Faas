import math
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.agents.coding_agents.Coder import Coder
from experiments.MessageTypesTest import TestCodeMessage, TestSystemMessage


# Test deterministici: nessuna inferenza reale, nessun container Docker.
GENERATED = "```python\ndef add(a, b):\n    return a + b\n```"


class TestCoderMissingUsage(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def coder(text=GENERATED, usage=None):
        response = SimpleNamespace(text=text, usage_details=usage)
        client = SimpleNamespace(get_response=AsyncMock(return_value=response))
        return Coder("fake", model_client=client), client

    @staticmethod
    def isolated_message():
        return TestCodeMessage("Add two numbers", "def add(a, b):", True)

    @staticmethod
    def isolated_context():
        return SimpleNamespace(yield_output=AsyncMock())

    @staticmethod
    def system_message():
        return TestSystemMessage(
            tokens={"assistant": 7},
            time={"assistant": 0.2},
            messages=2,
            prompt="Add two numbers",
            signature="def add(a, b):",
        )

    async def test_isolated_missing_usage_preserves_code_and_nan(self):
        coder, client = self.coder(usage={})
        ctx = self.isolated_context()
        await coder.handle_generate_test_code_message(self.isolated_message(), ctx)
        result = ctx.yield_output.await_args.args[0]
        self.assertEqual(result.content, GENERATED)
        self.assertTrue(math.isnan(result.tokens))
        self.assertGreaterEqual(result.time, 0)
        client.get_response.assert_awaited_once()

    async def test_isolated_total_only_is_exact(self):
        coder, _ = self.coder(usage={"total_token_count": 41})
        ctx = self.isolated_context()
        await coder.handle_generate_test_code_message(self.isolated_message(), ctx)
        self.assertEqual(ctx.yield_output.await_args.args[0].tokens, 41)

    async def test_isolated_input_and_output_keep_original_accounting(self):
        coder, _ = self.coder(usage={
            "input_token_count": 39, "output_token_count": 2,
            "total_token_count": 1000,
        })
        ctx = self.isolated_context()
        await coder.handle_generate_test_code_message(self.isolated_message(), ctx)
        self.assertEqual(ctx.yield_output.await_args.args[0].tokens, 41)

    async def test_isolated_blank_response_with_missing_usage_returns_nan(self):
        coder, _ = self.coder(text="  ", usage={})
        ctx = self.isolated_context()
        await coder.handle_generate_test_code_message(self.isolated_message(), ctx)
        result = ctx.yield_output.await_args.args[0]
        self.assertEqual(result.content, "  ")
        self.assertTrue(math.isnan(result.tokens))

    async def test_isolated_blank_response_with_usage_keeps_actual_tokens(self):
        coder, _ = self.coder(
            text="", usage={"input_token_count": 500, "output_token_count": 2500}
        )
        ctx = self.isolated_context()
        await coder.handle_generate_test_code_message(self.isolated_message(), ctx)
        result = ctx.yield_output.await_args.args[0]
        self.assertEqual(result.content, "")
        self.assertEqual(result.tokens, 3000)

    async def test_end_to_end_blank_response_still_raises(self):
        coder, _ = self.coder(text="", usage={})
        ctx = SimpleNamespace(send_message=AsyncMock())
        with self.assertRaisesRegex(RuntimeError, "risposta vuota"):
            await coder.handle_test_system_generate_code_message(
                self.system_message(), ctx
            )
        ctx.send_message.assert_not_awaited()

    async def test_isolated_non_text_response_raises(self):
        coder, _ = self.coder(text=None, usage={})
        ctx = self.isolated_context()
        with self.assertRaisesRegex(TypeError, "text response"):
            await coder.handle_generate_test_code_message(self.isolated_message(), ctx)
        ctx.yield_output.assert_not_awaited()

    async def test_end_to_end_missing_usage_preserves_generated_code(self):
        coder, client = self.coder(usage={})
        ctx = SimpleNamespace(send_message=AsyncMock())
        await coder.handle_test_system_generate_code_message(self.system_message(), ctx)
        response = ctx.send_message.await_args.args[0]
        self.assertEqual(response.original_func, "def add(a, b):\n    return a + b\n")
        self.assertTrue(math.isnan(response.tokens["coder"]))
        self.assertEqual(response.tokens["assistant"], 7)
        self.assertEqual(response.messages, 3)
        client.get_response.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
