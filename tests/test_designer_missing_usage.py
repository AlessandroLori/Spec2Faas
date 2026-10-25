import math
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.agents.coding_agents.TestDesigner import TestDesigner
from experiments.MessageTypesTest import TestCodeMessage


# Copertura della sola modalità sperimentale Q2.
# Nessuna chiamata reale a Ollama o Docker.
class TestDesignerMissingUsage(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def message():
        return TestCodeMessage(
            specification="Add two integers",
            function_signature="add",
            prompt=True,
        )

    @staticmethod
    def context():
        return SimpleNamespace(yield_output=AsyncMock())

    async def test_missing_usage_preserves_generated_tests_with_nan(self):
        client = SimpleNamespace(get_response=AsyncMock(return_value=SimpleNamespace(
            text="```python\nassert add(2, 3) == 5\n```",
            usage_details={},
        )))
        designer = TestDesigner("fake", model_client=client)
        ctx = self.context()

        await designer.handle_generate_test_code_message(self.message(), ctx)

        result = ctx.yield_output.await_args.args[0]
        self.assertIn("assert add(2, 3) == 5", result.content)
        self.assertTrue(math.isnan(result.tokens))
        client.get_response.assert_awaited_once()

    async def test_complete_usage_still_counts_exact_tokens(self):
        client = SimpleNamespace(get_response=AsyncMock(return_value=SimpleNamespace(
            text="```python\nassert add(2, 3) == 5\n```",
            usage_details={"input_token_count": 39, "output_token_count": 2},
        )))
        designer = TestDesigner("fake", model_client=client)
        ctx = self.context()

        await designer.handle_generate_test_code_message(self.message(), ctx)

        result = ctx.yield_output.await_args.args[0]
        self.assertEqual(result.tokens, 41)

    async def test_empty_text_with_missing_usage_still_raises(self):
        client = SimpleNamespace(get_response=AsyncMock(return_value=SimpleNamespace(
            text="  ", usage_details={},
        )))
        designer = TestDesigner("fake", model_client=client)
        ctx = self.context()

        with self.assertRaisesRegex(RuntimeError, "risposta vuota"):
            await designer.handle_generate_test_code_message(self.message(), ctx)
        ctx.yield_output.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
