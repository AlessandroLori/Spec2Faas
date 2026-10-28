"""Regression tests for literal backslash-n handler source in tool calls.

No network, Ollama, Docker, or Serverledge is accessed.
"""
import ast
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent_framework import Content, Message as LLMMessage

from app.agents.FaaSDeployer import FaasDeployer, validate_deployment_source
from experiments.MessageTypesTest import TestDeployMessage


VALID = (
    "def plus(a, b):\n"
    "    return a + b\n"
    "\n"
    "def handler(params, context):\n"
    "    return {'result': plus(params['a'], params['b'])}"
)
INVALID_ESCAPED = VALID.replace("\n", "\\n")
ARGS = {
    "code": INVALID_ESCAPED,
    "name": "diagnostic_plus",
    "runtime": "python310",
    "memoryMB": 128,
    "CPUDemand": 1,
}


def call_response(arguments):
    return SimpleNamespace(
        text="",
        messages=[LLMMessage(
            role="assistant",
            contents=[Content.from_function_call(
                call_id="call_1",
                name="create_json_serverledge",
                arguments=json.dumps(arguments),
            )],
        )],
        usage_details=SimpleNamespace(
            input_token_count=3, output_token_count=2,
        ),
    )


def end_response():
    return SimpleNamespace(
        text="done",
        messages=[LLMMessage(role="assistant", contents=["done"])],
        usage_details=SimpleNamespace(
            input_token_count=3, output_token_count=2,
        ),
    )


class FakeModel:
    def __init__(self, *responses):
        self.responses = list(responses)

    async def get_response(self, messages, options=None):
        return self.responses.pop(0)


class FakeContext:
    def __init__(self):
        self.sent = []

    async def send_message(self, message, target_id=None):
        self.sent.append((target_id, message))


class TestSourceValidation(unittest.TestCase):
    def test_valid_source_unchanged_with_real_escape_in_string_literal(self):
        source = (
            "def handler(params, context):\n"
            "    return {'result': 'a\\nb'}\n"
        )
        self.assertEqual(validate_deployment_source(source), source)

    def test_literal_newline_escaping_repaired(self):
        self.assertEqual(validate_deployment_source(INVALID_ESCAPED), VALID)
        ast.parse(validate_deployment_source(INVALID_ESCAPED))

    def test_no_arbitrary_repair(self):
        bad = "def handler(params, context):\\n    return ("
        with self.assertRaisesRegex(ValueError, "Invalid Python handler"):
            validate_deployment_source(bad)

    def test_missing_handler_rejected(self):
        with self.assertRaisesRegex(ValueError, "top-level"):
            validate_deployment_source("def plus(a, b): return a + b")

    def test_incomplete_function_rejected(self):
        bad = (
            "def get_closest_vowel(word):\n"
            "    # Implementation missing\n"
            "def handler(params, context):\n"
            "    return get_closest_vowel(params['word'])"
        )
        with self.assertRaisesRegex(ValueError, "Invalid Python handler"):
            validate_deployment_source(bad)

    def test_invalid_escaped_quote_rejected_not_guessed(self):
        bad = 'def handler(params, context):\\n    return {\\"result\\": 1}'
        with self.assertRaises(ValueError):
            validate_deployment_source(bad)


class TestToolPath(unittest.IsolatedAsyncioTestCase):
    async def test_success_deploys_and_records_identical_normalized_source(self):
        agent = FaasDeployer("fake", FakeModel(call_response(ARGS), end_response()))
        ctx = FakeContext()
        with patch(
            "app.agents.FaaSDeployer.create_json_serverledge",
            new_callable=AsyncMock,
        ) as api:
            api.return_value = {"name": "diagnostic_plus"}
            await agent.handle_test_deploy_message(TestDeployMessage("code"), ctx)
            api.assert_awaited_once_with(**{**ARGS, "code": VALID})
        self.assertEqual(ctx.sent[0][0], "assistant")
        result = ctx.sent[0][1]
        self.assertEqual(result.deployed_function, VALID)
        self.assertEqual(result.result, "diagnostic_plus")
        self.assertEqual(result.invocation_attempts, 1)

    async def test_invalid_code_not_sent_to_serverledge(self):
        args = {**ARGS, "code": "def handler(params, context):\\n    return ("}
        agent = FaasDeployer("fake", FakeModel(call_response(args), end_response()))
        ctx = FakeContext()
        with patch(
            "app.agents.FaaSDeployer.create_json_serverledge",
            new_callable=AsyncMock,
        ) as api:
            await agent.handle_test_deploy_message(TestDeployMessage("code"), ctx)
            api.assert_not_awaited()
        self.assertEqual(ctx.sent[0][1].result, "FAIL")


if __name__ == "__main__":
    unittest.main()
