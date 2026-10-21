
import hashlib
import unittest
from types import SimpleNamespace

from app.agents.coding_agents.TestExecutor import TestExecutor
from app.agents.coding_agents.messages.MessagesTypes import (
    DebugMessage,
    ExecuteMessage,
    Message,
)
from experiments.MessageTypesTest import (
    TestDebugMessage,
    TestDebugResult,
    TestExecCodeMessage,
    TestSystemMessage,
)


CODE = "```python\ndef add(a,b):\n    return a+b\n```"
TESTS = "```python\nassert add(2,3)==5\n```"


class FakeModel:
    def __init__(self, text="```python\npass\n```"):
        self.text = text
        self.calls = []

    async def get_response(self, messages):
        self.calls.append(list(messages))

        return SimpleNamespace(
            text=self.text,
            usage_details=SimpleNamespace(
                input_token_count=6,
                output_token_count=4,
            ),
        )


class FakeDockerExecutor:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def execute_code_blocks(self, blocks):
        self.calls.append(list(blocks))

        output = self.outputs.pop(0)
        return SimpleNamespace(
            output=output,
            exit_code=0 if output == "" else 1,
        )


class FakeContext:
    def __init__(self):
        self.sent = []
        self.outputs = []

    async def send_message(self, message, target_id=None):
        self.sent.append((target_id, message))

    async def yield_output(self, result):
        self.outputs.append(result)


class TestTestExecutorMAF(unittest.IsolatedAsyncioTestCase):

    def new(self, outputs=None):
        if outputs is None:
            outputs = [""]

        self.model = FakeModel()
        self.runner = FakeDockerExecutor(outputs)

        self.agent = TestExecutor(
            "fake",
            self.model,
            self.runner,
        )

        self.ctx = FakeContext()

    def test_prompt(self):
        self.new()

        digest = hashlib.sha256(
            self.agent.SYSTEM_PROMPT.encode()
        ).hexdigest()

        self.assertEqual(
            digest,
            "51431d25866f42b5d7875b3edbcaa183cd1df41e80cac36709fac0303508b8a4",
        )

    async def test_timeout_is_not_success(self):
        self.new(["\n Timeout"])
        # Reproduce the real Docker result: status 124, text has no 'Error'.
        async def timeout_run(blocks):
            return SimpleNamespace(exit_code=124, output="\n Timeout")
        self.runner.execute_code_blocks = timeout_run
        await self.agent.handle_execute_code_message(
            ExecuteMessage("add", CODE, "", "coder"), self.ctx,
        )
        await self.agent.handle_execute_code_message(
            ExecuteMessage("add", "", TESTS, "test_designer"), self.ctx,
        )
        self.assertEqual(self.ctx.sent[-1][0], "debugger")
        self.assertTrue(self.agent._awaiting_debugger)

    async def test_wait_then_success(self):
        self.new()

        await self.agent.handle_execute_code_message(
            ExecuteMessage("add numbers", CODE, "", "coder"),
            self.ctx,
        )

        self.assertEqual(
            self.ctx.sent[0],
            (
                "entry_point",
                Message(
                    "Still waiting",
                    "test_executor_response",
                ),
            ),
        )

        self.assertEqual(len(self.model.calls), 0)

        await self.agent.handle_execute_code_message(
            ExecuteMessage(
                "add numbers",
                "",
                TESTS,
                "test_designer",
            ),
            self.ctx,
        )

        self.assertEqual(len(self.model.calls), 1)

        self.assertEqual(
            self.model.calls[0][1].text,
            "Function code: "
            + CODE
            + "\nFunction tests: "
            + TESTS,
        )

        self.assertEqual(
            self.ctx.sent[-1],
            (
                "entry_point",
                Message(
                    CODE,
                    "test_executor_response",
                ),
            ),
        )

        self.assertEqual(len(self.runner.calls), 1)

    async def test_normal_debug(self):
        self.new([
            "AssertionError: wrong",
            "",
        ])

        await self.agent.handle_execute_code_message(
            ExecuteMessage("add", CODE, "", "coder"),
            self.ctx,
        )

        await self.agent.handle_execute_code_message(
            ExecuteMessage(
                "add",
                "",
                TESTS,
                "test_designer",
            ),
            self.ctx,
        )

        target, debug = self.ctx.sent[-1]

        self.assertEqual(target, "debugger")
        self.assertIsInstance(debug, DebugMessage)
        self.assertEqual(debug.code, CODE)

        await self.agent.handle_debug_code_message(
            DebugMessage(
                "add",
                "new code",
                "",
            ),
            self.ctx,
        )

        self.assertEqual(
            self.ctx.sent[-1],
            (
                "entry_point",
                Message(
                    "new code",
                    "test_executor_response",
                ),
            ),
        )

        self.assertEqual(len(self.model.calls), 2)

    async def test_prebuilt_image_skips_runtime_pip(self):
        self.new()
        self.runner.image = "spec2faas-eval:py3"
        await self.agent.handle_test_execute_code_message(
            TestExecCodeMessage(
                "add", "add",
                "def add(a,b):return a+b\n",
                "def check(f): assert f(2,3)==5\n",
            ),
            self.ctx,
        )
        self.assertTrue(self.ctx.outputs[-1].passed)
        self.assertEqual(len(self.runner.calls[-1]), 1)
        self.assertEqual(self.runner.calls[-1][0].language, "python")

    async def test_isolated_success(self):
        self.new()

        await self.agent.handle_test_execute_code_message(
            TestExecCodeMessage(
                "add",
                "add",
                "def add(a,b):return a+b\n",
                "def check(f): assert f(2,3)==5\n",
            ),
            self.ctx,
        )

        self.assertEqual(len(self.ctx.outputs), 1)
        self.assertTrue(self.ctx.outputs[0].passed)

        self.assertEqual(
            self.runner.calls[0][0].code,
            "pip install numpy\n",
        )

        self.assertEqual(
            self.runner.calls[0][1].language,
            "python",
        )

    async def test_isolated_debug(self):
        self.new([
            "AssertionError",
            "",
        ])

        await self.agent.handle_test_execute_code_message(
            TestExecCodeMessage(
                "add",
                "add",
                "def add(a,b):return a-b\n",
                "def check(f): assert f(2,3)==5\n",
            ),
            self.ctx,
        )

        self.assertIsInstance(
            self.ctx.sent[-1][1],
            TestDebugMessage,
        )

        await self.agent.handle_test_debug_result(
            TestDebugResult(
                "```python\ndef add(a,b):return a+b\n```",
                3,
            ),
            self.ctx,
        )

        self.assertTrue(self.ctx.outputs[0].passed)
        self.assertEqual(self.ctx.outputs[0].tokens, 3)
        self.assertEqual(self.ctx.outputs[0].attempts, 1)

    async def test_system_success(self):
        self.new()

        initial = TestSystemMessage(
            tokens={"coder": 5},
            time={},
            prompt="add",
            signature="def add(a,b):",
            code=CODE,
            new_chat=True,
            sender="coder",
        )

        await self.agent.handle_test_system_execute_code_message(
            initial,
            self.ctx,
        )

        self.assertEqual(
            self.ctx.sent[-1][0],
            "entry_point",
        )

        after_designer = TestSystemMessage(
            tokens={"coder": 5},
            time={},
            prompt="add",
            signature="def add(a,b):",
            tests=TESTS,
            sender="test_designer",
            messages=6,
        )

        await self.agent.handle_test_system_execute_code_message(
            after_designer,
            self.ctx,
        )

        target, result = self.ctx.sent[-1]

        self.assertEqual(target, "entry_point")
        self.assertTrue(result.generated)
        self.assertEqual(result.tokens["test_executor"], 10)
        self.assertEqual(result.tokens["debugger"], 0)

        self.assertEqual(
            result.final_func,
            "def add(a,b):\n    return a+b\n",
        )

    async def test_system_debug(self):
        self.new([
            "Traceback: ValueError",
            "",
        ])

        initial = TestSystemMessage(
            tokens={},
            time={},
            prompt="add",
            signature="def add(a,b):",
            code=CODE,
            new_chat=True,
            sender="coder",
        )

        await self.agent.handle_test_system_execute_code_message(
            initial,
            self.ctx,
        )

        after_designer = TestSystemMessage(
            tokens={},
            time={},
            prompt="add",
            signature="def add(a,b):",
            tests=TESTS,
            sender="test_designer",
            messages=5,
        )

        await self.agent.handle_test_system_execute_code_message(
            after_designer,
            self.ctx,
        )

        self.assertIsInstance(
            self.ctx.sent[-1][1],
            TestDebugMessage,
        )

        await self.agent.handle_test_debug_result(
            TestDebugResult(
                "```python\ndef add(a,b): return a+b\n```",
                2,
            ),
            self.ctx,
        )

        final = self.ctx.sent[-1][1]

        self.assertTrue(final.generated)
        self.assertEqual(final.attempts, 1)
        self.assertEqual(final.tokens["debugger"], 2)
        self.assertEqual(final.tokens["test_executor"], 20)


if __name__ == "__main__":
    unittest.main()
