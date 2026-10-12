
import unittest

from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)

from app.agents.coding_agents.messages.MessagesTypes import (
    Message,
    CodeMessage,
    ExecuteMessage,
)


class FakeEntryPoint(Executor):
    def __init__(self, trace):
        super().__init__(id="entry_point")
        self.trace = trace
        self.specification = ""
        self.signature = "def add(a, b):"

    @handler
    async def process(
        self,
        message: Message,
        ctx: WorkflowContext[CodeMessage, Message],
    ) -> None:

        if message.type == "request":
            self.trace.append("entry_point:request")
            self.specification = message.content

            # Prima viene attivato soltanto il Coder.
            await ctx.send_message(
                CodeMessage(self.specification, self.signature),
                target_id="coder",
            )

        elif message.content == "Still waiting":
            self.trace.append("entry_point:waiting")

            # Il Coder ha terminato la prima fase:
            # possiamo attivare il Test Designer.
            await ctx.send_message(
                CodeMessage(self.specification, self.signature),
                target_id="test_designer",
            )

        elif message.type == "test_executor_response":
            self.trace.append("entry_point:final")
            await ctx.yield_output(message)


class FakeCoder(Executor):
    def __init__(self, trace):
        super().__init__(id="coder")
        self.trace = trace

    @handler
    async def process(
        self,
        message: CodeMessage,
        ctx: WorkflowContext[ExecuteMessage],
    ) -> None:
        self.trace.append("coder")

        # Simuliamo la generazione del codice.
        await ctx.send_message(
            ExecuteMessage(
                specification=message.specification,
                code="def add(a, b): return a + b",
                tests="",
                sender="coder",
            )
        )


class FakeTestDesigner(Executor):
    def __init__(self, trace):
        super().__init__(id="test_designer")
        self.trace = trace

    @handler
    async def process(
        self,
        message: CodeMessage,
        ctx: WorkflowContext[ExecuteMessage],
    ) -> None:
        self.trace.append("test_designer")

        await ctx.send_message(
            ExecuteMessage(
                specification=message.specification,
                code="",
                tests="assert add(2, 3) == 5",
                sender="test_designer",
            )
        )


class FakeTestExecutor(Executor):
    def __init__(self, trace):
        super().__init__(id="test_executor")
        self.trace = trace
        self.code = ""
        self.tests = ""

    @handler
    async def process(
        self,
        message: ExecuteMessage,
        ctx: WorkflowContext[Message],
    ) -> None:

        if message.sender == "coder":
            self.trace.append("test_executor:code")
            self.code = message.code

            # Come nell'originale, ancora non abbiamo i test.
            assert self.tests == ""

            await ctx.send_message(
                Message("Still waiting", "test_executor_response")
            )

        elif message.sender == "test_designer":
            self.trace.append("test_executor:tests")
            self.tests = message.tests

            # Ora abbiamo entrambi i dati.
            assert self.code != ""
            assert self.tests != ""

            await ctx.send_message(
                Message(self.code, "test_executor_response")
            )


class TestMAFCodingSequence(unittest.IsolatedAsyncioTestCase):

    async def test_original_coding_order(self):
        trace = []

        entry = FakeEntryPoint(trace)
        coder = FakeCoder(trace)
        designer = FakeTestDesigner(trace)
        executor = FakeTestExecutor(trace)

        workflow = (
            WorkflowBuilder(start_executor=entry)
            .add_edge(entry, coder)
            .add_edge(entry, designer)
            .add_edge(coder, executor)
            .add_edge(designer, executor)
            .add_edge(executor, entry)
            .build()
        )

        result = await workflow.run(
            Message("Add two numbers", "request")
        )

        expected_trace = [
            "entry_point:request",
            "coder",
            "test_executor:code",
            "entry_point:waiting",
            "test_designer",
            "test_executor:tests",
            "entry_point:final",
        ]

        self.assertEqual(trace, expected_trace)

        outputs = result.get_outputs()
        self.assertEqual(len(outputs), 1)
        self.assertIsInstance(outputs[0], Message)
        self.assertEqual(outputs[0].type, "test_executor_response")
        self.assertEqual(
            outputs[0].content,
            "def add(a, b): return a + b",
        )

        print("\nPASS - Sequenza di generazione preservata")
        print(" -> ".join(trace))


if __name__ == "__main__":
    unittest.main()
