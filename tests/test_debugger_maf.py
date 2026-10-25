
import hashlib
import math
import unittest
from types import SimpleNamespace
from typing import Never

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler

from app.agents.coding_agents.Debugger import Debugger, DebuggerProviderError
from app.agents.coding_agents.TestExecutor import TestExecutor
from app.agents.coding_agents.messages.MessagesTypes import (
    DebugMessage,
    ExecuteMessage,
    Message,
)
from experiments.MessageTypesTest import (
    TestDebugMessage, TestDebugResult, TestExecCodeMessage, TestExecCodeResult,
)


FIXED_CODE = "```python\ndef add(a, b):\n    return a + b\n```"
SECOND_CODE = "```python\ndef add(a, b):\n    return b + a\n```"


class FakeModel:
    """
    Modello simulato con risposte predefinite.
    Registra anche i messaggi che riceve.
    """

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    async def get_response(self, messages):
        self.calls.append(list(messages))

        answer = self.answers.pop(0)

        if isinstance(answer, Exception):
            raise answer

        if isinstance(answer, SimpleNamespace):
            return answer

        return SimpleNamespace(
            text=answer,
            usage_details=SimpleNamespace(
                input_token_count=7,
                output_token_count=4,
            ),
        )


class FakeContext:
    """
    Registra i messaggi destinati agli altri agenti.
    """

    def __init__(self):
        self.sent = []
        self.outputs = []

    async def send_message(self, message, target_id=None):
        self.sent.append((target_id, message))

    async def yield_output(self, message):
        self.outputs.append(message)


class FakeDocker:
    """
    Prima esecuzione fallita, seconda riuscita.
    Non avvia un container reale.
    """

    def __init__(self):
        self.outputs = ["AssertionError: simulated", ""]
        self.calls = []

    async def execute_code_blocks(self, blocks):
        self.calls.append(list(blocks))

        output = self.outputs.pop(0)
        return SimpleNamespace(
            output=output,
            exit_code=0 if output == "" else 1,
        )


class ResultCollector(Executor):
    """
    Riceve i messaggi del Debugger in un vero workflow MAF.
    Usa l'ID del Test Executor originale.
    """

    def __init__(self):
        super().__init__(id="test_executor")

    @handler
    async def on_normal(
        self,
        message: DebugMessage,
        ctx: WorkflowContext[Never, DebugMessage],
    ) -> None:
        await ctx.yield_output(message)

    @handler
    async def on_experimental(
        self,
        message: TestDebugResult,
        ctx: WorkflowContext[Never, TestDebugResult],
    ) -> None:
        await ctx.yield_output(message)


class TestDebuggerMAF(unittest.IsolatedAsyncioTestCase):

    def test_system_prompt_immutato(self):
        debugger = Debugger("fake-model", FakeModel([]))

        self.assertEqual(
            hashlib.sha256(
                debugger.SYSTEM_PROMPT.encode()
            ).hexdigest(),
            "a4b1dc5e7e54e9d420358803f980cb4eb3b546e15ad8f49e1fb84edf7efe2e84",
        )

    async def test_normale_e_memoria(self):
        model = FakeModel([FIXED_CODE, SECOND_CODE])
        debugger = Debugger("fake-model", model)
        ctx = FakeContext()

        # Primo tentativo.
        await debugger.handle_debug_code_message(
            DebugMessage("sum", "bad code", "AssertionError"),
            ctx,
        )

        self.assertEqual(
            ctx.sent[-1],
            (
                "test_executor",
                DebugMessage("sum", FIXED_CODE, ""),
            ),
        )

        self.assertEqual(len(model.calls[0]), 2)
        self.assertEqual(model.calls[0][0].role, "system")

        # Controlliamo anche il prompt originale.
        self.assertEqual(
            model.calls[0][1].text,
            "This is the function specification: sum"
            "\nThis is the function code: bad code"
            "\nAnd this is the error message: AssertionError"
            "\nCan you correct the code?",
        )

        # Secondo tentativo: la cronologia deve essere mantenuta.
        await debugger.handle_debug_code_message(
            DebugMessage("sum", FIXED_CODE, "ValueError"),
            ctx,
        )

        self.assertEqual(ctx.sent[-1][1].code, SECOND_CODE)
        self.assertEqual(len(model.calls[1]), 4)

        self.assertEqual(model.calls[1][2].role, "assistant")
        self.assertEqual(model.calls[1][2].text, FIXED_CODE)
        self.assertEqual(debugger._counter, 2)

    async def test_normale_risposta_senza_markdown(self):
        debugger = Debugger(
            "fake-model",
            FakeModel(["plain text"]),
        )

        ctx = FakeContext()

        # Nel percorso normale un output non valido
        # deve essere segnalato come errore.
        with self.assertRaises(ValueError):
            await debugger.handle_debug_code_message(
                DebugMessage("sum", "bad code", "Error"),
                ctx,
            )

        self.assertEqual(ctx.sent, [])

    async def test_sperimentale_reset_e_continuita(self):
        model = FakeModel([
            FIXED_CODE,
            SECOND_CODE,
            FIXED_CODE,
        ])

        debugger = Debugger("fake-model", model)
        ctx = FakeContext()

        # Primo elemento del dataset.
        await debugger.handle_test_debug_code_message(
            TestDebugMessage(
                "sum",
                "bad code",
                "Error",
                True,
            ),
            ctx,
        )

        self.assertEqual(
            ctx.sent[-1],
            (
                "test_executor",
                TestDebugResult(FIXED_CODE, 11),
            ),
        )

        self.assertEqual(len(model.calls[0]), 2)

        # Secondo tentativo sullo stesso elemento.
        await debugger.handle_test_debug_code_message(
            TestDebugMessage(
                "sum",
                FIXED_CODE,
                "Error",
                False,
            ),
            ctx,
        )

        self.assertEqual(len(model.calls[1]), 4)
        self.assertEqual(ctx.sent[-1][1].code, SECOND_CODE)

        # Nuovo elemento: la cronologia deve ripartire.
        await debugger.handle_test_debug_code_message(
            TestDebugMessage(
                "new function",
                "bad",
                "Error",
                True,
            ),
            ctx,
        )

        self.assertEqual(len(model.calls[2]), 2)
        self.assertEqual(debugger._counter, 0)
        self.assertEqual(len(debugger._system_messages), 1)

    async def test_sperimentale_errore_modello(self):
        debugger = Debugger(
            "fake-model",
            FakeModel([RuntimeError("model failure")]),
        )
        ctx = FakeContext()

        # Una chiamata fallita per ragioni infrastrutturali non deve essere
        # trasformata in 10 tentativi o in un dato sperimentale negativo.
        with self.assertRaisesRegex(DebuggerProviderError, "model failure"):
            await debugger.handle_test_debug_code_message(
                TestDebugMessage("sum", "code", "Error", True),
                ctx,
            )

        self.assertEqual(ctx.sent, [])

    async def test_sperimentale_codice_con_token_mancanti(self):
        model = FakeModel([
            SimpleNamespace(text=FIXED_CODE, usage_details={}),
        ])
        debugger = Debugger("fake-model", model)
        ctx = FakeContext()

        await debugger.handle_test_debug_code_message(
            TestDebugMessage("sum", "bad code", "Error", True),
            ctx,
        )

        destination, result = ctx.sent[-1]
        self.assertEqual(destination, "test_executor")
        self.assertEqual(result.code, FIXED_CODE)
        self.assertTrue(math.isnan(result.tokens))

    async def test_ciclo_sperimentale_token_mancanti(self):
        # Controllo end-to-end dei due handler isolati: il metadato mancante
        # non deve trasformare codice corretto in un fallimento.
        debugger = Debugger(
            "fake-model",
            FakeModel([SimpleNamespace(text=FIXED_CODE, usage_details={})]),
        )
        executor = TestExecutor("fake-model", FakeModel([]), FakeDocker())
        executor_ctx = FakeContext()
        debugger_ctx = FakeContext()

        await executor.handle_test_execute_code_message(
            TestExecCodeMessage(
                specification="sum",
                function_signature="add",
                code="def add(a, b):\n    return a - b\n",
                tests="\nassert add(2, 3) == 5\n",
            ),
            executor_ctx,
        )
        target, request = executor_ctx.sent[-1]
        self.assertEqual(target, "debugger")
        self.assertIsInstance(request, TestDebugMessage)

        await debugger.handle_test_debug_code_message(request, debugger_ctx)
        target, correction = debugger_ctx.sent[-1]
        self.assertEqual(target, "test_executor")
        self.assertEqual(correction.code, FIXED_CODE)
        self.assertTrue(math.isnan(correction.tokens))

        await executor.handle_test_debug_result(correction, executor_ctx)
        self.assertEqual(len(executor_ctx.outputs), 1)
        result = executor_ctx.outputs[0]
        self.assertIsInstance(result, TestExecCodeResult)
        self.assertTrue(result.passed)
        self.assertEqual(result.attempts, 1)
        self.assertTrue(math.isnan(result.tokens))

    async def test_sperimentale_risposta_non_testuale(self):
        model = FakeModel([
            SimpleNamespace(text=None, usage_details={}),
        ])
        debugger = Debugger("fake-model", model)
        ctx = FakeContext()

        await debugger.handle_test_debug_code_message(
            TestDebugMessage("sum", "bad code", "Error", True),
            ctx,
        )

        self.assertEqual(
            ctx.sent[-1],
            ("test_executor", TestDebugResult("", 1048575)),
        )

    async def test_sperimentale_formato_errato(self):
        debugger = Debugger(
            "fake-model",
            FakeModel(["no fenced code"]),
        )

        ctx = FakeContext()

        await debugger.handle_test_debug_code_message(
            TestDebugMessage(
                "sum",
                "code",
                "Error",
                True,
            ),
            ctx,
        )

        self.assertEqual(
            ctx.sent[-1][1],
            TestDebugResult("", 1048575),
        )

    async def test_ciclo_con_test_executor(self):
        # Entrambi gli agenti sono le vere classi migrate.
        debugger = Debugger(
            "fake-debugger",
            FakeModel([FIXED_CODE]),
        )

        executor = TestExecutor(
            "fake-executor",
            FakeModel([
                "```python\npass\n```",
                "```python\npass\n```",
            ]),
            FakeDocker(),
        )

        executor_ctx = FakeContext()
        debugger_ctx = FakeContext()

        # 1. Il Test Executor rileva un errore.
        await executor.handle_execute_code_message(
            ExecuteMessage(
                "sum",
                "bad code",
                "tests",
                "coder",
            ),
            executor_ctx,
        )

        destination, request = executor_ctx.sent[-1]

        self.assertEqual(destination, "debugger")
        self.assertIsInstance(request, DebugMessage)

        # 2. Il Debugger produce il codice corretto.
        await debugger.handle_debug_code_message(
            request,
            debugger_ctx,
        )

        destination, corrected = debugger_ctx.sent[-1]

        self.assertEqual(destination, "test_executor")

        # 3. Il Test Executor riceve la correzione
        # e ripete la verifica.
        await executor.handle_debug_code_message(
            corrected,
            executor_ctx,
        )

        self.assertEqual(
            executor_ctx.sent[-1],
            (
                "entry_point",
                Message(
                    FIXED_CODE,
                    "test_executor_response",
                ),
            ),
        )

    async def test_workflow_maf(self):
        # Questo test usa realmente WorkflowBuilder di MAF.
        debugger = Debugger(
            "fake-model",
            FakeModel([FIXED_CODE]),
        )

        collector = ResultCollector()

        workflow = (
            WorkflowBuilder(start_executor=debugger)
            .add_edge(debugger, collector)
            .build()
        )

        result = await workflow.run(
            DebugMessage(
                "sum",
                "bad",
                "AssertionError",
            )
        )

        self.assertEqual(
            result.get_outputs(),
            [
                DebugMessage(
                    "sum",
                    FIXED_CODE,
                    "",
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
