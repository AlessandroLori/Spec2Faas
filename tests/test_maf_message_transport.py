
import unittest
from typing import Never

from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    handler,
)

# Riutilizziamo i messaggi ORIGINALI di Spec2FaaS.
from app.agents.coding_agents.messages.MessagesTypes import (
    CodeMessage,
    ExecuteMessage,
)


class FakeCoder(Executor):
    """
    Simula il Coder originale, senza interrogare alcun LLM.
    Serve esclusivamente a verificare il passaggio dei messaggi.
    """

    @handler
    async def handle_code(
        self,
        message: CodeMessage,
        ctx: WorkflowContext[ExecuteMessage],
    ) -> None:

        # Simuliamo una risposta deterministica del Coder.
        generated_code = "def add(a, b): return a + b"

        # Manteniamo il formato originale di ExecuteMessage.
        result = ExecuteMessage(
            specification=message.specification,
            code=generated_code,
            tests="",
            sender="coder",
        )

        # Inoltriamo il messaggio attraverso MAF.
        await ctx.send_message(result)


class FakeTestExecutor(Executor):
    """
    Riceve il messaggio inviato dal FakeCoder.
    Non esegue codice e non avvia Docker.
    """

    @handler
    async def handle_execute(
        self,
        message: ExecuteMessage,
        ctx: WorkflowContext[Never, ExecuteMessage],
    ) -> None:

        # Rendiamo disponibile il messaggio come risultato del test.
        await ctx.yield_output(message)


class TestMAFMessageTransport(unittest.IsolatedAsyncioTestCase):

    async def test_coder_to_test_executor(self):

        # Creiamo i due componenti MAF.
        coder = FakeCoder(id="coder")
        executor = FakeTestExecutor(id="test_executor")

        # Definiamo il collegamento fra i due.
        workflow = (
            WorkflowBuilder(start_executor=coder)
            .add_edge(coder, executor)
            .build()
        )

        # Messaggio identico nella struttura a quello usato
        # dall'Entry Point originale per attivare il Coder.
        initial_message = CodeMessage(
            specification="Add two numbers",
            function_signature="def add(a, b):",
            memory_context="",
        )

        # Eseguiamo il workflow senza LLM e senza Docker.
        result = await workflow.run(initial_message)
        outputs = result.get_outputs()

        # Deve essere arrivato esattamente un messaggio.
        self.assertEqual(len(outputs), 1)

        received = outputs[0]

        # Verifichiamo il tipo originale del messaggio.
        self.assertIsInstance(received, ExecuteMessage)

        # Verifichiamo che le informazioni non siano alterate.
        self.assertEqual(received.specification, "Add two numbers")
        self.assertEqual(received.code, "def add(a, b): return a + b")
        self.assertEqual(received.tests, "")
        self.assertEqual(received.sender, "coder")


if __name__ == "__main__":
    unittest.main()
