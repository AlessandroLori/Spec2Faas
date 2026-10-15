
import unittest
from types import SimpleNamespace
from typing import Never

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler

from app.agents.coding_agents.Debugger import Debugger
from app.agents.coding_agents.TestExecutor import TestExecutor
from app.agents.coding_agents.messages.MessagesTypes import (
    ExecuteMessage,
    Message,
)
from app.agents.coding_agents.utils.DockerCodeExecutor import DockerCodeExecutor


# Funzione originale volutamente sbagliata.
BAD_CODE = "```python\ndef add(a, b):\n    return a - b\n```"

# Correzione che il Debugger dovrà produrre.
CORRECTED_CODE = "```python\ndef add(a, b):\n    return a + b\n```"

# Test che la funzione deve superare.
TESTS = "```python\nassert add(2, 3) == 5\n```"


class ScriptedModel:
    """
    Simula un modello LLM con risposte deterministiche.

    Non utilizza API esterne, ma permette ai veri agenti MAF
    di ricevere risposte e comunicare normalmente.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def get_response(self, messages):
        # Registriamo i messaggi inviati al modello.
        self.calls.append(list(messages))

        if not self.responses:
            raise AssertionError("Chiamata LLM inattesa")

        # Restituiamo la prossima risposta predefinita.
        return SimpleNamespace(
            text=self.responses.pop(0)
        )


class EntryPointCollector(Executor):
    """
    Riceve il risultato finale del Test Executor.

    Sostituisce solamente il destinatario finale:
    non modifica il funzionamento del Test Executor.
    """

    def __init__(self):
        super().__init__(id="entry_point")

    @handler
    async def collect(
        self,
        message: Message,
        ctx: WorkflowContext[Never, Message],
    ) -> None:
        await ctx.yield_output(message)


class DockerWorkflowRealSmoke(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # Utilizziamo il DockerCodeExecutor reale.
        self.docker_executor = DockerCodeExecutor()
        await self.docker_executor.start()

    async def asyncTearDown(self):
        # Chiudiamo Docker anche se il test fallisce.
        await self.docker_executor.stop()

    async def test_ciclo_maf_debugger_test_executor(self):

        # Il primo output del modello genera codice errato.
        # Il secondo genera il programma corretto.
        test_executor_model = ScriptedModel([
            "```python\n"
            "def add(a, b):\n"
            "    return a - b\n"
            "assert add(2, 3) == 5\n"
            "```",

            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "assert add(2, 3) == 5\n"
            "```",
        ])

        # Il Debugger propone la correzione.
        debugger_model = ScriptedModel([
            CORRECTED_CODE
        ])

        # Creiamo gli agenti MAF realmente migrati.
        test_executor = TestExecutor(
            "fake-llm",
            test_executor_model,
            self.docker_executor,
        )

        debugger = Debugger(
            "fake-llm",
            debugger_model,
        )

        collector = EntryPointCollector()

        # Costruiamo un vero workflow MAF con un ciclo.
        workflow = (
            WorkflowBuilder(
                start_executor=test_executor,
                max_iterations=20,
            )
            .add_edge(test_executor, debugger)
            .add_edge(debugger, test_executor)
            .add_edge(test_executor, collector)
            .build()
        )

        # Il messaggio iniziale contiene codice e test.
        # Gli altri test hanno già verificato il loro
        # arrivo separato da Coder e Test Designer.
        result = await workflow.run(
            ExecuteMessage(
                specification="Scrivi una funzione che sommi due numeri",
                code=BAD_CODE,
                tests=TESTS,
                sender="coder",
            )
        )

        # Il risultato finale deve essere la funzione corretta.
        self.assertEqual(
            result.get_outputs(),
            [
                Message(
                    CORRECTED_CODE,
                    "test_executor_response",
                )
            ],
        )

        # Test Executor: prima esecuzione e secondo tentativo.
        self.assertEqual(
            len(test_executor_model.calls),
            2,
        )

        # Il Debugger deve essere stato chiamato esattamente una volta.
        self.assertEqual(
            len(debugger_model.calls),
            1,
        )

        # Prima del debugging il Test Executor usa il codice errato.
        self.assertIn(
            BAD_CODE,
            test_executor_model.calls[0][1].text,
        )

        # Il Debugger deve ricevere l'errore prodotto da Docker.
        self.assertIn(
            "AssertionError",
            debugger_model.calls[0][1].text,
        )

        # Dopo il debugging il Test Executor usa il codice corretto.
        self.assertIn(
            CORRECTED_CODE,
            test_executor_model.calls[1][1].text,
        )


if __name__ == "__main__":
    unittest.main()
