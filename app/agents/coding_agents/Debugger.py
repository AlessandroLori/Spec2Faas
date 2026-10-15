
import os
import re

from agent_framework import Executor, Message as LLMMessage, WorkflowContext, handler

from .messages.MessagesTypes import DebugMessage
from .utils.Utils import dialogue, print_green, print_purple
from experiments.MessageTypesTest import TestDebugMessage, TestDebugResult


class Debugger(Executor):
    SYSTEM_PROMPT = (
        "You are a very skilled software programmer specialized in debugging code."
        "As a debugger, you are required to correct the code given as input"
        "<INSTRUCTIONS>"
        "Step 1: inspect the code in input and the error message."
        "Step 2: explain the error message and understand why the code is not working"
        "Step 3: correct the code."
        "Return ONLY the code of the function in the specified format."
        "It is important to remember previous corrections of the same function."
        "</INSTRUCTIONS>"
        "<OUTPUT FORMAT>"
        "```python"
        "[Code]"
        "```"
        "</OUTPUT FORMAT>"
    )

    def __init__(self, llm: str, model_client, server=None) -> None:
        super().__init__(id="debugger")

        self._model_client = model_client

        # System prompt originale, rappresentato come messaggio MAF.
        self._system_messages = [
            LLMMessage(
                role="system",
                contents=[self.SYSTEM_PROMPT],
            )
        ]

        # Cronologia utilizzata per mantenere le correzioni precedenti.
        self._debug_chat = list(self._system_messages)

        self._counter = 0
        self._llm = llm
        self._role = "Debugger"
        self._server = server

        print_green(f"Hi I'm the debugger and I use {self._llm}.")

    @staticmethod
    def _build_prompt(
        specification: str,
        code: str,
        error_message: str,
    ) -> str:
        # Manteniamo esattamente il prompt originale.
        return (
            "This is the function specification: " + specification
            + "\nThis is the function code: " + code
            + "\nAnd this is the error message: " + error_message
            + "\nCan you correct the code?"
        )

    @staticmethod
    def _extract_code(text: str) -> str:
        # Espressione regolare identica a quella AutoGen.
        match = re.search(
            r"(```python\n.*?```)",
            text,
            re.DOTALL,
        )

        if match is None:
            raise ValueError(
                "Debugger response does not contain a python code block"
            )

        return match.group(1).strip()

    @staticmethod
    def _total_tokens(response) -> int:
        # In MAF i token sono disponibili in usage_details.
        usage = getattr(response, "usage_details", None)

        if (
            usage is None
            or usage.input_token_count is None
            or usage.output_token_count is None
        ):
            raise ValueError("Missing LLM token usage details")

        return (
            usage.input_token_count
            + usage.output_token_count
        )

    async def _generate_correction(
        self,
        specification: str,
        code: str,
        error_message: str,
    ):
        # Costruzione della richiesta.
        prompt = self._build_prompt(
            specification,
            code,
            error_message,
        )

        # Aggiungiamo la richiesta alla cronologia.
        self._debug_chat.append(
            LLMMessage(role="user", contents=[prompt])
        )

        # Chiamata al modello tramite l'API MAF.
        response = await self._model_client.get_response(
            self._debug_chat
        )

        if not isinstance(response.text, str):
            raise TypeError(
                "Debugger expected a text response"
            )

        # Conserviamo anche la risposta del modello.
        self._debug_chat.append(
            LLMMessage(
                role="assistant",
                contents=[response.text],
            )
        )

        return response

    @handler
    async def handle_debug_code_message(
        self,
        message: DebugMessage,
        ctx: WorkflowContext[DebugMessage],
    ) -> None:
        """
        Gestisce il debugging nel normale funzionamento
        della pipeline.
        """

        self._counter += 1

        print_green(
            f"{self.id} received message. Starting to debug code. "
            f"Attempt {self._counter}"
        )

        # Richiediamo al modello di correggere il codice.
        response = await self._generate_correction(
            message.specification,
            message.code,
            message.error_message,
        )

        # Mostriamo la risposta nella console.
        dialogue(response.text, self._role)

        if os.getenv("UI") == "True":
            self._server.send_chunk(
                response.text,
                "debugger",
            )

        # Estraiamo il codice Python in formato Markdown.
        code_block = self._extract_code(response.text)

        # MAF: inviamo la risposta al Test Executor.
        await ctx.send_message(
            DebugMessage(
                message.specification,
                code_block,
                "",
            ),
            target_id="test_executor",
        )

    @handler
    async def handle_test_debug_code_message(
        self,
        message: TestDebugMessage,
        ctx: WorkflowContext[TestDebugResult],
    ) -> None:
        """
        Gestisce il debugging sperimentale.

        Mantiene il conteggio dei token e il comportamento
        originale in caso di eccezioni del modello.
        """

        # Un nuovo elemento del dataset deve iniziare
        # senza la cronologia dell'elemento precedente.
        if message.new_chat:
            self._counter = 0
            self._debug_chat = list(self._system_messages)
        else:
            self._counter += 1

        print_green(
            f"{self.id} received message. Attempt {self._counter}"
        )

        print_purple(str(message))

        try:
            # Correzione tramite LLM.
            response = await self._generate_correction(
                message.specification,
                message.code,
                message.error_message,
            )

            # Registriamo i token consumati.
            tokens = self._total_tokens(response)

            # Estraiamo il blocco di codice Python.
            code_block = self._extract_code(response.text)

            result = TestDebugResult(
                code_block,
                tokens,
            )

        except Exception:
            # Stesso comportamento dell'originale:
            # risposta vuota e valore sentinella.
            result = TestDebugResult(
                "",
                1048575,
            )

        # La risposta sperimentale viene inviata
        # direttamente al Test Executor.
        await ctx.send_message(
            result,
            target_id="test_executor",
        )
