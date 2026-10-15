import os
import re
import time
from typing import Never

from agent_framework import (
    Executor,
    Message as LLMMessage,
    WorkflowContext,
    handler,
)

from .messages.MessagesTypes import CodeMessage, ExecuteMessage
from .utils.Utils import dialogue, print_green, print_purple

from experiments.MessageTypesTest import (
    TestCodeMessage,
    TestCodeResult,
    TestSystemMessage,
)
from app.Utils import total_tokens_from_response


# Stesso pattern dell'estrattore originale di Spec2FaaS.
# Qui è locale per non importare ancora CodeBlock di AutoGen.
_CODE_BLOCK = re.compile(
    r"```(?:\s*([\w\+\-]+))?\n([\s\S]*?)```"
)


class Coder(Executor):
    """
    Genera funzioni per la pipeline ordinaria
    e per i due percorsi sperimentali.
    """

    SYSTEM_PROMPT = (
        "You are a very skilled software programmer."
        "<TASK>"
        "As a programmer, you are required to code a function that adheres to the function specification."
        "Make sure to use the function signature provided."
        "<CODE FORMATTING>"
        "Please write code in this format:"
        "```python"
        "[Code]"
        "```"
        "</CODE FORMATTING>"
        "<INSTRUCTIONS>"
        "1. Make sure you understand the task."
        "2. Decide on the most efficient way to solve the task."
        "3. Use the function signature provided."
        "3. Make sure your code is correct and complete."
        "4. RETURN ONLY THE CODE OF THE FUNCTION IN THE SPECIFIED FORMAT"
        "</INSTRUCTIONS>"
    )

    def __init__(
        self,
        llm: str,
        model_client=None,
        server=None,
    ) -> None:

        # Identificativo dell'Executor nel workflow MAF.
        super().__init__(id="coder")

        self._model_client = model_client
        self._llm = llm
        self._text = ""
        self._role = "Software Programmer"
        self._client = None
        self._server = server

        print_green(
            f"Hi I'm the software programmer and I use {self._llm}."
        )

    async def _generate(
        self,
        specification: str,
        signature: str,
        *,
        with_system: bool = True,
    ):
        """
        Esegue una singola chiamata al modello.

        Preserva la costruzione dei prompt AutoGen.
        """
        prompt = (
            "Function specification: "
            + specification
            + "\nFunction signature: "
            + signature
        )

        messages = []

        if with_system:
            messages.append(
                LLMMessage(
                    role="system",
                    contents=[self.SYSTEM_PROMPT],
                )
            )

        messages.append(
            LLMMessage(
                role="user",
                contents=[prompt],
            )
        )

        # Equivalente MAF della chiamata create() di AutoGen.
        return await self._model_client.get_response(messages)

    @staticmethod
    def _total_tokens(response) -> int:
        """
        Recupera i token consumati dalla risposta MAF.
        """
        return total_tokens_from_response(
            response,
            error_message=(
                "Il provider MAF non ha restituito "
                "il conteggio dei token."
            ),
        )

    @handler
    async def handle_generate_code_message(
        self,
        message: CodeMessage,
        ctx: WorkflowContext[ExecuteMessage],
    ) -> None:
        """
        Generazione ordinaria del codice.
        """

        print_green(
            f"{self.id} received message. "
            f"Staring to generate code with {self._llm}."
        )

        response = await self._generate(
            message.specification,
            message.function_signature,
        )

        assert isinstance(response.text, str)

        dialogue(response.text, self._role)

        if os.getenv("UI") == "True":
            self._server.send_chunk(response.text, "coder")

        # Il routing verrà gestito dal workflow MAF.
        await ctx.send_message(
            ExecuteMessage(
                message.specification,
                response.text,
                "",
                self.id,
            )
        )

    @handler
    async def handle_generate_test_code_message(
        self,
        message: TestCodeMessage,
        ctx: WorkflowContext[Never, TestCodeResult],
    ) -> None:
        """
        Valutazione isolata del Coder.
        """

        start_time = time.perf_counter()

        response = await self._generate(
            message.specification,
            message.function_signature,
            with_system=message.prompt,
        )

        execution_time = time.perf_counter() - start_time

        total_tokens = self._total_tokens(response)

        assert isinstance(response.text, str)

        # L'adapter della cancellazione Docker sarà
        # completato quando migreremo TestExecutor.
        await ctx.yield_output(
            TestCodeResult(
                response.text,
                execution_time,
                total_tokens,
                None,
            )
        )

    @handler
    async def handle_test_system_generate_code_message(
        self,
        message: TestSystemMessage,
        ctx: WorkflowContext[TestSystemMessage],
    ) -> None:
        """
        Generazione del codice nella valutazione end-to-end.
        """

        print_green(
            f"{self.id} received message. "
            f"Staring to generate code with {self._llm}."
        )
        print_purple(str(message))

        total_time = message.time
        total_tokens = message.tokens

        start_time = time.perf_counter()

        response = await self._generate(
            message.prompt,
            message.signature,
        )

        elapsed = time.perf_counter() - start_time
        tokens = self._total_tokens(response)

        total_time["coder"] = elapsed
        total_tokens["coder"] = tokens

        assert isinstance(response.text, str)

        # Estrazione identica alla logica originale.
        match = _CODE_BLOCK.search(response.text)

        if match:
            original_function = match.group(2)
        else:
            original_function = response.text

        await ctx.send_message(
            TestSystemMessage(
                tokens=total_tokens,
                time=total_time,
                messages=message.messages + 1,
                prompt=message.prompt,
                signature=message.signature,
                original_func=original_function,
                code=response.text,
                new_chat=True,
                sender=self.id,
            )
        )
