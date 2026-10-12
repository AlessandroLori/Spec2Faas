
import time

from agent_framework import Executor, Message as LLMMessage, WorkflowContext, handler

from experiments.MessageTypesTest import TestSystemMessage
from .messages.MessagesTypes import CodeMessage, Message
from .utils.Utils import dialogue, print_green, print_purple


class EntryPoint(Executor):
    SYSTEM_PROMPT = (
        "Given the function specification create a signature for the function. "
        "Return only the signature of the function as output."
    )

    def __init__(self, llm: str, model_client, server=None) -> None:
        super().__init__(id="entry_point")
        self._model_client = model_client
        self._llm = llm
        self._server = server

        # In MAF send_message non restituisce direttamente la risposta
        # del destinatario. Conserviamo la fase della richiesta corrente.
        self._normal_phase = "idle"
        self._normal_code_message = None
        self._system_phase = "idle"

        print_green(
            f"Hi I'm the entry point of the coding system and I use {self._llm}."
        )

    async def _generate_signature(self, specification: str):
        messages = [
            LLMMessage(role="system", contents=[self.SYSTEM_PROMPT]),
            LLMMessage(role="user", contents=[specification]),
        ]

        response = await self._model_client.get_response(messages)

        if not isinstance(response.text, str):
            raise TypeError(
                "EntryPoint expected a text response from the LLM."
            )

        return response

    @staticmethod
    def _total_tokens(response) -> int:
        usage = getattr(response, "usage_details", None)

        if (
            usage is None
            or usage.input_token_count is None
            or usage.output_token_count is None
        ):
            raise RuntimeError(
                "The MAF client did not return token usage."
            )

        return (
            usage.input_token_count
            + usage.output_token_count
        )

    @handler
    async def handle_assistant_message(
        self,
        message: Message,
        ctx: WorkflowContext[CodeMessage, Message],
    ) -> None:

        # Gestione della risposta ricevuta dal Test Executor.
        if message.type == "test_executor_response":

            # Il codice è pronto, ma mancano ancora i test.
            if message.content == "Still waiting":
                if self._normal_phase != "coder":
                    raise RuntimeError(
                        "Unexpected 'Still waiting' response."
                    )

                self._normal_phase = "designer"

                await ctx.send_message(
                    self._normal_code_message,
                    target_id="test_designer",
                )

            # Il Test Executor ha restituito il risultato finale.
            else:
                print_green(
                    f"{self.id} received message. "
                    "Communicating final outcome."
                )

                self._normal_phase = "idle"
                self._normal_code_message = None

                await ctx.yield_output(message)

            return

        if self._normal_phase != "idle":
            raise RuntimeError(
                "EntryPoint already has a normal request in progress."
            )

        print_green(
            f"{self.id} received message. "
            "Activating Coder and Test Designer."
        )

        # Stessa richiesta LLM del codice AutoGen originale.
        response = await self._generate_signature(message.content)

        dialogue(
            "The function signature is: " + response.text,
            "Entry Point",
        )

        # Conserviamo la specifica e la firma per il Test Designer.
        self._normal_code_message = CodeMessage(
            message.content,
            response.text,
        )

        # Prima attiviamo soltanto il Coder.
        self._normal_phase = "coder"

        await ctx.send_message(
            self._normal_code_message,
            target_id="coder",
        )

    @handler
    async def handle_test_system_assistant_message(
        self,
        message: TestSystemMessage,
        ctx: WorkflowContext[
            TestSystemMessage, TestSystemMessage
        ],
    ) -> None:

        # Risultato finale dell'esperimento.
        if message.type == "test_executor_response":
            print_green(
                f"{self.id} received message. "
                "Communicating final outcome."
            )
            print_purple(str(message))

            self._system_phase = "idle"
            await ctx.yield_output(message)
            return

        # Dopo il Coder, attiviamo il Test Designer.
        if self._system_phase == "coder":
            self._system_phase = "designer"

            await ctx.send_message(
                TestSystemMessage(
                    tokens=message.tokens,
                    time=message.time,
                    messages=message.messages + 1,
                    prompt=message.prompt,
                    signature=message.signature,
                    original_func=message.original_func,
                    code=message.code,
                ),
                target_id="test_designer",
            )
            return

        if self._system_phase != "idle":
            raise RuntimeError(
                "Unexpected experimental message in EntryPoint."
            )

        print_green(
            f"{self.id} received message. "
            "Activating Coder and Test Designer."
        )
        print_purple(str(message))

        # Misuriamo tempo e token della generazione della firma.
        start_time = time.perf_counter()

        response = await self._generate_signature(message.prompt)

        elapsed = time.perf_counter() - start_time
        tokens = self._total_tokens(response)

        message.time["entry_point"] = elapsed
        message.tokens["entry_point"] = tokens

        # Conserviamo la sequenza originale: prima Coder.
        self._system_phase = "coder"

        await ctx.send_message(
            TestSystemMessage(
                tokens=message.tokens,
                time=message.time,
                messages=message.messages + 1,
                prompt=message.prompt,
                signature=response.text,
            ),
            target_id="coder",
        )
