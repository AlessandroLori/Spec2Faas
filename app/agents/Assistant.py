
import os
import time
from typing import Never

from agent_framework import Executor, Message as LLMMessage, WorkflowContext, handler

from .coding_agents.messages.MessagesTypes import DeployMessage, FaasMessage, Message
from .coding_agents.utils.Utils import dialogue, print_green
from experiments.MessageTypesTest import (
    TestDeployMessage,
    TestDeployResult,
    TestMessage,
    TestMessageResult,
    TestSystemMessage,
)


class Assistant(Executor):
    # Le stringhe dei due system prompt sono identiche all'originale AutoGen.
    SYSTEM_PROMPT = (
        "You are the entry point to a code generator and deployment app."
        "<INSTRUCTIONS>"
        "Inspect the input received by the user:"
        "1.If it contains a specification of a function or it asks to code a function, translate it in English and return the translation. "
        "-Make the translation clear and direct, so that an AI assistant can easily generate code from it."
        "-Put the word 'translation' as incipit of the text. Return only one translation."
        "-If it is already in English don't translate, but still put translation as incipit of text."
        "-Be as precise as possible with the translation."
        "2. If it contains a function already complete return the word 'deployment' followed by the function specified by the user."
        "3. If the input does not fall in the previous two categories or the specification is unsure asks for a clarification."
        "-If the user continue to provide an input that is neither a function specification nor a function code continue to ask for clarification."
        "Be careful when inspecting user prompt."
        "</INSTRUCTIONS>"
    )

    SYSTEM_PROMPT_TEST = (
        "You are the entry point to a code generator and deployment app."
        "<INSTRUCTIONS>"
        "Inspect the input received by the user:"
        "1.If it contains a specification of a function or it asks to code a function, translate it in English and return the translation. "
        "-Make the translation clear and direct, so that an AI assistant can easily generate code from it."
        "-Return only one translation."
        "-If it is already in English don't translate."
        "-Be as precise as possible with the translation."
        "Be careful when inspecting user prompt."
        "</INSTRUCTIONS>"
    )

    def __init__(self, llm: str, model_client, server=None) -> None:
        super().__init__(id="assistant")

        self._model_client = model_client
        self._llm = llm
        self._role = "Assistant"
        self._server = server

        # I due prompt originali vengono convertiti in messaggi MAF.
        self._system_messages = [
            LLMMessage(
                role="system",
                contents=[self.SYSTEM_PROMPT],
            )
        ]

        self._system_messages_test = [
            LLMMessage(
                role="system",
                contents=[self.SYSTEM_PROMPT_TEST],
            )
        ]

        # AutoGen attendeva direttamente le risposte degli agenti.
        # In MAF registriamo la fase e riprendiamo alla risposta.
        self._normal_phase = None
        self._system_phase = None

        # Dati utilizzati per le metriche end-to-end.
        self._system_start = None
        self._system_result = None

        print_green(f"Hi I'm the assistant and I use {self._llm}.")

    def _ui(self, text: str, message_type: str) -> None:
        # Manteniamo l'invio dei messaggi al server UI.
        if os.getenv("UI") == "True":
            self._server.send_chunk(
                text,
                "assistant",
                message_type,
            )

    async def _completion(
        self,
        content: str,
        *,
        system_test: bool = False,
    ):
        # L'esperimento end-to-end utilizza un prompt diverso.
        system_messages = (
            self._system_messages_test
            if system_test
            else self._system_messages
        )

        response = await self._model_client.get_response([
            *system_messages,
            LLMMessage(
                role="user",
                contents=[content],
            ),
        ])

        if not isinstance(response.text, str):
            raise TypeError("Assistant expected a text response")

        return response

    @staticmethod
    def _tokens(response) -> int:
        # Recuperiamo i token dall'interfaccia MAF.
        usage = getattr(response, "usage_details", None)

        if (
            usage is None
            or usage.input_token_count is None
            or usage.output_token_count is None
        ):
            raise RuntimeError(
                "Assistant LLM response is missing token usage"
            )

        return (
            usage.input_token_count
            + usage.output_token_count
        )

    @handler
    async def handle_user_message(
        self,
        message: Message,
        ctx: WorkflowContext[DeployMessage | Message, Message],
    ) -> None:
        """
        Gestisce il percorso ordinario:
        traduzione, deployment e chiarimento.
        """

        # Se il messaggio arriva dall'Entry Point,
        # contiene il risultato della generazione.
        if message.type == "test_executor_response":

            if self._normal_phase != "waiting_entry_point":
                raise RuntimeError(
                    "Unexpected EntryPoint response"
                )

            # Il coding team non ha generato una funzione valida.
            if message.content == "FAIL":
                self._normal_phase = None

                self._ui(
                    "We couldn't generate a correct function given "
                    "the specification. Do you want to try again?",
                    "final",
                )

                await ctx.yield_output(
                    Message(
                        "We couldn't generate a correct function "
                        "given the specification.",
                        "failure",
                    )
                )
                return

            # La generazione è riuscita.
            # Passiamo al deployment.
            self._normal_phase = "waiting_deployer"

            await ctx.send_message(
                DeployMessage(code=message.content),
                target_id="faas_deployer",
            )
            return

        # Non iniziamo una nuova richiesta mentre
        # è ancora in elaborazione quella precedente.
        if self._normal_phase is not None:
            raise RuntimeError(
                "Assistant already processing a normal request"
            )

        print_green(
            f"{self.id} received message. "
            "Staring to analyze user's prompt."
        )

        response = await self._completion(message.content)

        # CASO 1: l'utente ha già fornito una funzione completa.
        if response.text.startswith("deployment"):

            dialogue(
                "I'll deploy the function you provided as input.",
                self._role,
            )

            self._ui(
                "I'll deploy the function you provided as input.",
                "text",
            )

            self._normal_phase = "waiting_deployer"

            await ctx.send_message(
                DeployMessage(
                    code=response.text.removeprefix("deployment:")
                ),
                target_id="faas_deployer",
            )

        # CASO 2: l'utente ha fornito una specifica.
        elif response.text.startswith("translation"):

            dialogue(response.text, self._role)

            self._ui(
                "Perfect! Starting code generation based on "
                "the provided specification.",
                "text",
            )

            self._normal_phase = "waiting_entry_point"

            await ctx.send_message(
                Message(
                    response.text.removeprefix("translation:"),
                    "request",
                ),
                target_id="entry_point",
            )

        # CASO 3: richiesta da chiarire.
        else:

            dialogue(response.text, self._role)

            self._ui("", "request")

            await ctx.yield_output(
                Message(
                    response.text,
                    "request",
                )
            )

    @handler
    async def handle_faas_message(
        self,
        message: FaasMessage,
        ctx: WorkflowContext[Never, Message],
    ) -> None:
        """
        Riceve dal FaaS Deployer il risultato
        del deployment ordinario.
        """

        if self._normal_phase != "waiting_deployer":
            raise RuntimeError(
                "Unexpected FaaS Deployer response"
            )

        # L'elaborazione normale è terminata.
        self._normal_phase = None

        if message.error_message != "":

            dialogue(
                "I'm sorry that the function couldn't be deployed.\n"
                "Make sure that the FaaS platform is up and running "
                "before trying again.",
                self._role,
            )

            self._ui(
                "I'm sorry that the function couldn't be deployed.\n"
                "Make sure that the FaaS platform is up and running "
                "before trying again. Do you want to try again?",
                "final",
            )

        else:

            dialogue(
                "Thanks for using Spec2FaaS. I hope to see you soon!",
                self._role,
            )

            self._ui(
                "Thanks for using Spec2FaaS. "
                "Do you have something else in mind?",
                "final",
            )

        # Risultato finale destinato al chiamante del workflow.
        await ctx.yield_output(
            Message(
                message.content,
                "deployment",
            )
        )

    @handler
    async def handle_test_user_message(
        self,
        message: TestMessage,
        ctx: WorkflowContext[Never, TestMessageResult],
    ) -> None:
        """
        Valutazione isolata dell'Assistant.

        Misura tempo e token consumati per classificare
        una singola richiesta.
        """

        start = time.perf_counter()

        # Nell'originale si usa il prompt normale.
        response = await self._completion(message.content)

        elapsed = time.perf_counter() - start

        if response.text.startswith("deployment"):
            category = "deployment"
        elif response.text.startswith("translation"):
            category = "translation"
        else:
            category = "other"

        await ctx.yield_output(
            TestMessageResult(
                content=response.text,
                time=elapsed,
                tokens=self._tokens(response),
                type=category,
            )
        )

    @handler
    async def handle_test_system_user_message(
        self,
        message: TestSystemMessage,
        ctx: WorkflowContext[
            TestSystemMessage | TestDeployMessage,
            TestSystemMessage,
        ],
    ) -> None:
        """
        Valutazione sperimentale end-to-end.

        Gestisce la richiesta iniziale e il ritorno
        del risultato dal coding team.
        """

        # RISPOSTA DEL CODING TEAM.
        if message.type == "test_executor_response":

            if self._system_phase != "waiting_entry_point":
                raise RuntimeError(
                    "Unexpected experimental EntryPoint response"
                )

            # Il tempo del sistema comprende tutta la generazione.
            message.time["system"] = (
                time.perf_counter() - self._system_start
            )

            # Se la funzione non è stata generata,
            # concludiamo senza chiamare il Deployer.
            if not message.generated:

                self._system_phase = None
                self._system_start = None

                await ctx.yield_output(message)
                return

            # La funzione è stata generata.
            # Salviamo il risultato per completare le metriche
            # dopo la risposta del Deployer.
            self._system_result = message
            self._system_phase = "waiting_deployer"

            await ctx.send_message(
                TestDeployMessage(
                    message.code_final_func
                ),
                target_id="faas_deployer",
            )
            return

        # RICHIESTA INIZIALE DELL'ESPERIMENTO.
        if self._system_phase is not None:
            raise RuntimeError(
                "Assistant already processing an experimental request"
            )

        print_green(
            f"{self.id} received message. "
            "Staring to analyze user's prompt."
        )

        self._system_start = time.perf_counter()

        # Usiamo il prompt sperimentale originale.
        response = await self._completion(
            message.prompt,
            system_test=True,
        )

        # Registriamo le metriche dell'Assistant.
        message.tokens["assistant"] = self._tokens(response)

        message.time["assistant"] = (
            time.perf_counter() - self._system_start
        )

        self._system_phase = "waiting_entry_point"

        # Avviamo il coding team attraverso MAF.
        await ctx.send_message(
            TestSystemMessage(
                tokens=message.tokens,
                time=message.time,
                messages=message.messages + 1,
                prompt=response.text,
            ),
            target_id="entry_point",
        )

    @handler
    async def handle_test_deploy_result(
        self,
        message: TestDeployResult,
        ctx: WorkflowContext[Never, TestSystemMessage],
    ) -> None:
        """
        Completa la valutazione sperimentale
        quando arriva il risultato del Deployer.
        """

        if (
            self._system_phase != "waiting_deployer"
            or self._system_result is None
        ):
            raise RuntimeError(
                "Unexpected experimental FaaS Deployer response"
            )

        result = self._system_result

        # Aggiorniamo le metriche ricevute dal Deployer.
        result.tokens["faas_deployer"] = message.tokens
        result.time["faas_deployer"] = message.time

        # Ripuliamo lo stato della richiesta.
        self._system_result = None
        self._system_start = None
        self._system_phase = None

        # Manteniamo i campi restituiti nell'Assistant originale.
        await ctx.yield_output(
            TestSystemMessage(
                tokens=result.tokens,
                time=result.time,
                messages=result.messages + 1,
                prompt=result.prompt,
                signature=result.signature,
                original_func=result.original_func,
                tests_str=result.tests_str,
                final_func=result.final_func,
                attempts=result.attempts,
                generated=result.generated,
                result_deployment=message.result,
                deployed=message.result != "FAIL",
                deployed_function=message.deployed_function,
            )
        )
