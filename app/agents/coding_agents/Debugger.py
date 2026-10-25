
import os
import re
import sys

from agent_framework import Executor, Message as LLMMessage, WorkflowContext, handler

from .messages.MessagesTypes import DebugMessage
from .utils.Utils import dialogue, print_green, print_purple
from experiments.MessageTypesTest import TestDebugMessage, TestDebugResult
from app.Utils import total_tokens_from_response


class DebuggerProviderError(RuntimeError):
    """Errore della richiesta al provider: non e' un fallimento del codice."""


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
        return total_tokens_from_response(
            response,
            error_message="Missing LLM token usage details",
            error_type=ValueError,
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

        # Il fallimento della chiamata al provider non equivale a una
        # correzione sbagliata: il runner deve potersi fermare e riprendere
        # senza salvare un falso risultato per questo task.
        try:
            response = await self._model_client.get_response(
                self._debug_chat
            )
        except Exception as exc:
            raise DebuggerProviderError(
                f"Richiesta al modello {self._llm} fallita "
                f"({type(exc).__name__}: {exc})"
            ) from exc

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

        Mantiene il messaggio di risposta e il formato originali.
        Separa errori di formato, statistiche token mancanti e
        fallimenti della richiesta al provider.
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

        # Il contratto dei messaggi e il formato del codice restano
        # invariati rispetto ad AutoGen: TestDebugResult(code, tokens).
        try:
            response = await self._generate_correction(
                message.specification,
                message.code,
                message.error_message,
            )
        except DebuggerProviderError as exc:
            # Un'eccezione di trasporto/provider NON prova che il codice
            # sia incorreggibile. Non produciamo una sentinella nel Parquet:
            # interrompiamo il run, che potra' riprendere dai task salvati.
            print(
                f"ERRORE Q3: richiesta al Debugger fallita; "
                f"task non registrato. {exc}",
                file=sys.stderr,
                flush=True,
            )
            raise
        except TypeError as exc:
            # Risposta del provider ricevuta ma senza testo utilizzabile:
            # il codice non e' stato generato nel formato richiesto.
            print(
                f"ATTENZIONE Q3: risposta testuale non valida "
                f"({type(exc).__name__}: {exc}); "
                f"uso la sentinella prevista da AutoGen.",
                file=sys.stderr,
                flush=True,
            )
            result = TestDebugResult("", 1048575)
        else:
            try:
                # La regex e il formato richiesto sono quelli originali.
                code_block = self._extract_code(response.text)
            except ValueError as exc:
                # La risposta non contiene codice Python nel formato
                # richiesto: AutoGen restituisce stringa vuota e sentinella.
                print(
                    f"ATTENZIONE Q3: blocco Python assente "
                    f"({type(exc).__name__}: {exc}); "
                    f"text_len={len(response.text)} "
                    f"finish_reason={getattr(response, 'finish_reason', None)!r}; "
                    f"uso la sentinella prevista da AutoGen.",
                    file=sys.stderr,
                    flush=True,
                )
                result = TestDebugResult("", 1048575)
            else:
                try:
                    tokens = self._total_tokens(response)
                except ValueError as exc:
                    if not str(exc).startswith(
                        "Missing LLM token usage details"
                    ):
                        raise
                    # Come gia' avviene nei Q1/Q2 MAF: una metrica
                    # mancante non invalida codice effettivamente ricevuto.
                    # NaN significa sconosciuto, NON zero token.
                    tokens = float("nan")
                    print(
                        f"ATTENZIONE Q3: codice valido ma conteggio token "
                        f"assente; proseguo con tokens=NaN. {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

                result = TestDebugResult(code_block, tokens)

        # La risposta sperimentale viene inviata
        # direttamente al Test Executor.
        await ctx.send_message(
            result,
            target_id="test_executor",
        )
