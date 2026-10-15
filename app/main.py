
import asyncio
import inspect
import logging
import os
import sys
from pathlib import Path

from agent_framework import WorkflowBuilder
from agent_framework.ollama import OllamaChatClient
from agent_framework.openai import OpenAIChatCompletionClient
from openai import AsyncOpenAI

from app.Utils import (
    get_config_data,
    load_env_variables,
    set_logging_config,
)
from app.agents.Assistant import Assistant
from app.agents.FaaSDeployer import FaasDeployer
from app.agents.coding_agents.Coder import Coder
from app.agents.coding_agents.Debugger import Debugger
from app.agents.coding_agents.EntryPoint import EntryPoint
from app.agents.coding_agents.TestDesigner import TestDesigner
from app.agents.coding_agents.TestExecutor import TestExecutor
from app.agents.coding_agents.messages.MessagesTypes import Message
from app.agents.coding_agents.utils.DockerCodeExecutor import DockerCodeExecutor
from app.agents.coding_agents.utils.Utils import dialogue


# Punto d'ingresso della pipeline standard non-RAG.
# I modelli e Docker sono creati qui, mentre gli agenti
# restano definiti nei rispettivi file originali.
ROOT = Path(__file__).resolve().parent.parent


def create_model_clients(llm: dict[str, str]):
    """
    Crea i client MAF per tutti i ruoli configurati.

    Gemini utilizza l'endpoint compatibile OpenAI.
    Gli altri modelli mantengono il provider Ollama originale.
    """
    models = {}

    for role, model_name in llm.items():

        if model_name in (
            "gemini-2.5-pro",
            "gemini-2.0-flash",
        ):
            # Manteniamo endpoint e numero di retry
            # configurati nel main AutoGen originale.
            sdk_client = AsyncOpenAI(
                api_key=os.environ["GEMINI_API_KEY"],
                base_url=(
                    "https://generativelanguage.googleapis.com/"
                    "v1beta/openai/"
                ),
                max_retries=10,
            )

            models[role] = OpenAIChatCompletionClient(
                model=model_name,
                async_client=sdk_client,
            )

        else:
            # Indirizzo Ollama originale, eventualmente
            # sovrascrivibile tramite variabile d'ambiente.
            host = os.getenv(
                "OLLAMA_HOST",
                "http://160.80.97.151:11434",
            )

            if "://" not in host:
                host = "http://" + host

            models[role] = OllamaChatClient(
                model=model_name,
                host=host,
            )

    return models


def build_workflow(llm, models, code_executor, server=None):
    """
    Costruisce il grafo MAF usando i sette agenti originali.

    Questa funzione è separata dalla creazione dei modelli
    per permettere test deterministici senza credenziali LLM.
    """

    # Assistant: interpreta la richiesta iniziale.
    assistant = Assistant(
        llm=llm["assistant"],
        model_client=models["assistant"],
        server=server,
    )

    # Entry Point: genera la firma e coordina il coding.
    entry_point = EntryPoint(
        llm=llm["entry_point"],
        model_client=models["entry_point"],
        server=server,
    )

    # Coder: genera il codice.
    coder = Coder(
        llm=llm["coder"],
        model_client=models["coder"],
        server=server,
    )

    # Test Designer: genera i test.
    test_designer = TestDesigner(
        llm=llm["test_designer"],
        model_client=models["test_designer"],
        server=server,
    )

    # Test Executor: usa il nostro esecutore Docker reale.
    test_executor = TestExecutor(
        llm=llm["test_executor"],
        model_client=models["test_executor"],
        code_executor=code_executor,
        server=server,
    )

    # Debugger: corregge il codice se i test falliscono.
    debugger = Debugger(
        llm=llm["debugger"],
        model_client=models["debugger"],
        server=server,
    )

    # FaaS Deployer: prepara il payload e contatta Serverledge.
    faas_deployer = FaasDeployer(
        llm=llm["faas_deployer"],
        model_client=models["faas_deployer"],
        server=server,
    )

    # Ogni edge rappresenta una comunicazione consentita
    # fra due executor MAF.
    #
    # Le comunicazioni bidirezionali sono necessarie perché
    # MAF non restituisce automaticamente una risposta quando
    # un agente usa ctx.send_message().
    workflow = (
        WorkflowBuilder(
            start_executor=assistant,
            max_iterations=200,
        )

        # Assistant <-> Entry Point.
        .add_edge(assistant, entry_point)
        .add_edge(entry_point, assistant)

        # Entry Point -> Coder -> Test Executor.
        .add_edge(entry_point, coder)
        .add_edge(coder, test_executor)

        # Risposta di attesa o risultato dal Test Executor.
        .add_edge(test_executor, entry_point)

        # Entry Point -> Test Designer -> Test Executor.
        .add_edge(entry_point, test_designer)
        .add_edge(test_designer, test_executor)

        # Ciclo di debugging.
        .add_edge(test_executor, debugger)
        .add_edge(debugger, test_executor)

        # Deployment diretto oppure dopo la generazione.
        .add_edge(assistant, faas_deployer)
        .add_edge(faas_deployer, assistant)

        .build()
    )

    return workflow


async def _close_model_clients(models):
    """
    Chiude le connessioni HTTP dei provider MAF.

    I client OpenAI e Ollama non espongono necessariamente
    il metodo di chiusura allo stesso livello.
    """

    for model in models.values():

        close = getattr(model, "close", None)

        if not callable(close):
            inner = getattr(model, "client", None)

            close = (
                getattr(inner, "aclose", None)
                or getattr(inner, "close", None)
            )

            if not callable(close):
                http = getattr(inner, "_client", None)
                close = getattr(http, "aclose", None)

        if callable(close):
            result = close()

            if inspect.isawaitable(result):
                await result


async def main(llm, server, user_text):
    """
    Avvia la pipeline non-RAG.

    Mantiene la modalità terminale e quella utilizzabile
    da un server applicativo, senza dipendere da AutoGen.
    """

    # Creiamo i client dei modelli indicati in config.yaml.
    models = create_model_clients(llm)

    # Utilizziamo l'adattatore Docker validato in precedenza.
    executor = DockerCodeExecutor()

    try:
        await executor.start()

        # Costruzione del workflow con tutti e sette gli agenti.
        workflow = build_workflow(
            llm,
            models,
            executor,
            server,
        )

        if os.getenv("UI", "False") == "False":

            # Modalità interattiva da terminale.
            dialogue(
                "Hi! Write here your function to deploy or "
                "the specification of the function you want to write.\n"
                "Press ENTER two times to continue.",
                "Assistant",
            )

            while True:
                lines = []

                # Leggiamo la richiesta fino alla riga vuota.
                for line in sys.stdin:
                    if not line.strip():
                        break
                    lines.append(line)

                if not lines:
                    return

                # L'Assistant è il primo executor del workflow.
                result = await workflow.run(
                    Message("".join(lines), "request")
                )

                outputs = result.get_outputs()

                if not outputs:
                    raise RuntimeError(
                        "Il workflow MAF non ha restituito "
                        "un risultato"
                    )

                response = outputs[-1]

                # Se la richiesta è stata chiarita e completata,
                # terminiamo come nell'originale.
                if response.type != "request":
                    return

                dialogue(
                    "The input is not clear. Please provide more "
                    "information about the function you want to deploy.",
                    "Assistant",
                )

        else:

            # Modalità chiamata da un server applicativo.
            result = await workflow.run(
                Message(user_text, "request")
            )

            outputs = result.get_outputs()

            if not outputs:
                raise RuntimeError(
                    "Il workflow MAF non ha restituito "
                    "un risultato"
                )

            if outputs[-1].type == "request":
                dialogue(
                    "The input is not clear. Please provide more "
                    "information about the function you want to deploy.",
                    "Assistant",
                )

    finally:

        # Anche in caso di errore rilasciamo Docker
        # e i client HTTP dei modelli.
        try:
            await executor.stop()
        finally:
            await _close_model_clients(models)


if __name__ == "__main__":

    # Manteniamo l'avvio da terminale come comportamento
    # predefinito del main originale.
    os.environ["UI"] = "False"

    load_env_variables()

    log_path = os.getenv("LOG_PATH")

    if log_path:
        set_logging_config(log_path)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Percorso assoluto rispetto alla repository:
    # non dipende dalla directory corrente del terminale.
    config = get_config_data(
        str(ROOT / "config.yaml")
    )

    try:
        asyncio.run(
            main(config, None, "")
        )
    finally:
        logging.shutdown()
