from pathlib import Path
import logging
import os
import re

import yaml
import json
import uuid
from typing import List
from collections.abc import Mapping

# Logger utilizzato per la pipeline Microsoft Agent Framework.
TRACE_LOGGER_NAME = "agent_framework"

def total_tokens_from_response(
    response,
    *,
    error_message: str = "Missing MAF token usage details",
    error_type: type[Exception] = RuntimeError,
) -> int:
    """
    Calcola i token utilizzati da una risposta MAF.

    Supporta sia usage_details come dizionario
    (restituito da Ollama reale), sia come oggetto
    con attributi (utilizzato anche nei test esistenti).

    Mantiene il criterio originale di Spec2FaaS:
    token totali = token di input + token di output.
    """

    usage = getattr(response, "usage_details", None)

    if isinstance(usage, Mapping):
        # Ollama/MAF puo' restituire un dizionario.
        input_tokens = usage.get("input_token_count")
        output_tokens = usage.get("output_token_count")
        total_tokens = usage.get("total_token_count")
        available_fields = sorted(str(key) for key in usage.keys())

    elif usage is not None:
        # Altri provider/test usano un oggetto con attributi.
        input_tokens = getattr(usage, "input_token_count", None)
        output_tokens = getattr(usage, "output_token_count", None)
        total_tokens = getattr(usage, "total_token_count", None)
        known_fields = (
            "input_token_count", "output_token_count", "total_token_count"
        )
        available_fields = [
            name for name in known_fields if getattr(usage, name, None) is not None
        ]

    else:
        # Nessuna statistica: non inventiamo un conteggio.
        raise error_type(f"{error_message} (usage_details=None)")

    # Manteniamo la definizione del progetto originale: input + output.
    if input_tokens is not None and output_tokens is not None:
        return input_tokens + output_tokens

    # Se manca una componente ma MAF fornisce il TOTALE ufficiale,
    # possiamo usarlo direttamente senza stimare o inventare token.
    if total_tokens is not None:
        return total_tokens

    # Non mostriamo mai testo/prompt/credenziali: solo i nomi dei campi.
    raise error_type(
        f"{error_message} "
        f"(usage_type={type(usage).__name__}, "
        f"available_fields={available_fields})"
    )

def load_env_variables():

    try:
        # load environment variables from .env file (requires `pip install python-dotenv`)
        from dotenv import load_dotenv

        env_path = Path(__file__).resolve().parents[1] / ".env"

        load_dotenv(dotenv_path=env_path)

    except ImportError:
        pass


def set_logging_config(log_path):

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(TRACE_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    # File's handler
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    # Forcing reconfiguration of logging
    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler], force=True)

    logging.getLogger("agent_framework").setLevel(logging.DEBUG)
    logging.getLogger(TRACE_LOGGER_NAME).setLevel(logging.DEBUG)

    logging.getLogger(TRACE_LOGGER_NAME).propagate = True


def get_config_data(path):

    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        return config['llm']


def get_config_data_full(path):

    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        return config


def save_yaml(path, data):
    with open(path, 'w') as file:
        yaml.dump(data, file)

