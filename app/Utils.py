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
        # Caso: provider MAF restituisce un dizionario.
        input_tokens = usage.get("input_token_count")
        output_tokens = usage.get("output_token_count")

    elif usage is not None:
        # Caso: oggetto con attributi.
        input_tokens = getattr(
            usage, "input_token_count", None
        )
        output_tokens = getattr(
            usage, "output_token_count", None
        )

    else:
        # Nessuna informazione disponibile.
        raise error_type(error_message)

    if input_tokens is None or output_tokens is None:
        raise error_type(error_message)

    return input_tokens + output_tokens

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

