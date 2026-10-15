from pathlib import Path
import logging
import os
import re

import yaml
import json
import uuid
from typing import List

# Logger utilizzato per la pipeline Microsoft Agent Framework.
TRACE_LOGGER_NAME = "agent_framework"


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

