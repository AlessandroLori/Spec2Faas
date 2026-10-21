import asyncio
import logging
import os
import sys

from agent_framework import WorkflowBuilder

from app.Utils import get_config_data, load_env_variables, set_logging_config
from app.main import ROOT, _close_model_clients, create_model_clients
from app.Utils_database import ChromaRAGMemory
from app.agents.Assistant import Assistant
from app.agents.FaaSDeployer import FaasDeployer
from app.agents.coding_agents.Coder_RAG import Coder_RAG
from app.agents.coding_agents.Debugger import Debugger
from app.agents.coding_agents.EntryPoint_RAG import EntryPoint_RAG
from app.agents.coding_agents.TestDesigner_RAG import TestDesigner_RAG
from app.agents.coding_agents.TestExecutor import TestExecutor
from app.agents.coding_agents.messages.MessagesTypes import Message
from app.agents.coding_agents.utils.DockerCodeExecutor import DockerCodeExecutor
from app.agents.coding_agents.utils.Utils import dialogue


# Same seven-agent graph as the standard pipeline, with the three RAG variants.
# The Chroma memory is injected to allow deterministic tests without downloading embeddings.
def build_workflow_rag(llm, models, code_executor, memory, server=None):
    assistant = Assistant(llm=llm["assistant"], model_client=models["assistant"], server=server)
    entry_point = EntryPoint_RAG(
        llm=llm["entry_point"], model_client=models["entry_point"],
        memory=memory, server=server,
    )
    coder = Coder_RAG(llm=llm["coder"], model_client=models["coder"], server=server)
    test_designer = TestDesigner_RAG(
        llm=llm["test_designer"], model_client=models["test_designer"], server=server
    )
    test_executor = TestExecutor(
        llm=llm["test_executor"], model_client=models["test_executor"],
        code_executor=code_executor, server=server,
    )
    debugger = Debugger(llm=llm["debugger"], model_client=models["debugger"], server=server)
    faas_deployer = FaasDeployer(
        llm=llm["faas_deployer"], model_client=models["faas_deployer"], server=server
    )

    return (
        WorkflowBuilder(start_executor=assistant, max_iterations=200)
        .add_edge(assistant, entry_point)
        .add_edge(entry_point, assistant)
        .add_edge(entry_point, coder)
        .add_edge(coder, test_executor)
        .add_edge(test_executor, entry_point)
        .add_edge(entry_point, test_designer)
        .add_edge(test_designer, test_executor)
        .add_edge(test_executor, debugger)
        .add_edge(debugger, test_executor)
        .add_edge(assistant, faas_deployer)
        .add_edge(faas_deployer, assistant)
        .build()
    )


async def main(llm, server, user_text):
    models = create_model_clients(llm)
    executor = DockerCodeExecutor()
    try:
        # Real Chroma memory uses PERSISTENCE_PATH and all-MiniLM-L6-v2.
        memory = ChromaRAGMemory()
        try:
            await executor.start()
            workflow = build_workflow_rag(llm, models, executor, memory, server)

            if os.getenv("UI", "False") == "False":
                dialogue(
                    "Hi! Write here your function to deploy or the specification of "
                    "the function you want to write.\nPress ENTER two times to continue.",
                    "Assistant",
                )
                while True:
                    lines = []
                    for line in sys.stdin:
                        if not line.strip():
                            break
                        lines.append(line)
                    if not lines:
                        return
                    result = await workflow.run(Message("".join(lines), "request"))
                    outputs = result.get_outputs()
                    if not outputs:
                        raise RuntimeError("The RAG workflow produced no output")
                    if outputs[-1].type != "request":
                        return
                    dialogue(
                        "The input is not clear. Please provide more information "
                        "about the function you want to deploy.",
                        "Assistant",
                    )
            else:
                result = await workflow.run(Message(user_text, "request"))
                outputs = result.get_outputs()
                if not outputs:
                    raise RuntimeError("The RAG workflow produced no output")
                if outputs[-1].type == "request":
                    dialogue(
                        "The input is not clear. Please provide more information "
                        "about the function you want to deploy.",
                        "Assistant",
                    )
        finally:
            try:
                await executor.stop()
            finally:
                await memory.close()
    finally:
        await _close_model_clients(models)


if __name__ == "__main__":
    os.environ["UI"] = "False"
    load_env_variables()
    log_path = os.getenv("LOG_PATH")
    if log_path:
        set_logging_config(log_path)
    else:
        logging.basicConfig(level=logging.WARNING)
    llm = get_config_data(str(ROOT / "config.yaml"))
    try:
        asyncio.run(main(llm, None, ""))
    finally:
        logging.shutdown()
