import asyncio
import os

from agent_framework.ollama import OllamaChatClient

from app.agents.coding_agents.EntryPoint_RAG import EntryPoint_RAG
from app.agents.coding_agents.messages.MessagesTypes import Message
from app.Utils_database import ChromaRAGMemory


class RecordingContext:
    def __init__(self):
        self.sent = []

    async def send_message(self, message, target_id=None):
        self.sent.append((target_id, message))


async def main():
    # Run only the RAG Entry Point with a local LLM; NEVER invoke Serverledge.
    from app.Utils import load_env_variables
    load_env_variables()
    memory = ChromaRAGMemory()
    client = OllamaChatClient(
        model="qwen2.5:1.5b", host="http://127.0.0.1:11434"
    )
    agent = EntryPoint_RAG("qwen2.5:1.5b", client, memory=memory)
    ctx = RecordingContext()
    try:
        await agent.handle_assistant_message(
            Message("ALGORITHM BinarySearch(array, target)", "request"),
            ctx,
        )
        destination, generated = ctx.sent[-1]
        assert destination == "coder"
        assert generated.function_signature.strip()
        assert "ALGORITHM BinarySearch" in generated.memory_context, (
            "ERRORE: il contesto BinarySearch non è stato recuperato"
        )

        print("Signature:", generated.function_signature)
        print("Retrieved context:", repr(generated.memory_context[:500]))
        print("PASS - RAG Entry Point with local Ollama")
    finally:
        await memory.close()
        close = getattr(client, "close", None)
        if callable(close):
            result = close()
            if asyncio.iscoroutine(result):
                await result


if __name__ == "__main__":
    asyncio.run(main())
