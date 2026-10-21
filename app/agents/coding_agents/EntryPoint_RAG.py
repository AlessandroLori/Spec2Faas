import os

from agent_framework import Executor, Message as LLMMessage, WorkflowContext, handler

from .messages.MessagesTypes import CodeMessage, Message
from .utils.Utils import dialogue, print_green


class EntryPoint_RAG(Executor):
    SYSTEM_PROMPT = 'Given the function specification and the memory context create a signature for the function.Do not implement the function.Return only the signature of the function as output.'

    def __init__(self, llm: str, model_client, server=None, memory=None) -> None:
        super().__init__(id="entry_point")
        if memory is None:
            raise ValueError("EntryPoint_RAG requires a ChromaDB memory instance")
        self._model_client = model_client
        self._llm = llm
        self._server = server
        self._memory = memory
        self._pending_code_message = None
        self._system_messages = [LLMMessage(role="system", contents=[self.SYSTEM_PROMPT])]
        print_green(f"Hi I'm the entry point of the coding system and I use {self._llm}.")

    @handler
    async def handle_assistant_message(
        self, message: Message, ctx: WorkflowContext[CodeMessage | Message]
    ) -> None:
        # The Test Executor first asks us to activate Test Designer; then it
        # returns the final generated function (or FAIL) to the Assistant.
        if message.type == "test_executor_response":
            if message.content == "Still waiting":
                if self._pending_code_message is None:
                    raise RuntimeError("Missing RAG CodeMessage for Test Designer")
                await ctx.send_message(
                    self._pending_code_message, target_id="test_designer"
                )
                return

            self._pending_code_message = None
            print_green(f"{self.id} received message. Communicating final outcome.")
            await ctx.send_message(message, target_id="assistant")
            return

        print_green(f"{self.id} received message. Activating Coder and Test Designer.")
        retrieved = await self._memory.query(message.content)
        retrieved_content = "\n\n".join(retrieved)
        if retrieved:
            rag_context = (
                "\n\n<CONTEXT FROM MEMORY>\n"
                "The following useful pseudocode was retrieved from memory:\n"
                f"{retrieved_content}"
                "\n</CONTEXT FROM MEMORY>\n"
            )
            memory_content = (
                "The following useful code pseudocode were retrieved from memory:\n"
                + retrieved_content + "\n\n"
            )
        else:
            rag_context = ""
            memory_content = "No content was retrieved from memory.\n\n"

        response = await self._model_client.get_response([
            *self._system_messages,
            LLMMessage(role="system", contents=[rag_context]),
            LLMMessage(role="user", contents=[message.content]),
        ])
        if not isinstance(response.text, str):
            raise TypeError("EntryPoint_RAG expected a text response")

        announcement = (
            memory_content
            + "\n\n Asking to generate code and test for the function "
            + response.text
        )
        dialogue(announcement, "Entry Point")
        if os.getenv("UI") == "True":
            self._server.send_chunk(announcement, "entry_point")

        self._pending_code_message = CodeMessage(
            message.content, response.text, rag_context
        )
        await ctx.send_message(self._pending_code_message, target_id="coder")
