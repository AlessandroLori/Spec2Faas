import os
import time
from typing import Never

from agent_framework import Executor, Message as LLMMessage, WorkflowContext, handler

from app.Utils import total_tokens_from_response
from experiments.MessageTypesTest import TestCodeMessage, TestCodeResult, TestSystemMessage
from .messages.MessagesTypes import CodeMessage, ExecuteMessage
from .utils.Code_Extractors import extract_markdown_code_blocks
from .utils.Utils import dialogue, print_green, print_purple


class Coder_RAG(Executor):
    SYSTEM_PROMPT = 'You are a very skilled software programmer.<TASK>As a programmer, you are required to code a function that adheres to the function specification.Make sure to use the function signature provided.<CODE FORMATTING>Please write code in this format:```python[Code]```</CODE FORMATTING><INSTRUCTIONS>1. Make sure you understand the task.2. Decide on the most efficient way to solve the task.3. Use the function signature provided.4. Make sure your code is correct and complete.5. If you find useful pseudocode in memory, use it.6. RETURN ONLY THE CODE OF THE FUNCTION IN THE SPECIFIED FORMAT</INSTRUCTIONS>'

    def __init__(self, llm: str, model_client=None, server=None) -> None:
        super().__init__(id="coder")
        self._model_client = model_client
        self._llm = llm
        self._server = server
        self._role = "Software Programmer"
        self._system_messages = [LLMMessage(role="system", contents=[self.SYSTEM_PROMPT])]
        print_green(f"Hi I'm the software programmer and I use {self._llm}.")

    @staticmethod
    def _total_tokens(response) -> int:
        return total_tokens_from_response(response)

    @staticmethod
    def _prompt(specification: str, signature: str) -> str:
        return "Function specification: " + specification + "\nFunction signature: " + signature

    @handler
    async def handle_generate_code_message(
        self, message: CodeMessage, ctx: WorkflowContext[ExecuteMessage]
    ) -> None:
        print_green(f"{self.id} received message. Staring to generate code with {self._llm}.")
        response = await self._model_client.get_response([
            *self._system_messages,
            LLMMessage(role="system", contents=[message.memory_context]),
            LLMMessage(role="user", contents=[self._prompt(message.specification, message.function_signature)]),
        ])
        if not isinstance(response.text, str):
            raise TypeError("Coder_RAG expected a text response")
        dialogue(response.text, self._role)
        if os.getenv("UI") == "True":
            self._server.send_chunk(response.text, "coder")
        await ctx.send_message(
            ExecuteMessage(message.specification, response.text, "", self.id),
            target_id="test_executor",
        )

    @handler
    async def handle_generate_test_code_message(
        self, message: TestCodeMessage, ctx: WorkflowContext[Never, TestCodeResult]
    ) -> None:
        messages = [LLMMessage(role="user", contents=[self._prompt(message.specification, message.function_signature)])]
        if message.prompt:
            messages = [*self._system_messages, *messages]
        start = time.perf_counter()
        response = await self._model_client.get_response(messages)
        elapsed = time.perf_counter() - start
        if not isinstance(response.text, str):
            raise TypeError("Coder_RAG expected a text response")
        await ctx.yield_output(TestCodeResult(response.text, elapsed, self._total_tokens(response), None))

    @handler
    async def handle_test_system_generate_code_message(
        self, message: TestSystemMessage, ctx: WorkflowContext[TestSystemMessage]
    ) -> None:
        print_green(f"{self.id} received message. Staring to generate code with {self._llm}.")
        print_purple(str(message))
        start = time.perf_counter()
        response = await self._model_client.get_response([
            *self._system_messages,
            LLMMessage(role="user", contents=[self._prompt(message.prompt, message.signature)]),
        ])
        elapsed = time.perf_counter() - start
        if not isinstance(response.text, str):
            raise TypeError("Coder_RAG expected a text response")
        message.time["coder"] = elapsed
        message.tokens["coder"] = self._total_tokens(response)
        blocks = extract_markdown_code_blocks(response.text)
        original_function = blocks[0].code if blocks else response.text
        await ctx.send_message(
            TestSystemMessage(
                tokens=message.tokens, time=message.time,
                messages=message.messages + 1,
                prompt=message.prompt, signature=message.signature,
                original_func=original_function, code=response.text,
                new_chat=True, sender=self.id,
            ),
            target_id="test_executor",
        )
