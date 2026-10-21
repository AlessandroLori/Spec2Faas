import os
import time
from typing import Never

from agent_framework import Executor, Message as LLMMessage, WorkflowContext, handler

from app.Utils import total_tokens_from_response
from experiments.MessageTypesTest import TestCodeMessage, TestCodeResult, TestSystemMessage
from .messages.MessagesTypes import CodeMessage, ExecuteMessage
from .utils.Code_Extractors import extract_markdown_code_blocks
from .utils.Utils import dialogue, print_green, print_purple


class TestDesigner_RAG(Executor):
    SYSTEM_PROMPT = "You are a very skilled test designer specializing in designing comprehensive and well-documented test cases for Python functions.Your goal is to generate exhaustive test cases for the given function specification, covering all relevant input scenarios and ensuring correctness, robustness, and edge coverage.<INSTRUCTIONS>First of all, make CORRECT tests.- Focus on correctness, clarity, and diversity of test inputs.- Each test case must:-- Follow the exact format below.-- Include a short, meaningful comment or description of what it tests.-- Be self-contained and syntactically valid Python.-- Be semantically correct with respect to the specification.- Only output test cases in the specified format Python format, nothing else.- After you have produced the tests take your time to reflect if each on of them is correct.- If you find the memory context useful, use it.</INSTRUCTIONS><OUTPUT FORMAT>The format of test cases should be:```pythonassert function_name(input) == expected_output, 'Test Case <number>: Description'```</OUTPUT FORMAT>"

    def __init__(self, llm: str, model_client, server=None) -> None:
        super().__init__(id="test_designer")
        self._model_client = model_client
        self._llm = llm
        self._server = server
        self._role = "Test Designer"
        self._system_messages = [LLMMessage(role="system", contents=[self.SYSTEM_PROMPT])]
        print_green(f"Hi I'm the test designer and I use {self._llm}.")

    @staticmethod
    def _total_tokens(response) -> int:
        return total_tokens_from_response(response)

    @staticmethod
    def _prompt(specification: str, signature: str) -> str:
        return (
            "Function specification: " + specification
            + "\nFunction signature: " + signature
            + "\nPlease generate the tests:"
        )

    @handler
    async def handle_generate_code_message(
        self, message: CodeMessage, ctx: WorkflowContext[ExecuteMessage]
    ) -> None:
        print_green(f"{self.id} received message. Creating now a test suite for the function.")
        response = await self._model_client.get_response([
            *self._system_messages,
            LLMMessage(role="system", contents=[message.memory_context]),
            LLMMessage(role="user", contents=[self._prompt(message.specification, message.function_signature)]),
        ])
        if not isinstance(response.text, str):
            raise TypeError("TestDesigner_RAG expected a text response")
        dialogue(response.text, self._role)
        if os.getenv("UI") == "True":
            self._server.send_chunk(response.text, "test_designer")
        await ctx.send_message(
            ExecuteMessage(message.specification, "", response.text, self.id),
            target_id="test_executor",
        )

    @handler
    async def handle_generate_test_code_message(
        self, message: TestCodeMessage, ctx: WorkflowContext[Never, TestCodeResult]
    ) -> None:
        print_green(f"{self.id} received message. Creating now a test suite for the function.")
        start = time.perf_counter()
        response = await self._model_client.get_response([
            *self._system_messages,
            LLMMessage(role="user", contents=[self._prompt(message.specification, message.function_signature)]),
        ])
        elapsed = time.perf_counter() - start
        if not isinstance(response.text, str):
            raise TypeError("TestDesigner_RAG expected a text response")
        await ctx.yield_output(TestCodeResult(response.text, elapsed, self._total_tokens(response), None))

    @handler
    async def handle_test_system_generate_code_message(
        self, message: TestSystemMessage, ctx: WorkflowContext[TestSystemMessage]
    ) -> None:
        print_green(f"{self.id} received message. Creating now a test suite for the function.")
        print_purple(str(message))
        start = time.perf_counter()
        response = await self._model_client.get_response([
            *self._system_messages,
            LLMMessage(role="user", contents=[self._prompt(message.prompt, message.signature)]),
        ])
        elapsed = time.perf_counter() - start
        if not isinstance(response.text, str):
            raise TypeError("TestDesigner_RAG expected a text response")
        message.time["test_designer"] = elapsed
        message.tokens["test_designer"] = self._total_tokens(response)
        blocks = extract_markdown_code_blocks(response.text)
        tests = blocks[0].code if blocks else response.text
        await ctx.send_message(
            TestSystemMessage(
                tokens=message.tokens, time=message.time,
                messages=message.messages + 1, prompt=message.prompt,
                signature=message.signature, original_func=message.original_func,
                code=message.code, tests=response.text, tests_str=tests,
                sender=self.id,
            ),
            target_id="test_executor",
        )
