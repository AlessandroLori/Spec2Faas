
import os
import re
import time
from typing import Never

from agent_framework import Executor, Message as LLMMessage, WorkflowContext, handler

from .messages.MessagesTypes import CodeMessage, ExecuteMessage
from .utils.Utils import dialogue, print_green, print_purple
from experiments.MessageTypesTest import TestCodeMessage, TestCodeResult, TestSystemMessage


_CODE_BLOCK = re.compile(r"```(?:\s*([\w\+\-]+))?\n([\s\S]*?)```")


class TestDesigner(Executor):
    SYSTEM_PROMPT = (
        "You are a very skilled test designer specializing in designing comprehensive and well-documented test cases for Python functions."
        "Your goal is to generate exhaustive test cases for the given function specification, covering all relevant input scenarios and ensuring correctness, robustness, and edge coverage."
        "<INSTRUCTIONS>"
        "First of all, make CORRECT tests."
        "- Focus on correctness, clarity, and diversity of test inputs."
        "- Each test case must:"
        "-- Follow the exact format below."
        "-- Include a short, meaningful comment or description of what it tests."
        "-- Be self-contained and syntactically valid Python."
        "-- Be semantically correct with respect to the specification."
        "- Only output test cases in the specified format Python format, nothing else."
        "- After you have produced the tests take your time to reflect if each on of them is correct."
        "</INSTRUCTIONS>"
        "<OUTPUT FORMAT>"
        "The format of test cases should be:"
        "```python"
        "assert function_name(input) == expected_output, 'Test Case <number>: Description'"
        "```"
        "</OUTPUT FORMAT>"
    )

    def __init__(self, llm: str, model_client=None, server=None) -> None:
        super().__init__(id="test_designer")
        self._model_client = model_client
        self._llm = llm
        self._client = None
        self._role = "Test Designer"
        self._server = server
        print_green(f"Hi I'm the test designer and I use {self._llm}.")

    def _build_messages(self, specification: str, signature: str):
        prompt = (
            "Function specification: "
            + specification
            + "\nFunction signature: "
            + signature
            + "\nPlease generate the tests:"
        )

        return [
            LLMMessage(role="system", contents=[self.SYSTEM_PROMPT]),
            LLMMessage(role="user", contents=[prompt]),
        ]

    @staticmethod
    def _total_tokens(response) -> int:
        usage = getattr(response, "usage_details", None)

        if (
            usage is None
            or usage.input_token_count is None
            or usage.output_token_count is None
        ):
            raise RuntimeError(
                "Il client MAF non ha restituito il conteggio dei token."
            )

        return usage.input_token_count + usage.output_token_count

    @handler
    async def handle_generate_code_message(
        self, message: CodeMessage, ctx: WorkflowContext[ExecuteMessage]
    ) -> None:
        print_green(
            f"{self.id} received message. Creating now a test suite for the function."
        )

        response = await self._model_client.get_response(
            self._build_messages(message.specification, message.function_signature)
        )

        if not isinstance(response.text, str):
            raise TypeError("TestDesigner expected a text response.")

        dialogue(response.text, self._role)

        if os.getenv("UI") == "True":
            self._server.send_chunk(response.text, "test_designer")

        await ctx.send_message(
            ExecuteMessage(message.specification, "", response.text, self.id)
        )

    @handler
    async def handle_generate_test_code_message(
        self,
        message: TestCodeMessage,
        ctx: WorkflowContext[Never, TestCodeResult],
    ) -> None:
        print_green(
            f"{self.id} received message. Creating now a test suite for the function."
        )

        messages = self._build_messages(
            message.specification, message.function_signature
        )

        start_time = time.perf_counter()
        response = await self._model_client.get_response(messages)
        execution_time = time.perf_counter() - start_time

        if not isinstance(response.text, str):
            raise TypeError("TestDesigner expected a text response.")

        tokens = self._total_tokens(response)

        await ctx.yield_output(
            TestCodeResult(response.text, execution_time, tokens, None)
        )

    @handler
    async def handle_test_system_generate_code_message(
        self, message: TestSystemMessage, ctx: WorkflowContext[TestSystemMessage]
    ) -> None:
        print_green(
            f"{self.id} received message. Creating now a test suite for the function."
        )
        print_purple(str(message))

        total_tokens = message.tokens
        total_time = message.time

        messages = self._build_messages(message.prompt, message.signature)

        start_time = time.perf_counter()
        response = await self._model_client.get_response(messages)
        elapsed = time.perf_counter() - start_time

        if not isinstance(response.text, str):
            raise TypeError("TestDesigner expected a text response.")

        tokens = self._total_tokens(response)

        total_time["test_designer"] = elapsed
        total_tokens["test_designer"] = tokens

        match = _CODE_BLOCK.search(response.text)
        tests = match.group(2) if match else response.text

        await ctx.send_message(
            TestSystemMessage(
                tokens=total_tokens,
                time=total_time,
                messages=message.messages + 1,
                prompt=message.prompt,
                signature=message.signature,
                original_func=message.original_func,
                code=message.code,
                tests=response.text,
                tests_str=tests,
                sender=self.id,
            )
        )
