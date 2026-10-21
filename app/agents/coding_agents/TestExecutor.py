
import os
import time

from agent_framework import Executor, Message as LLMMessage, WorkflowContext, handler

from .messages.MessagesTypes import DebugMessage, ExecuteMessage, Message
from .utils.Code_Extractors import CodeBlock, extract_markdown_code_blocks, extract_markdown_code_string
from .utils.Utils import dialogue, print_blue, print_green, print_purple, print_yellow
from experiments.MessageTypesTest import (
    TestDebugMessage,
    TestDebugResult,
    TestExecCodeMessage,
    TestExecCodeResult,
    TestSystemMessage,
)
from app.Utils import total_tokens_from_response


class TestExecutor(Executor):
    SYSTEM_PROMPT = (
        "You are a very skilled test executor."
        "Given the function to test and the tests, create a code that will make them runnable."
        "<INSTRUCTIONS>"
        "DO NOT MODIFY THE FUNCTION AND THE TESTS PROVIDED, just merge them in a single code."
        "</INSTRUCTIONS>"
    )

    def __init__(self, llm: str, model_client, code_executor, server=None) -> None:
        super().__init__(id="test_executor")

        self._model_client = model_client
        self._code_executor = code_executor
        self._llm = llm
        self._role = "Test Executor"
        self._server = server

        self._max_attempts = int(os.getenv("MAX_DEBUG_ATTEMPTS", "10"))

        if self._max_attempts <= 0:
            raise ValueError("MAX_DEBUG_ATTEMPTS must be greater than zero")

        self._reset()

        print_green(f"Hi I'm the test executor and I use {self._llm}.")

    def _reset(self) -> None:
        # Ripristina lo stato dopo una richiesta completata.
        self._mode = None
        self._code = ""
        self._tests = ""
        self._specification = ""
        self._signature = ""
        self._original_code = ""
        self._attempts = 0
        self._awaiting_debugger = False
        self._first_chat = True
        self._first_error = ""
        self._last_error = ""
        self._started_at = 0.0
        self._tokens = 0
        self._system_message = None
        self._tokens_executor = 0
        self._time_executor = 0.0
        self._tokens_debugger = 0
        self._time_debugger = 0.0
        self._debug_started = 0.0

    def _require_mode(self, mode: str) -> None:
        if self._mode is None:
            self._mode = mode
        elif self._mode != mode:
            raise RuntimeError(
                f"TestExecutor is processing '{self._mode}', not '{mode}'"
            )

    def _update_code_and_tests(self, code: str, tests: str) -> None:
        if not self._code:
            self._code = code

        if not self._tests:
            self._tests = tests

    def _build_messages(self):
        prompt = (
            "Function code: "
            + self._code
            + "\nFunction tests: "
            + self._tests
        )

        return [
            LLMMessage(role="system", contents=[self.SYSTEM_PROMPT]),
            LLMMessage(role="user", contents=[prompt]),
        ]

    @staticmethod
    def _total_tokens(response) -> int:
        return total_tokens_from_response(
            response,
            error_message="Il client MAF non ha restituito i token",
        )

    async def _execute(self, blocks: list[CodeBlock]):
        # Non eseguire mai il codice generato direttamente sull'host.
        result = await self._code_executor.execute_code_blocks(blocks)
        if not isinstance(getattr(result, "exit_code", None), int):
            raise RuntimeError("Docker did not return an integer exit_code")
        return result

    async def _send_normal_result(self, ctx, content: str) -> None:
        self._reset()

        await ctx.send_message(
            Message(content, "test_executor_response"),
            target_id="entry_point",
        )

    @handler
    async def handle_execute_code_message(
        self,
        message: ExecuteMessage,
        ctx: WorkflowContext[Message | DebugMessage],
    ) -> None:
        # Percorso normale della pipeline.
        self._require_mode("normal")

        if self._awaiting_debugger:
            raise RuntimeError(
                "Received new code/tests while waiting for Debugger"
            )

        print_green(
            f"{self.id} received message from {message.sender}"
        )

        self._specification = message.specification
        self._update_code_and_tests(message.code, message.tests)

        if not self._original_code and message.code:
            self._original_code = message.code

        # Come nell'originale, non eseguiamo nulla finché
        # non sono arrivati sia il codice sia i test.
        if not self._code or not self._tests:
            await ctx.send_message(
                Message("Still waiting", "test_executor_response"),
                target_id="entry_point",
            )
            return

        await self._run_normal(ctx)

    async def _run_normal(self, ctx) -> None:
        if self._attempts >= self._max_attempts:
            await self._send_normal_result(ctx, "FAIL")
            return

        response = await self._model_client.get_response(
            self._build_messages()
        )

        if not isinstance(response.text, str):
            raise TypeError(
                "TestExecutor expected a text response"
            )

        dialogue(response.text, self._role)

        blocks = extract_markdown_code_blocks(response.text)

        if not blocks:
            raise RuntimeError(
                "TestExecutor LLM did not return any executable code block"
            )

        execution = await self._execute(blocks)
        output = execution.output

        if output == "":
            print_yellow("The function passes all the tests.")

            if os.getenv("UI") == "True":
                self._server.send_chunk(
                    "The function passes all the tests.",
                    "test_executor",
                )
        else:
            print_yellow(output)

        if execution.exit_code != 0:
            if os.getenv("UI") == "True":
                self._server.send_chunk(
                    "The function fails the tests. Starting to debug...",
                    "test_executor",
                )

            self._awaiting_debugger = True

            # Inviamo al Debugger il codice effettivamente memorizzato.
            await ctx.send_message(
                DebugMessage(
                    self._specification,
                    self._code,
                    output,
                ),
                target_id="debugger",
            )
            return

        await self._send_normal_result(ctx, self._code)

    @handler
    async def handle_debug_code_message(
        self,
        message: DebugMessage,
        ctx: WorkflowContext[Message | DebugMessage],
    ) -> None:
        # Riprende la verifica dopo la risposta del Debugger.
        if self._mode != "normal" or not self._awaiting_debugger:
            raise RuntimeError(
                "Unexpected normal Debugger response"
            )

        self._awaiting_debugger = False
        self._code = message.code
        self._attempts += 1

        await self._run_normal(ctx)

    @handler
    async def handle_test_execute_code_message(
        self,
        message: TestExecCodeMessage,
        ctx: WorkflowContext[TestDebugMessage, TestExecCodeResult],
    ) -> None:
        # Valutazione isolata del codice.
        self._require_mode("isolated")

        self._code = message.code
        self._original_code = message.code
        self._tests = message.tests
        self._specification = message.specification
        self._signature = message.function_signature
        self._attempts = 0
        self._started_at = time.perf_counter()

        print_green(f"{self.id} received message.")

        await self._run_isolated(ctx)

    async def _finish_isolated(
        self,
        ctx,
        passed: bool,
        *,
        attempts: int | None = None,
    ) -> None:
        result = TestExecCodeResult(
            final_function=self._code,
            passed=passed,
            time=time.perf_counter() - self._started_at,
            tokens=self._tokens,
            attempts=(
                self._attempts
                if attempts is None
                else attempts
            ),
            first_error=self._first_error,
            last_error=self._last_error,
        )

        self._reset()

        await ctx.yield_output(result)

    async def _run_isolated(self, ctx) -> None:
        if self._attempts >= self._max_attempts:
            await self._finish_isolated(
                ctx,
                False,
                attempts=self._max_attempts,
            )
            return

        # Conserviamo la preparazione dell'ambiente originale.
        preinstalled = getattr(self._code_executor, "image", None) == "spec2faas-eval:py3"
        dependencies = (
            [] if preinstalled else
            extract_markdown_code_blocks("```sh\npip install numpy\n```")
        )

        code = (
            self._code
            + self._tests
            + "\n\ncheck("
            + self._signature
            + ")\n"
        )

        blocks = dependencies + [
            CodeBlock(code=code, language="python")
        ]

        execution = await self._execute(blocks)
        output = execution.output
        self._last_error = output

        if execution.exit_code != 0:
            print_blue(output)

            if self._attempts == 0:
                self._first_error = output

            self._awaiting_debugger = True

            await ctx.send_message(
                TestDebugMessage(
                    self._specification,
                    self._original_code,
                    output,
                    self._first_chat,
                ),
                target_id="debugger",
            )
            return

        await self._finish_isolated(ctx, True)

    @handler
    async def handle_test_system_execute_code_message(
        self,
        message: TestSystemMessage,
        ctx: WorkflowContext[TestSystemMessage | TestDebugMessage],
    ) -> None:
        # Valutazione completa del sistema.
        self._require_mode("system")

        if self._awaiting_debugger:
            raise RuntimeError(
                "Received system message while waiting for Debugger"
            )

        print_green(
            f"{self.id} received message from {message.sender}"
        )

        if message.new_chat:
            self._code = ""
            self._tests = ""
            self._attempts = 0

        self._update_code_and_tests(message.code, message.tests)

        if not self._code or not self._tests:
            await ctx.send_message(
                message,
                target_id="entry_point",
            )
            return

        print_purple(str(message))

        self._system_message = message
        self._started_at = time.perf_counter()

        await self._run_system(ctx)

    async def _run_system(self, ctx) -> None:
        if self._attempts >= self._max_attempts:
            await self._finish_system(ctx, False)
            return

        start = time.perf_counter()

        response = await self._model_client.get_response(
            self._build_messages()
        )

        self._tokens_executor += self._total_tokens(response)

        if not isinstance(response.text, str):
            raise TypeError(
                "TestExecutor expected a text response"
            )

        blocks = extract_markdown_code_blocks(response.text)

        if not blocks:
            raise RuntimeError(
                "TestExecutor LLM did not return any executable code block"
            )

        execution = await self._execute(blocks)
        output = execution.output

        self._time_executor += time.perf_counter() - start
        self._last_error = output

        if execution.exit_code != 0:
            if self._attempts == 0:
                self._first_error = output

            self._awaiting_debugger = True
            self._debug_started = time.perf_counter()

            await ctx.send_message(
                TestDebugMessage(
                    self._system_message.prompt,
                    self._system_message.code,
                    output,
                    self._first_chat,
                ),
                target_id="debugger",
            )
            return

        await self._finish_system(ctx, True)

    async def _finish_system(self, ctx, generated: bool) -> None:
        message = self._system_message

        message.time["test_executor"] = self._time_executor
        message.tokens["test_executor"] = self._tokens_executor
        message.time["debugger"] = self._time_debugger
        message.tokens["debugger"] = self._tokens_debugger

        final_function = ""

        if generated:
            blocks = extract_markdown_code_blocks(self._code)
            final_function = (
                blocks[0].code if blocks else self._code
            )

        result = TestSystemMessage(
            tokens=message.tokens,
            time=message.time,
            messages=message.messages + 1 + self._attempts,
            prompt=message.prompt,
            signature=message.signature,
            original_func=message.original_func,
            code=message.code,
            tests=message.tests,
            tests_str=message.tests_str,
            final_func=final_function,
            code_final_func=self._code if generated else "",
            attempts=(
                self._attempts
                if generated
                else (
                    self._max_attempts
                    if self._attempts >= self._max_attempts
                    else self._attempts
                )
            ),
            generated=generated,
            type="test_executor_response",
            first_error=self._first_error,
            last_error=self._last_error,
        )

        self._reset()

        await ctx.send_message(
            result,
            target_id="entry_point",
        )

    @handler
    async def handle_test_debug_result(
        self,
        message: TestDebugResult,
        ctx: WorkflowContext[
            TestDebugMessage | TestSystemMessage,
            TestExecCodeResult,
        ],
    ) -> None:
        # Riprende una valutazione sperimentale dopo il Debugger.
        if (
            not self._awaiting_debugger
            or self._mode not in ("isolated", "system")
        ):
            raise RuntimeError(
                "Unexpected experimental Debugger response"
            )

        self._awaiting_debugger = False

        if self._mode == "isolated":
            self._tokens += message.tokens

            if message.code == "":
                await self._finish_isolated(
                    ctx,
                    False,
                    attempts=self._max_attempts,
                )
                return

            self._code = extract_markdown_code_string(
                message.code
            )

            self._attempts += 1
            self._first_chat = False

            await self._run_isolated(ctx)

        else:
            self._tokens_debugger += message.tokens

            self._time_debugger += (
                time.perf_counter() - self._debug_started
            )

            if message.code == "":
                await self._finish_system(ctx, False)
                return

            self._code = message.code
            self._attempts += 1
            self._first_chat = False

            await self._run_system(ctx)
