
import ast
import asyncio
import base64
import io
import json
import os
import tarfile
import time

import aiohttp
from agent_framework import Content, Executor, FunctionTool, Message as LLMMessage, WorkflowContext, handler

from .Utils_deployer import find_arguments, get_func_call_from_json
from .coding_agents.messages.MessagesTypes import DeployMessage, FaasMessage
from .coding_agents.utils.Utils import dialogue, print_green, print_purple, print_yellow
from experiments.MessageTypesTest import TestDeployMessage, TestDeployResult
from app.Utils import total_tokens_from_response

# Conserviamo la preparazione del TAR e la richiesta HTTP originali.
# La validazione del codice protegge l'interfaccia tra l'LLM e Serverledge.


def validate_deployment_source(code: str) -> str:
    """Validate an LLM-produced handler and repair one extra newline-escaping layer.

    Normal Python is never rewritten. For syntactically invalid Python only,
    repair literal backslash-n separators if the source has no real line breaks
    and the repaired source parses. Reject every other invalid representation
    so the existing tool-error feedback can request another response from LLM.

    This is deliberately not a generic unicode_escape / JSON double-decode:
    those operations can silently alter string constants and backslashes.
    """
    if not isinstance(code, str):
        raise TypeError("Serverledge Python handler must be a string")

    source = code.lstrip("\n")
    if not source.strip():
        raise ValueError("Serverledge Python handler is empty")

    try:
        tree = ast.parse(source)
    except SyntaxError as original_error:
        # Do not modify mixed newline encodings or unrelated syntax errors.
        if "\n" in source or "\\n" not in source:
            raise ValueError(
                "Invalid Python handler: "
                f"{original_error.msg} (line {original_error.lineno}); "
                "return executable Python source with real line breaks"
            ) from original_error

        candidate = source.replace("\\n", "\n")
        try:
            tree = ast.parse(candidate)
        except SyntaxError as repair_error:
            raise ValueError(
                "Invalid Python handler after newline-escaping check: "
                f"{repair_error.msg} (line {repair_error.lineno}); "
                "return executable Python source with real line breaks"
            ) from repair_error

        source = candidate
        print_yellow(
            "FaaSDeployer: repaired literal backslash-n line breaks "
            "after successful Python syntax validation."
        )

    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "handler"
        for node in tree.body
    ):
        raise ValueError(
            "Python handler must define a top-level function named 'handler'"
        )

    return source


def _pystring_to_tarBase64(py_code, filename) -> str:
    data = py_code.encode("utf-8")
    mem = io.BytesIO()

    with tarfile.open(fileobj=mem, mode="w") as tar:
        info = tarfile.TarInfo(name=filename)
        info.size = len(data)
        info.mtime = int(time.time())
        info.mode = 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        tar.addfile(info, io.BytesIO(data))

    mem.seek(0)
    return base64.b64encode(mem.getvalue()).decode("ascii")


async def create_json_serverledge(
    code: str,
    name: str,
    runtime: str,
    memoryMB: int,
    CPUDemand: int,
) -> dict:
    new_code = validate_deployment_source(code)

    filename = name + ".py"
    print(new_code)

    tar_b64 = _pystring_to_tarBase64(
        new_code,
        filename=filename,
    )

    handler = name + ".handler"

    payload = {
        "Name": name,
        "Runtime": runtime,
        "MemoryMB": memoryMB,
        "CPUDemand": CPUDemand,
        "Handler": handler,
        "TarFunctionCode": tar_b64,
    }

    headers = {"Content-Type": "application/json"}
    url = os.getenv("SERVERLEDGE_URL") + "/create"

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        async with session.post(
            url,
            headers=headers,
            json=payload,
        ) as resp:
            print_yellow(
                f"Resp status: {resp.status} "
                f"Resp text: {await resp.text()}"
            )

            if resp.ok:
                return await resp.json()
            else:
                text = await resp.text()
                return {
                    "status": resp.status,
                    "text": text,
                }


class FaasDeployer(Executor):

    # System prompt IDENTICO alla versione originale AutoGen.
    SYSTEM_PROMPT = (
        "You are an expert cloud engineer responsible for deploying Python functions to a FaaS platform."
        "Your task is to follow a strict two-step process using the available tools."
        "The input will be a Python function code."
        "The tool you can use is create_json_serverledge and its description is 'Create the json payload for a request for Serverledge and deploy the function on Serveledge.'"
        "Step 1: Create the handler."
        "First, you must reformat the user's Python code into a valid handler structure. Do not modify the user's code provided but just add the handler."
        "Import necessary libraries outside the handler."
        "Example 1: "
        "def handler(params, context):"
        "n = params['n']"
        "return ''.join(fibonacci_nums(int(n)))"
        "def fibonacci_nums(n):"
        "sequence = \"\""
        "if n <= 0:"
        "sequence += \"0\""
        "return sequence"
        "sequence = \"0, 1\""
        "count = 2"
        "n1 = 0"
        "n2 = 1"
        "while count <= n:"
        "next_value = n2 + n1"
        "sequence += \",\" + \"\".join(str(next_value))"
        "n1 = n2"
        "n2 = next_value"
        "count += 1"
        "return sequence"
        "Example 2:"
        "def handler(params, context):"
        "print(\"Executing function....\")"
        "return \"Hello, Serverledge!\nParams: {}\".format(params)"
        "Example 3:"
        "import re"
        "def handler(params, context):"
        "return grep(\"grep\", params[\"InputText\"])"
        "def grep(pattern, text):"
        "lines = text.split('\n')"
        "result = [line for line in lines if re.search(pattern, line)]"
        "return '\n'.join(result)"
        "Step 2: Prepare the json payload and deploy the function to Serverledge."
        "Choose the appropriate Name, Runtime (default is python310), MemoryMB, CPUDemand and Handler."
        "The handler should invoke the function and return the result. The definition of the function has to be outside of the handler. Look carefully at the examples provided."
        "The handler should return a dictionary."
        "If you obtain errors from the server reflect on these errors. For example if the error is 404 'Chosen runtime does not exists' choose python310. "
        "If the error is 409 'Function already exists' choose another name for the function."
        "Only stop invoking tools when the server response is 200."
    )

    TOOL_DESCRIPTION = (
        "Create the json payload for a request for Serverledge and deploy "
        "the function on Serveledge."
    )

    def __init__(
        self,
        llm: str,
        model_client,
        tool_schema=None,
        server=None,
    ) -> None:
        super().__init__(id="faas_deployer")

        self._model_client = model_client
        self._llm = llm
        self._role = "FaaS Deployer"
        self._server = server

        self._system_messages = [
            LLMMessage(
                role="system",
                contents=[self.SYSTEM_PROMPT],
            )
        ]

        # MAF permette di dichiarare uno strumento senza
        # delegarne automaticamente l'esecuzione al client.
        #
        # In questo caso lo eseguiamo esplicitamente,
        # come nell'originale AutoGen, per misurare ogni
        # invocazione e gestire i tentativi successivi.
        #
        # Un eventuale tool_schema esterno deve quindi
        # contenere tool MAF senza invocazione automatica.
        self._tools = (
            tool_schema
            if tool_schema is not None
            else [
                FunctionTool(
                    name="create_json_serverledge",
                    description=self.TOOL_DESCRIPTION,
                    func=None,
                    input_model={
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "name": {"type": "string"},
                            "runtime": {"type": "string"},
                            "memoryMB": {"type": "integer"},
                            "CPUDemand": {"type": "integer"},
                        },
                        "required": [
                            "code",
                            "name",
                            "runtime",
                            "memoryMB",
                            "CPUDemand",
                        ],
                    },
                )
            ]
        )

        print_green(
            f"Hi I'm the faas deployer and I use {self._llm}."
        )

    @staticmethod
    def _total_tokens(response) -> int:
        return total_tokens_from_response(
            response,
            error_message="Missing token usage in FaaS Deployer response",
        )

    @staticmethod
    def _tool_calls(response) -> list[Content]:
        # Cerca le chiamate agli strumenti nella risposta MAF.
        messages = getattr(response, "messages", None) or []

        if not messages:
            return []

        return [
            item
            for item in messages[-1].contents
            if getattr(item, "type", None) == "function_call"
        ]

    @staticmethod
    def _arguments(call: Content) -> dict:
        # Gli argomenti possono arrivare come stringa JSON
        # oppure come dizionario.
        value = call.arguments

        parsed = (
            json.loads(value)
            if isinstance(value, str)
            else value
        )

        if not isinstance(parsed, dict):
            raise TypeError(
                "Tool call arguments must be a JSON object"
            )

        # Validate/normalize the code BEFORE both sending it to the tool
        # and retaining it as deployed_function in experiment results.
        # Never mutate the raw MAF function-call payload in the conversation.
        if "code" in parsed:
            parsed = {**parsed, "code": validate_deployment_source(parsed["code"])}

        return parsed

    @staticmethod
    async def _execute_tool_call(call: Content):
        # Spec2FaaS dispone di un solo strumento di deployment.
        if call.name != "create_json_serverledge":
            return None, f"Unknown tool: {call.name}"

        try:
            args = FaasDeployer._arguments(call)

            # Questa è la funzione originale che chiama
            # realmente l'API HTTP di Serverledge.
            result = await create_json_serverledge(**args)

            # Una risposta HTTP 4xx/5xx non equivale
            # a un deployment riuscito.
            if (
                isinstance(result, dict)
                and isinstance(result.get("status"), int)
                and result["status"] >= 400
            ):
                return None, json.dumps(
                    result,
                    ensure_ascii=False,
                )

            return result, None

        except Exception as exc:
            # Come nell'originale, l'errore viene restituito
            # al modello per consentire un eventuale retry.
            return None, str(exc)

    async def _run_deployment(
        self,
        code: str,
        *,
        experimental: bool,
    ):
        # Ogni richiesta avvia una nuova conversazione,
        # inizialmente composta da system prompt e codice.
        session = [
            *self._system_messages,
            LLMMessage(
                role="user",
                contents=["This is the code: " + code],
            ),
        ]

        start = time.perf_counter()

        tokens = 0
        attempts = 0

        last_arguments = {}
        full_function = ""
        last_result = None
        last_error = ""

        while True:
            # Ogni iterazione corrisponde a una richiesta LLM.
            response = await self._model_client.get_response(
                session,
                options={"tools": self._tools},
            )

            if experimental:
                tokens += self._total_tokens(response)

            calls = self._tool_calls(response)

            # Alcuni modelli Qwen 2.5 restituiscono
            # la chiamata allo strumento in forma JSON testuale.
            # Manteniamo questa compatibilità sperimentale.
            if (
                experimental
                and not calls
                and "qwen2.5" in self._llm
                and isinstance(response.text, str)
                and '"name": "create_json_serverledge"'
                in response.text
            ):
                try:
                    calls = get_func_call_from_json(response.text)
                except Exception:
                    return (
                        "FAIL",
                        time.perf_counter() - start,
                        tokens,
                        attempts,
                        last_arguments,
                        full_function,
                        "Invalid Qwen tool call",
                        "",
                    )

            # Se non ci sono più chiamate, la richiesta termina.
            if not calls:
                deployed_name = "FAIL"

                if last_result is not None and not last_error:
                    if isinstance(last_result, dict) and last_result:
                        deployed_name = str(
                            next(iter(last_result.values()))
                        )

                if deployed_name == "FAIL" and not last_error:
                    last_error = (
                        "No valid Serverledge deployment "
                        "was confirmed."
                    )

                return (
                    deployed_name,
                    time.perf_counter() - start,
                    tokens,
                    attempts,
                    last_arguments,
                    full_function,
                    last_error,
                    (
                        response.text
                        if isinstance(response.text, str)
                        else ""
                    ),
                )

            # Inserisce la richiesta del modello nella cronologia.
            session.append(
                LLMMessage(
                    role="assistant",
                    contents=list(calls),
                )
            )

            # Come nell'originale, consideriamo gli argomenti
            # del primo tool call per il risultato sperimentale.
            try:
                last_arguments = self._arguments(calls[0])
                full_function = last_arguments.get("code", "")
            except Exception:
                last_arguments = {}
                full_function = ""

            # Esegue tutte le chiamate del batch,
            # mantenendo la struttura asyncio.gather originale.
            results = await asyncio.gather(
                *(
                    self._execute_tool_call(call)
                    for call in calls
                )
            )

            attempts += 1

            contents = []

            for call, (tool_result, error) in zip(
                calls,
                results,
            ):
                last_result = tool_result
                last_error = error or ""

                tool_message = (
                    last_error
                    if error
                    else json.dumps(
                        tool_result,
                        ensure_ascii=False,
                    )
                )

                contents.append(
                    Content.from_function_result(
                        call_id=call.call_id,
                        result=tool_message,
                    )
                )

            # Il modello riceverà i risultati
            # degli strumenti nella prossima iterazione.
            session.append(
                LLMMessage(
                    role="tool",
                    contents=contents,
                )
            )

    @handler
    async def handle_deploy_message(
        self,
        message: DeployMessage,
        ctx: WorkflowContext[FaasMessage],
    ) -> None:
        # Deployment ordinario.
        print_green(
            f"{self.id} received message. "
            "Starting to deploy the function in FaaS."
        )

        (
            name,
            _elapsed,
            _tokens,
            _attempts,
            arguments,
            _code,
            error,
            final_text,
        ) = await self._run_deployment(
            message.code,
            experimental=False,
        )

        if name != "FAIL" and not error:
            details = find_arguments(
                arguments.get("code", "")
            )

            summary = (
                "The function was successfully deployed.\n\n"
                "You can invoke the function using the name "
                + str(arguments.get("name", name))
                + ".\n\nThe parameters are: '"
                + details
                + "'"
            )
        else:
            summary = (
                "The function couldn't be deployed. "
                "The error is:\n\n" + error
            )

        dialogue(summary, self._role)

        if os.getenv("UI") == "True":
            self._server.send_chunk(
                summary,
                "faas_deployer",
            )

        # Con MAF il risultato viene inviato all'Assistant.
        await ctx.send_message(
            FaasMessage(
                content=final_text,
                arguments=arguments,
                error_message=error,
            ),
            target_id="assistant",
        )

    @handler
    async def handle_test_deploy_message(
        self,
        message: TestDeployMessage,
        ctx: WorkflowContext[TestDeployResult],
    ) -> None:
        # Valutazione isolata ed end-to-end.
        print_green(
            f"{self.id} received message. "
            "Starting to deploy the function in FaaS."
        )

        print_purple(str(message))

        data = await self._run_deployment(
            message.code,
            experimental=True,
        )

        (
            name,
            elapsed,
            tokens,
            attempts,
            _arguments,
            full_function,
            *_,
        ) = data

        # In MAF non esiste il vecchio CancellationToken AutoGen.
        # Il campo ctx rimane per compatibilità con i messaggi
        # sperimentali esistenti.
        await ctx.send_message(
            TestDeployResult(
                result=name,
                time=elapsed,
                tokens=tokens,
                ctx=None,
                deployed_function=full_function,
                invocation_attempts=attempts,
            ),
            target_id="assistant",
        )
