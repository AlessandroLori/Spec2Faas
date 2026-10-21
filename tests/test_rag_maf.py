import hashlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent_framework import Content, Message as LLMMessage

from app.Utils_database import ChromaRAGMemory, SCORE_THRESHOLD, TOP_K
from app.agents.coding_agents.Coder_RAG import Coder_RAG
from app.agents.coding_agents.EntryPoint_RAG import EntryPoint_RAG
from app.agents.coding_agents.TestDesigner_RAG import TestDesigner_RAG
from app.agents.coding_agents.messages.MessagesTypes import CodeMessage, Message
from app.main_RAG import build_workflow_rag
from experiments.MessageTypesTest import TestCodeMessage, TestSystemMessage


ROLES = (
    "assistant", "entry_point", "coder", "test_designer",
    "test_executor", "debugger", "faas_deployer",
)


class FakeModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def get_response(self, messages, options=None):
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("Unexpected LLM request")
        response = self.responses.pop(0)
        if isinstance(response, SimpleNamespace):
            return response
        return SimpleNamespace(
            text=response,
            messages=[LLMMessage(role="assistant", contents=[response])],
            usage_details={"input_token_count": 7, "output_token_count": 3},
        )


class FakeMemory:
    def __init__(self, documents):
        self.documents = documents
        self.queries = []

    async def query(self, text):
        self.queries.append(text)
        return self.documents


class FakeContext:
    def __init__(self):
        self.sent = []
        self.outputs = []

    async def send_message(self, message, target_id=None):
        self.sent.append((target_id, message))

    async def yield_output(self, message):
        self.outputs.append(message)


class FakeDocker:
    def __init__(self):
        self.calls = []

    async def execute_code_blocks(self, blocks):
        self.calls.append(blocks)
        return SimpleNamespace(output="", exit_code=0)


class FakeCollection:
    def __init__(self, distance):
        self.distance = distance
        self.query_calls = []
        self.add_calls = []

    def count(self):
        return 1

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return {"documents": [["ALGORITHM BinarySearch"]],
                "distances": [[self.distance]]}

    def add(self, **kwargs):
        self.add_calls.append(kwargs)


def call_response():
    arguments = {
        "code": "def handler(params, context): return 42\n",
        "name": "rag_smoke", "runtime": "python310",
        "memoryMB": 1024, "CPUDemand": 1,
    }
    content = Content.from_function_call(
        call_id="call_rag_1", name="create_json_serverledge",
        arguments=json.dumps(arguments),
    )
    return SimpleNamespace(
        text="", messages=[LLMMessage(role="assistant", contents=[content])],
        usage_details={"input_token_count": 7, "output_token_count": 3},
    )


class TestRAGMAF(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        ui = patch.dict(os.environ, {"UI": "False"})
        ui.start()
        self.addCleanup(ui.stop)

    def test_original_system_prompts(self):
        originals = {
            EntryPoint_RAG: "a67653b360cf6ffe6fb2832e9553669b05b2ccbaa9d674385d1760fbfeb33f94",
            Coder_RAG: "8a8f7213072bd518e0a97e7153f9c548d25bf8015557807e17fff76a3e2889d0",
            TestDesigner_RAG: "e9f7971398c5da7a525e58ced9d18cb073528e18b4835516be40fc0bbb10eebc",
        }
        for cls, expected_hash in originals.items():
            with self.subTest(cls=cls.__name__):
                self.assertEqual(hashlib.sha256(cls.SYSTEM_PROMPT.encode()).hexdigest(), expected_hash)

    async def test_retrieval_cosine_threshold_and_insertion(self):
        for distance, expected in [(0.2, ["ALGORITHM BinarySearch"]), (0.4, [])]:
            with self.subTest(distance=distance):
                collection = FakeCollection(distance)
                memory = ChromaRAGMemory(collection=collection)
                self.assertEqual(await memory.query("binary search"), expected)
                self.assertEqual(collection.query_calls[-1]["n_results"], 1)
                self.assertEqual(collection.query_calls[-1]["include"], ["documents", "distances"])
        self.assertEqual(TOP_K, 1)
        self.assertEqual(SCORE_THRESHOLD, 0.7)
        memory = ChromaRAGMemory(collection=collection)
        await memory.add("pseudocode")
        self.assertEqual(collection.add_calls[0]["documents"], ["pseudocode"])

    async def test_entry_point_rag_routes_context(self):
        memory = FakeMemory(["ALGORITHM BinarySearch"])
        model = FakeModel("def search(a, x):")
        agent = EntryPoint_RAG("fake", model, memory=memory)
        ctx = FakeContext()
        await agent.handle_assistant_message(Message("find an element", "request"), ctx)
        self.assertEqual(memory.queries, ["find an element"])
        self.assertEqual(ctx.sent[0][0], "coder")
        code = ctx.sent[0][1]
        self.assertIsInstance(code, CodeMessage)
        self.assertIn("ALGORITHM BinarySearch", code.memory_context)
        self.assertIn("<CONTEXT FROM MEMORY>", code.memory_context)
        self.assertIn("ALGORITHM BinarySearch", model.calls[0][1].text)
        self.assertEqual(len(model.calls[0]), 3)
        await agent.handle_assistant_message(Message("Still waiting", "test_executor_response"), ctx)
        self.assertEqual(ctx.sent[-1], ("test_designer", code))
        final = Message("correct code", "test_executor_response")
        await agent.handle_assistant_message(final, ctx)
        self.assertEqual(ctx.sent[-1], ("assistant", final))

    async def test_entry_point_no_memory(self):
        model = FakeModel("def f():")
        agent = EntryPoint_RAG("fake", model, memory=FakeMemory([]))
        ctx = FakeContext()
        await agent.handle_assistant_message(Message("task", "request"), ctx)
        self.assertEqual(ctx.sent[0][1].memory_context, "")

    async def test_coder_and_designer_receive_same_context(self):
        code_msg = CodeMessage("find an element", "def search(a, x):", "CONTEXT: BinarySearch")
        coder_model = FakeModel("```python\ndef search(a,x): return 1\n```")
        designer_model = FakeModel("```python\nassert search([1],1)==1\n```")
        coder = Coder_RAG("fake", coder_model)
        designer = TestDesigner_RAG("fake", designer_model)
        ctx1, ctx2 = FakeContext(), FakeContext()
        await coder.handle_generate_code_message(code_msg, ctx1)
        await designer.handle_generate_code_message(code_msg, ctx2)
        self.assertEqual(coder_model.calls[0][1].text, "CONTEXT: BinarySearch")
        self.assertEqual(designer_model.calls[0][1].text, "CONTEXT: BinarySearch")
        self.assertEqual(ctx1.sent[0][0], "test_executor")
        self.assertEqual(ctx2.sent[0][0], "test_executor")
        self.assertEqual(ctx1.sent[0][1].sender, "coder")
        self.assertEqual(ctx2.sent[0][1].sender, "test_designer")

    async def test_coder_isolated_and_system_metrics(self):
        model = FakeModel("```python\ndef f():return 1\n```", "```python\ndef f():return 2\n```")
        agent = Coder_RAG("fake", model)
        ctx = FakeContext()
        await agent.handle_generate_test_code_message(TestCodeMessage("spec", "def f():", True), ctx)
        self.assertEqual(ctx.outputs[0].tokens, 10)
        self.assertEqual(ctx.outputs[0].content.startswith("```python"), True)
        message = TestSystemMessage(tokens={}, time={}, prompt="spec", signature="def f():")
        await agent.handle_test_system_generate_code_message(message, ctx)
        dest, generated = ctx.sent[-1]
        self.assertEqual(dest, "test_executor")
        self.assertEqual(generated.tokens["coder"], 10)
        self.assertTrue(generated.new_chat)
        self.assertIn("return 2", generated.original_func)

    async def test_designer_isolated_and_system_metrics(self):
        model = FakeModel("```python\nassert f()==1\n```", "```python\nassert f()==2\n```")
        agent = TestDesigner_RAG("fake", model)
        ctx = FakeContext()
        await agent.handle_generate_test_code_message(TestCodeMessage("spec", "def f():", True), ctx)
        self.assertEqual(ctx.outputs[0].tokens, 10)
        msg = TestSystemMessage(tokens={}, time={}, prompt="spec", signature="def f():")
        await agent.handle_test_system_generate_code_message(msg, ctx)
        self.assertEqual(ctx.sent[-1][0], "test_executor")
        self.assertEqual(ctx.sent[-1][1].tokens["test_designer"], 10)
        self.assertIn("assert f()==2", ctx.sent[-1][1].tests_str)

    async def test_complete_rag_workflow_mocked_llms_and_serverledge(self):
        models = {role: FakeModel() for role in ROLES}
        models["assistant"] = FakeModel("translation: Sort numbers")
        models["entry_point"] = FakeModel("def sort_nums(nums):")
        models["coder"] = FakeModel("```python\ndef sort_nums(nums): return sorted(nums)\n```")
        models["test_designer"] = FakeModel("```python\nassert sort_nums([3,1]) == [1,3]\n```")
        models["test_executor"] = FakeModel(
            "```python\ndef sort_nums(nums): return sorted(nums)\nassert sort_nums([3,1]) == [1,3]\n```"
        )
        models["faas_deployer"] = FakeModel(call_response(), "Deployment completed")
        memory = FakeMemory(["ALGORITHM MergeSort"])
        docker = FakeDocker()
        workflow = build_workflow_rag(
            {role: "fake" for role in ROLES}, models, docker, memory,
        )
        with patch("app.agents.FaaSDeployer.create_json_serverledge", new_callable=AsyncMock) as deployment:
            deployment.return_value = {"name": "rag_smoke"}
            result = await workflow.run(Message("Ordina dei numeri", "request"))
            deployment.assert_awaited_once()
        self.assertEqual(result.get_outputs(), [Message("Deployment completed", "deployment")])
        self.assertEqual(memory.queries, [" Sort numbers"])
        self.assertIn("ALGORITHM MergeSort", models["coder"].calls[0][1].text)
        self.assertIn("ALGORITHM MergeSort", models["test_designer"].calls[0][1].text)
        self.assertEqual(len(docker.calls), 1)
        self.assertEqual(len(models["debugger"].calls), 0)


if __name__ == "__main__":
    unittest.main()
