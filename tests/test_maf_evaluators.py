
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_framework import Executor, WorkflowContext, handler

from app.agents.Assistant import Assistant
from app.agents.coding_agents.Coder import Coder
from app.agents.coding_agents.TestDesigner import TestDesigner
from experiments.MessageTypesTest import TestMessage, TestCodeMessage, TestExecCodeMessage, TestSystemMessage, TestDeployMessage, TestDeployResult
from experiments.maf_evaluator_common import (
    ROLES, Results, OfflineDeployer, base_columns, complexity,
    debugging_workflow, first_code,
)
from experiments.maf_evaluator_common import (
    legacy_passed, result_from_workflow, single_agent, isolated_deployment, system_workflow,
    run_in_docker, require_remote,
)
from experiments.system.System_evaluator import initial_metrics
from app.agents.coding_agents.utils.Code_Extractors import CodeBlock

"""Test di regressione per deterministic MAF: senza network docker or serverledge."""

class FakeModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def get_response(self, messages, options=None):
        self.calls.append((messages, options))
        if not self.responses:
            raise AssertionError('Unexpected additional LLM call')
        value = self.responses.pop(0)
        return SimpleNamespace(
            text=value,
            usage_details={'input_token_count': 11, 'output_token_count': 7},
        )


class FakeDocker:
    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def execute_code_blocks(self, blocks):
        self.calls.append(blocks)
        output = self.outputs.pop(0) if self.outputs else ''
        return SimpleNamespace(output=output, exit_code=0 if not output else 1)


class FakeDeployer(Executor):
    def __init__(self):
        super().__init__(id='faas_deployer')

    @handler
    async def deploy(self, message: TestDeployMessage,
                     ctx: WorkflowContext[TestDeployResult]) -> None:
        await ctx.send_message(TestDeployResult(
            result='dry_test', time=0.5, tokens=15, ctx=None,
            deployed_function=message.code, invocation_attempts=2,
        ), target_id='assistant')


class MetricsTest(unittest.TestCase):
    def test_legacy_output_columns(self):
        expected = {'task_id','CoG_original','CoG_final','time_system',
                    'number_messages_exchanged','token_debugger','deployed_function'}
        self.assertTrue(expected.issubset(set(base_columns())))

    def test_no_false_success_for_execution(self):
        self.assertFalse(legacy_passed(SimpleNamespace(exit_code=1, output='AssertionError: wrong')))
        self.assertTrue(legacy_passed(SimpleNamespace(exit_code=0, output='success')))
        self.assertFalse(legacy_passed(SimpleNamespace(exit_code=124, output='\n Timeout')))
        self.assertFalse(legacy_passed(SimpleNamespace(exit_code=137, output='Killed')))
        self.assertFalse(legacy_passed(SimpleNamespace(exit_code=2, output='')))
        with self.assertRaises(RuntimeError):
            legacy_passed(SimpleNamespace(output=''))

    def test_initial_roles(self):
        tokens, times = initial_metrics()
        self.assertEqual(set(tokens), set(ROLES))
        self.assertEqual(times['system'], 0)

    def test_default_deployment_denied(self):
        with self.assertRaises(RuntimeError):
            require_remote(SimpleNamespace(allow_deployment=False))

    def test_output_schema_written_and_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.parquet'
            table = Results(path, ['task_id', 'tokens', 'time'])
            table.append({'task_id': 'HumanEval/0', 'tokens': 18, 'time': 0.3})
            reopened = Results(path, ['task_id','tokens','time'])
            self.assertTrue(reopened.seen('HumanEval/0'))
            self.assertEqual(reopened.frame['tokens'].iloc[0], 18)

    def test_plain_python_code_extraction(self):
        self.assertEqual(first_code('```python\nx = 3\n```').strip(), 'x = 3')


class MAFWorkflowsTest(unittest.IsolatedAsyncioTestCase):
    async def test_deployment_result_uses_MAF_assistant_route(self):
        result = await isolated_deployment(FakeDeployer(), TestDeployMessage('def f(): return 1'))
        self.assertEqual(result.result, 'dry_test')
        self.assertEqual(result.tokens, 15)
        self.assertEqual(result.invocation_attempts, 2)

    async def test_assistant_isolated_usage(self):
        assistant = Assistant('fake', FakeModel('translation: Add two numbers'))
        answer = await single_agent(assistant, TestMessage('Add two numbers'))
        self.assertEqual(answer.type, 'translation')
        self.assertEqual(answer.tokens, 18)

    async def test_assistant_accepts_capitalized_directives(self):
        assistant = Assistant('fake', FakeModel('Translation: A function that adds numbers'))
        answer = await single_agent(assistant, TestMessage('Add two numbers'))
        self.assertEqual(answer.type, 'translation')
        self.assertEqual(answer.content, 'Translation: A function that adds numbers')

    async def test_assistant_does_not_change_wrong_prediction(self):
        # A translation answer to a deployment request is still a model error.
        assistant = Assistant('fake', FakeModel('Translation: some text'))
        answer = await single_agent(assistant, TestMessage('def f(): return 1'))
        self.assertEqual(answer.type, 'translation')

    async def test_coder_isolated_usage(self):
        coder = Coder('fake', FakeModel('```python\ndef add(a,b): return a+b\n```'))
        answer = await single_agent(coder, TestCodeMessage('sum', 'def add(a,b):', True))
        self.assertEqual(answer.tokens, 18)
        self.assertIn('def add', answer.content)

    async def test_test_designer_isolated_usage(self):
        agent = TestDesigner('fake', FakeModel('```python\nassert add(1,2)==3\n```'))
        answer = await single_agent(agent, TestCodeMessage('sum', 'def add(a,b):', True))
        self.assertEqual(answer.tokens, 18)
        self.assertIn('assert', answer.content)

    async def test_debugger_workflow_reaches_final_result(self):
        docker = FakeDocker('AssertionError', '')
        executor = FakeModel('```python\ndef f(): assert False\n```',
                             '```python\ndef f(): return 42\n```')
        debugger = FakeModel('```python\ndef f(): return 42\n```')
        answer = await debugging_workflow(
            executor, debugger, docker,
            TestExecCodeMessage('Return 42', 'f', 'def f(): assert False',
                                'def check(candidate): assert candidate() == 42'),
            ('fake', 'fake'),
        )
        self.assertTrue(answer.passed)
        self.assertGreater(answer.tokens, 0)

    async def test_system_generation_without_deployment(self):
        llm = {name: 'fake' for name in ROLES}
        models = {name: FakeModel() for name in ROLES}
        models['assistant'] = FakeModel('Add two numbers')
        models['entry_point'] = FakeModel('def add(a,b):')
        models['coder'] = FakeModel('```python\ndef add(a,b): return a+b\n```')
        models['test_designer'] = FakeModel('```python\nassert add(2,3)==5\n```')
        models['test_executor'] = FakeModel('```python\ndef add(a,b): return a+b\nassert add(2,3)==5\n```')
        docker = FakeDocker()
        workflow = system_workflow(llm, models, docker, allow_deployment=False)
        tokens, times = initial_metrics()
        response = result_from_workflow(await workflow.run(
            TestSystemMessage(tokens=tokens, time=times, prompt='Add two numbers')
        ))
        self.assertTrue(response.generated)
        self.assertFalse(response.deployed)
        self.assertEqual(response.tokens['coder'], 18)
        self.assertEqual(response.tokens['faas_deployer'], 0)
        self.assertGreater(response.messages, 1)
        self.assertEqual(len(models['faas_deployer'].calls), 0)

    async def test_benchmark_docker_image_is_required(self):
        fake = FakeDocker()
        fake.image = 'python:3-slim'
        with self.assertRaisesRegex(RuntimeError, 'prebuilt'):
            await run_in_docker(fake, 'def f(): return 1', 'assert f() == 1', 'f')

    async def test_docker_script_uses_isolated_executor(self):
        fake = FakeDocker()
        result, elapsed = await run_in_docker(fake, 'def add(x,y): return x+y',
                                              'def check(candidate): assert candidate(1,2)==3',
                                              'add')
        self.assertEqual(result.output, '')
        self.assertEqual(len(fake.calls[0]), 1)
        self.assertIn('check(add)', fake.calls[0][-1].code)
        self.assertGreaterEqual(elapsed, 0)


if __name__ == '__main__':
    unittest.main()
