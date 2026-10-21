"""Utilities shared by Spec2FaaS evaluators after the AutoGen -> MAF migration.

Run evaluators as modules from repository root. Never overwrite historical
experiment files: default destinations have a _maf suffix.
"""
import argparse
import asyncio
import os
import re
import time
from pathlib import Path
from typing import Never

from agent_framework import Executor, WorkflowBuilder, WorkflowContext, handler

from app.agents.Assistant import Assistant
from app.agents.FaaSDeployer import FaasDeployer
from app.agents.coding_agents.Coder import Coder
from app.agents.coding_agents.messages.MessagesTypes import DeployMessage, FaasMessage
from app.agents.coding_agents.Debugger import Debugger
from app.agents.coding_agents.EntryPoint import EntryPoint
from app.agents.coding_agents.TestDesigner import TestDesigner
from app.agents.coding_agents.TestExecutor import TestExecutor
from app.agents.coding_agents.utils.Code_Extractors import (
    CodeBlock, extract_markdown_code_blocks,
)
from app.agents.coding_agents.utils.DockerCodeExecutor import DockerCodeExecutor
from app.Utils import load_env_variables
from app.main import create_model_clients, _close_model_clients
from experiments.MessageTypesTest import TestDeployMessage, TestDeployResult

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = 'hf://datasets/evalplus/humanevalplus/data/test-00000-of-00001-5973903632b82d40.parquet'
ROLES = ('assistant', 'entry_point', 'coder', 'test_designer', 'test_executor', 'debugger', 'faas_deployer')


def options(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--dataset', default=DEFAULT_DATASET,
                   help='Parquet HumanEval+ (a local file is recommended for first tests)')
    p.add_argument('--limit', type=int, default=1, help='Maximum number of new tasks (0 = entire dataset)')
    p.add_argument('--model', default='qwen2.5:1.5b',
                   help='Local Ollama model. Specify remote models explicitly only if intended')
    p.add_argument('--output', default=None, help='Override result file; never write original experiment files')
    p.add_argument('--prompt', choices=['yes', 'no'], default='yes',
                   help='Include the original system prompt in isolated coder/designer tests')
    p.add_argument('--debugger-model', default=None)
    p.add_argument('--coder-results', default=None, help='Prior coder Parquet for Debugger')
    p.add_argument('--source', default=None, help='Prior system Parquet for replay/redeploy')
    p.add_argument('--experiment-number', type=int, default=1)
    p.add_argument('--allow-deployment', action='store_true',
                   help='Explicit opt-in: real deployment to SERVERLEDGE_URL')
    return p


def require_remote(args):
    if not args.allow_deployment:
        raise RuntimeError('This evaluator requires Serverledge. Repeat with --allow-deployment only after verifying SERVERLEDGE_URL.')
    url = os.environ.get('SERVERLEDGE_URL', '')
    if not url.startswith(('http://', 'https://')):
        raise RuntimeError('SERVERLEDGE_URL is not configured as an HTTP(S) URL')
    return url.rstrip('/')


def dataset(args):
    import pandas as pd
    df = pd.read_parquet(args.dataset)
    required = {'task_id', 'entry_point', 'prompt', 'canonical_solution', 'test'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'HumanEval+ dataset missing required columns: {sorted(missing)}')
    return df


def output_path(args, subfolder, name):
    if args.output:
        path = Path(args.output)
    else:
        path = ROOT / 'experiments' / subfolder / (name + '_maf.parquet')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class Results:
    def __init__(self, path, columns):
        import pandas as pd
        self.path = Path(path)
        self.columns = list(columns)
        self.frame = pd.read_parquet(self.path) if self.path.exists() else pd.DataFrame(columns=self.columns)
        missing = set(self.columns) - set(self.frame.columns)
        if missing:
            raise ValueError(f'Result file {self.path} missing expected columns {sorted(missing)}')

    def seen(self, task_id, *, category=None):
        rows = self.frame[self.frame['task_id'].astype(str) == str(task_id)]
        if category is not None:
            rows = rows[rows['actual_type'] == category]
        return not rows.empty

    def append(self, record):
        import pandas as pd
        missing = set(self.columns) - record.keys()
        if missing:
            raise ValueError(f'Record missing fields: {sorted(missing)}')
        frame = pd.DataFrame([{key: record[key] for key in self.columns}])
        self.frame = pd.concat([self.frame, frame], ignore_index=True)
        self.frame['task_id'] = self.frame['task_id'].astype('string')
        self.frame.to_parquet(self.path, engine='pyarrow', index=False)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_parquet(self.path, engine='pyarrow', index=False)


def new_items(frame, results, limit):
    done = 0
    for row in frame.itertuples(index=False):
        if results.seen(row.task_id):
            continue
        yield row
        done += 1
        if limit > 0 and done >= limit:
            break


def base_columns():
    return [
        'task_id','generated','deployed','correctly_executed','debugged',
        'original_function_correct','final_function_correct','test_correct',
        'prompt','signature','original_function','final_function',
        'CC_original','CC_final','CC_canonical',
        'CoG_final','CoG_original','CoG_canonical',
        'time_assistant','token_assistant','time_entry_point','token_entry_point',
        'time_coder','token_coder','time_designer','token_designer',
        'time_executor','token_executor','time_debugger','token_debugger',
        'time_deployer','token_deployer','time_system',
        'tests','coverage','debugging_attempts','number_messages_exchanged',
        'canonical_solution','deployed_function',
    ]


def isolated(agent, message):
    """Run an existing @handler that yields its experimental result."""
    workflow = WorkflowBuilder(start_executor=agent).build()
    return workflow.run(message)


def result_from_workflow(run_result):
    values = run_result.get_outputs()
    if len(values) != 1:
        raise RuntimeError(f'Expected 1 MAF output, got {len(values)}')
    return values[0]


async def single_agent(agent, message):
    return result_from_workflow(await isolated(agent, message))


async def debugging_workflow(client_executor, client_debugger, docker, message, names):
    executor = TestExecutor(llm=names[0], model_client=client_executor, code_executor=docker)
    debugger = Debugger(llm=names[1], model_client=client_debugger)
    workflow = (WorkflowBuilder(start_executor=executor, max_iterations=200)
                .add_edge(executor, debugger)
                .add_edge(debugger, executor)
                .build())
    return result_from_workflow(await workflow.run(message))


class DeploymentCollector(Executor):
    """Receive the result sent by the actual MAF FaasDeployer to target_id assistant."""
    def __init__(self):
        super().__init__(id='assistant')

    @handler
    async def collect(self, message: TestDeployResult,
                      ctx: WorkflowContext[Never, TestDeployResult]) -> None:
        await ctx.yield_output(message)


async def isolated_deployment(deployer, message):
    collector = DeploymentCollector()
    workflow = (WorkflowBuilder(start_executor=deployer)
                .add_edge(deployer, collector).build())
    return result_from_workflow(await workflow.run(message))


class OfflineDeployer(Executor):
    """Explicitly NOT a successful deploy; used only for system generation checks."""
    def __init__(self):
        super().__init__(id='faas_deployer')

    @handler
    async def handle(self, message: TestDeployMessage,
                     ctx: WorkflowContext[TestDeployResult]) -> None:
        await ctx.send_message(
            TestDeployResult(result='FAIL', time=0.0, tokens=0, ctx=None,
                             deployed_function='', invocation_attempts=0),
            target_id='assistant',
        )

    @handler
    async def handle_normal(self, message: DeployMessage,
                            ctx: WorkflowContext[FaasMessage]) -> None:
        await ctx.send_message(
            FaasMessage(content='FAIL', arguments={}, error_message='Remote deployment disabled'),
            target_id='assistant',
        )


def system_workflow(llm, clients, docker, *, allow_deployment=False):
    # Construct all seven existing MAF agents, with a fail-closed deploy stub by default.
    assistant = Assistant(llm['assistant'], clients['assistant'])
    entry = EntryPoint(llm['entry_point'], clients['entry_point'])
    coder = Coder(llm['coder'], clients['coder'])
    designer = TestDesigner(llm['test_designer'], clients['test_designer'])
    executor = TestExecutor(llm['test_executor'], clients['test_executor'], docker)
    debugger = Debugger(llm['debugger'], clients['debugger'])
    deployer = (FaasDeployer(llm['faas_deployer'], clients['faas_deployer'])
                if allow_deployment else OfflineDeployer())
    return (WorkflowBuilder(start_executor=assistant, max_iterations=200)
            .add_edge(assistant, entry).add_edge(entry, assistant)
            .add_edge(entry, coder).add_edge(coder, executor)
            .add_edge(executor, entry).add_edge(entry, designer)
            .add_edge(designer, executor).add_edge(executor, debugger)
            .add_edge(debugger, executor).add_edge(assistant, deployer)
            .add_edge(deployer, assistant).build())


def model_mapping(args, roles):
    role_names = {role: args.debugger_model or args.model if role == 'debugger'
                  else args.model for role in roles}
    return role_names, create_model_clients(role_names)


async def close_models(models):
    await _close_model_clients(models)


def first_code(text):
    blocks = extract_markdown_code_blocks(text)
    return blocks[0].code if blocks else text


# Immagine preparata una volta sola: nessuna installazione pip durante i benchmark.
EVALUATOR_IMAGE = "spec2faas-eval:py3"


def evaluator_docker():
    """Esecutore sperimentale con numpy e coverage preinstallati."""
    return DockerCodeExecutor(image=EVALUATOR_IMAGE)


async def run_in_docker(docker, function, tests, entry_point=None, *, install_numpy=True):
    """Esegue i test in Docker senza alterare il tempo con installazioni pip.

    L'argomento install_numpy e' mantenuto soltanto per compatibilita' con i
    chiamanti precedenti; nel runtime sperimentale le dipendenze sono gia'
    presenti nell'immagine spec2faas-eval:py3.
    """
    if install_numpy and getattr(docker, "image", EVALUATOR_IMAGE) != EVALUATOR_IMAGE:
        raise RuntimeError(
            "HumanEval+ requires the prebuilt spec2faas-eval:py3 Docker image. "
            "Build it first; do not install packages while measuring execution time."
        )
    script = function + '\n' + tests
    if entry_point is not None:
        script += '\n\ncheck(' + entry_point + ')\n'

    blocks = [CodeBlock(code=script, language='python')]
    start = time.perf_counter()
    result = await docker.execute_code_blocks(blocks)
    elapsed = time.perf_counter() - start
    return result, elapsed


def legacy_passed(result):
    """Esito autorevole: 0 = successo, ogni altro codice = fallimento.

    Il nome e' conservato per compatibilita' con gli evaluator migrati.
    Non applica piu' la vecchia euristica sulla presenza di 'Error'.
    """
    exit_code = getattr(result, 'exit_code', None)
    if not isinstance(exit_code, int):
        raise RuntimeError('Docker execution result is missing integer exit_code')
    return exit_code == 0


def complexity(code):
    # The same Metrics implementation used by the original project.
    from experiments.coder.Metrics import compute_CC, compute_CoG
    return compute_CC(code), compute_CoG(code)


def task_id(row):
    return str(row.task_id)


def initialize():
    load_env_variables()
    # No accidental use of the university's former Ollama endpoint.
    os.environ.setdefault('OLLAMA_HOST', 'http://127.0.0.1:11434')
    os.environ['UI'] = 'False'


async def evaluate_module(module_func, description):
    initialize()
    args = options(description).parse_args()
    await module_func(args)


def module_entry(module_func, description):
    asyncio.run(evaluate_module(module_func, description))
