import asyncio
import re
from pathlib import Path

import aiohttp
import yaml

from app.main import create_model_clients

from experiments.faas_deployer.Utils import (
    extract_function_name, extract_param_names, extract_param_values,
)

from app.agents.coding_agents.utils.Code_Extractors import CodeBlock
from experiments.MessageTypesTest import TestSystemMessage
from experiments.maf_evaluator_common import (
    evaluator_docker,
    Results, ROLES, base_columns, close_models, complexity, dataset,
    legacy_passed, new_items, output_path,
    require_remote, run_in_docker, system_workflow, options, initialize,
)


def load_model_configuration(args):
    """Load model configuration for each agent, as in the original implementation."""
    with Path(args.config).open('r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get('llm'), dict):
        raise ValueError('Configuration must contain an llm mapping')
    if config.get('experiment_number') != args.experiment_number:
        raise ValueError('Experiment number differs between YAML and --experiment-number')
    missing = set(ROLES) - set(config['llm'])
    if missing:
        raise ValueError(f'Missing models for roles: {sorted(missing)}')
    llm = {role: config['llm'][role] for role in ROLES}
    if not all(isinstance(model, str) and model.strip() for model in llm.values()):
        raise ValueError('Every role needs a nonempty model name')
    return llm, create_model_clients(llm)


def _metric(values, role):
    return values.get(role, 0)


def initial_metrics():
    tokens = {role: 0.0 for role in ROLES}
    times = {**{role: 0.0 for role in ROLES}, 'system': 0.0}
    return tokens, times


async def validate_function(docker, code, row):
    if not code or not str(code).strip():
        return False
    result, _ = await run_in_docker(docker, code, row.test, row.entry_point)
    return legacy_passed(result)


async def validate_generated_tests(docker, canonical_code, tests, signature):
    """Run generated tests on the canonical code, matching the original name adjustment."""
    canonical_name = extract_function_name(canonical_code)
    generated_name = extract_function_name(signature)
    if canonical_name and generated_name and canonical_name != generated_name:
        # Matches the name-alignment step in the original AutoGen evaluator.
        canonical_code = canonical_code.replace(canonical_name, generated_name)
    script = (
        'import coverage\n\n'
        'cov = coverage.Coverage()\n'
        'cov.start()\n\n'
        + canonical_code + '\n' + tests + '\n\n'
        'cov.stop()\ncov.report()\n'
    )
    result = await docker.execute_code_blocks([
        CodeBlock(code=script, language='python'),
    ])
    passed = legacy_passed(result)
    match = re.search(r'(\d+)\s*%', result.output)
    coverage = int(match.group(1)) if (passed and match) else 0
    return passed, coverage


async def invoke_function(base_url, function_name, params):
    """Count an invocation only when Serverledge reports Success=true.

    The original system evaluator inspected the `Success` field returned
    by Serverledge, rather than using the HTTP status alone.
    Network/protocol errors are raised so that the campaign can be resumed
    instead of silently counting an infrastructure outage as task failure.
    """
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            f'{base_url}/invoke/{function_name}',
            json={'Params': params, 'ReturnOutput': True},
        ) as response:
            if response.status != 200:
                await response.read()
                return False
            body = await response.json()
            return isinstance(body, dict) and body.get('Success') is True


async def main(args):
    base_url = require_remote(args) if args.allow_deployment else None
    llm, clients = load_model_configuration(args)
    result_file = output_path(
        args, 'system/system_results',
        f'experiment_{args.experiment_number}/results',
    )
    results = Results(result_file, base_columns())
    docker = evaluator_docker()
    try:
        await docker.start()
        workflow = system_workflow(
            llm, clients, docker,
            allow_deployment=args.allow_deployment,
        )
        for row in new_items(dataset(args), results, args.limit):
            tokens, times = initial_metrics()
            from experiments.maf_evaluator_common import result_from_workflow
            response = result_from_workflow(
                await workflow.run(TestSystemMessage(tokens=tokens, time=times,
                                                      prompt=row.prompt))
            )
            original = response.original_func
            final = response.final_func
            canonical = row.prompt + row.canonical_solution

            original_ok = await validate_function(docker, original, row)
            final_ok = await validate_function(docker, final, row) if response.generated else False

            # As in original experiment, run generated tests on canonical code,
            # collecting coverage inside the isolated Docker container.
            generated_tests = response.tests_str
            test_ok, test_coverage = (False, 0)
            if generated_tests:
                test_ok, test_coverage = await validate_generated_tests(
                    docker, canonical, generated_tests, response.signature
                )
            cc_o, cog_o = complexity(original) if original else (0, 0)
            cc_f, cog_f = complexity(final) if final else (0, 0)
            cc_c, cog_c = complexity(canonical)
            measured_deployed = bool(response.deployed) if args.allow_deployment else None
            measured_invoked = None
            if args.allow_deployment:
                measured_invoked = False
                if response.deployed:
                    try:
                        # A generated function may use different parameter names
                        # from those in the canonical function.
                        names = extract_param_names(response.signature, '', True)
                        values = extract_param_values(row.test)
                        if names is None or values is None or len(names) != len(values):
                            raise ValueError('cannot extract compatible invocation arguments')
                        params = dict(zip(names, values))
                    except (ValueError, TypeError, SyntaxError, KeyError) as exc:
                        # Keep per-task argument extraction failures separate from
                        # Serverledge connection/protocol failures. The latter stop
                        # the run, allowing resumption from the saved Parquet.
                        print(f'Invocation parameters unavailable for '
                              f'{row.task_id}: {type(exc).__name__}: {exc}', flush=True)
                    else:
                        measured_invoked = await invoke_function(
                            base_url, response.result_deployment, params
                        )
            record = {
                'task_id': str(row.task_id),
                'generated': bool(response.generated),
                'deployed': measured_deployed,
                'correctly_executed': measured_invoked,
                'debugged': bool(response.attempts),
                'original_function_correct': original_ok,
                'final_function_correct': final_ok,
                'test_correct': test_ok,
                'prompt': response.prompt,
                'signature': response.signature,
                'original_function': original,
                'final_function': final,
                'CC_original': cc_o, 'CC_final': cc_f, 'CC_canonical': cc_c,
                'CoG_final': cog_f, 'CoG_original': cog_o, 'CoG_canonical': cog_c,
                'time_assistant': _metric(response.time,'assistant'),
                'token_assistant': _metric(response.tokens,'assistant'),
                'time_entry_point': _metric(response.time,'entry_point'),
                'token_entry_point': _metric(response.tokens,'entry_point'),
                'time_coder': _metric(response.time,'coder'),
                'token_coder': _metric(response.tokens,'coder'),
                'time_designer': _metric(response.time,'test_designer'),
                'token_designer': _metric(response.tokens,'test_designer'),
                'time_executor': _metric(response.time,'test_executor'),
                'token_executor': _metric(response.tokens,'test_executor'),
                'time_debugger': _metric(response.time,'debugger'),
                'token_debugger': _metric(response.tokens,'debugger'),
                'time_deployer': _metric(response.time,'faas_deployer'),
                'token_deployer': _metric(response.tokens,'faas_deployer'),
                'time_system': _metric(response.time,'system'),
                'tests': generated_tests,
                'coverage': test_coverage,
                'debugging_attempts': response.attempts,
                'number_messages_exchanged': response.messages,
                'canonical_solution': canonical,
                'deployed_function': response.deployed_function,
            }
            results.append(record)
    finally:
        await docker.stop()
        await close_models(clients)


if __name__ == '__main__':
    initialize()
    parser = options('Complete MAF system experiment (offline deployment by default)')
    parser.add_argument('--config', required=True,
                        help='YAML file with the per-agent models, as in the original implementation')
    asyncio.run(main(parser.parse_args()))