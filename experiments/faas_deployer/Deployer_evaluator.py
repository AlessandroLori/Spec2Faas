"""Serverledge-only deploy experiment: explicit opt-in required for real HTTP calls."""
import aiohttp

from app.agents.FaaSDeployer import FaasDeployer
from experiments.MessageTypesTest import TestDeployMessage
from experiments.faas_deployer.Utils import extract_param_names, extract_param_values
from experiments.maf_evaluator_common import (
    Results, close_models, dataset, model_mapping, module_entry, new_items,
    output_path, require_remote, isolated_deployment,
)

# Retain EXACT source column label with a space in "invocation attempts".
COLUMNS = ['task_id','tokens','deployment time','deployed','correctly executed',
           'invocation attempts']


async def invoke(url, handler, params):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        async with session.post(f'{url}/invoke/{handler}', json={'Params': params}) as response:
            await response.read()
            return response.status == 200


async def main(args):
    base_url = require_remote(args)
    llm, clients = model_mapping(args, ('faas_deployer',))
    results = Results(output_path(args, 'faas_deployer/deployer_results',
                                  llm['faas_deployer']), COLUMNS)
    deployer = FaasDeployer(llm['faas_deployer'], clients['faas_deployer'])
    try:
        for row in new_items(dataset(args), results, args.limit):
            source = row.prompt + row.canonical_solution
            answer = await isolated_deployment(deployer, TestDeployMessage(source))
            deployed = answer.result != 'FAIL'
            executed = False
            if deployed:
                names = extract_param_names(row.prompt, row.entry_point)
                values = extract_param_values(row.test)
                if names is None or values is None or len(names) != len(values):
                    raise ValueError(f'Cannot extract invocation parameters for {row.task_id}')
                executed = await invoke(base_url, answer.result, dict(zip(names, values)))
            results.append({
                'task_id': str(row.task_id), 'tokens': answer.tokens,
                'deployment time': answer.time, 'deployed': deployed,
                'correctly executed': executed,
                'invocation attempts': answer.invocation_attempts,
            })
    finally:
        await close_models(clients)


if __name__ == '__main__':
    module_entry(main, 'MAF Deployer Serverledge experiment (remote only)')
