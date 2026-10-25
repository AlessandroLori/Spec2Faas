"""Serverledge-only deploy experiment: explicit opt-in required for real HTTP calls."""
import json
from pathlib import Path

import aiohttp

from app.agents.FaaSDeployer import FaasDeployer
from experiments.MessageTypesTest import TestDeployMessage
from experiments.faas_deployer.Utils import (
    create_json,
    extract_param_names,
    extract_param_values,
)
from experiments.maf_evaluator_common import (
    Results,
    close_models,
    dataset,
    model_mapping,
    module_entry,
    new_items,
    output_path,
    require_remote,
    isolated_deployment,
)

# Conservare la denominazione effettiva della colonna del report originale.
COLUMNS = [
    'task_id', 'tokens', 'deployment time', 'deployed',
    'correctly executed', 'invocation attempts',
]

# AutoGen usa gli input JSON gia' predisposti per HumanEval+.
# Nel repository corrente esiste un file per ciascuno dei 164 task.
INPUTS_DIR = Path(__file__).resolve().parent / 'inputs'


def invocation_params(row):
    task_number = str(row.task_id).split('/')[-1]
    input_file = INPUTS_DIR / f'{task_number}_{row.entry_point}.json'

    if not input_file.exists():
        # Fallback IDENTICO nel criterio all'evaluator AutoGen:
        # calcolare l'input e salvarlo solo se non e' gia' disponibile.
        names = extract_param_names(row.prompt, row.entry_point)
        values = extract_param_values(row.test)
        if names is None or values is None or len(names) != len(values):
            raise ValueError(
                f'Cannot extract invocation parameters for {row.task_id}; '
                f'expected input file: {input_file}'
            )
        input_file.parent.mkdir(parents=True, exist_ok=True)
        create_json(names, values, str(input_file))

    with input_file.open('r', encoding='utf-8') as stream:
        params = json.load(stream)

    if not isinstance(params, dict):
        raise ValueError(f'Invocation parameters are not a JSON object: {input_file}')
    return params


async def invoke(url, handler, params):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        async with session.post(f'{url}/invoke/{handler}', json={'Params': params}) as response:
            await response.read()
            return response.status == 200


async def main(args):
    base_url = require_remote(args)
    llm, clients = model_mapping(args, ('faas_deployer',))
    results = Results(
        output_path(args, 'faas_deployer/deployer_results', llm['faas_deployer']),
        COLUMNS,
    )
    deployer = FaasDeployer(llm['faas_deployer'], clients['faas_deployer'])
    try:
        for row in new_items(dataset(args), results, args.limit):
            source = row.prompt + row.canonical_solution
            answer = await isolated_deployment(deployer, TestDeployMessage(source))
            deployed = answer.result != 'FAIL'
            executed = False
            if deployed:
                params = invocation_params(row)
                executed = await invoke(base_url, answer.result, params)
            results.append({
                'task_id': str(row.task_id),
                'tokens': answer.tokens,
                'deployment time': answer.time,
                'deployed': deployed,
                'correctly executed': executed,
                'invocation attempts': answer.invocation_attempts,
            })
    finally:
        await close_models(clients)


if __name__ == '__main__':
    module_entry(main, 'MAF Deployer Serverledge experiment (remote only)')
