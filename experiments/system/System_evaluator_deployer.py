"""Serverledge-only replay of system deployment, from an existing system Parquet."""
from pathlib import Path

import pandas as pd

from app.agents.FaaSDeployer import FaasDeployer
from experiments.MessageTypesTest import TestDeployMessage
from experiments.faas_deployer.Utils import extract_param_names, extract_param_values
from experiments.faas_deployer.Deployer_evaluator import invoke
from experiments.maf_evaluator_common import (
    Results, base_columns, close_models, dataset, model_mapping, module_entry,
    output_path, require_remote, isolated_deployment,
)

COLUMNS = base_columns() + ['execution_output', 'deployment_attempts']


async def main(args):
    url = require_remote(args)
    if not args.source:
        raise ValueError('Specify --source with previous MAF system results (do not use an inferred historical file)')
    previous = pd.read_parquet(args.source)
    previous = previous.set_index('task_id')
    llm, clients = model_mapping(args, ('faas_deployer',))
    results = Results(output_path(args, 'system/system_results',
                                  f'experiment_{args.experiment_number}/redeploy'), COLUMNS)
    deployer = FaasDeployer(llm['faas_deployer'], clients['faas_deployer'])
    count = 0
    try:
        for row in dataset(args).itertuples(index=False):
            task = str(row.task_id)
            if results.seen(task):
                continue
            if task not in previous.index:
                raise ValueError(f'Missing previous system result for {task}')
            old = previous.loc[task]
            record = {key: old[key] if key in old.index else None
                      for key in base_columns()}
            record['task_id'] = task
            deployed = False
            invoked = False
            attempts = 0
            deployment_time = 0
            deployment_tokens = 0
            deployed_function = ''
            output = ''
            if bool(old['generated']):
                answer = await isolated_deployment(deployer, TestDeployMessage(old['final_function']))
                deployed = answer.result != 'FAIL'
                attempts = answer.invocation_attempts
                deployment_time = answer.time
                deployment_tokens = answer.tokens
                deployed_function = answer.deployed_function
                if deployed:
                    names = extract_param_names(old['signature'], '', True)
                    values = extract_param_values(row.test)
                    if names is None or values is None or len(names) != len(values):
                        raise ValueError(f'Cannot extract test arguments for task {task}')
                    invoked = await invoke(url, answer.result, dict(zip(names, values)))
                    output = 'HTTP 200' if invoked else 'HTTP non-200'
                record['number_messages_exchanged'] = (old['number_messages_exchanged'] + 1)
            record.update({
                'deployed': deployed, 'correctly_executed': invoked,
                'time_deployer': deployment_time, 'token_deployer': deployment_tokens,
                'deployed_function': deployed_function, 'execution_output': output,
                'deployment_attempts': attempts,
            })
            results.append(record)
            count += 1
            if args.limit > 0 and count >= args.limit:
                break
    finally:
        await close_models(clients)


if __name__ == '__main__':
    module_entry(main, 'MAF system deployment replay (explicitly remote)')
