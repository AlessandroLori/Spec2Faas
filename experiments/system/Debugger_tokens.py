"""Recompute experimental debugger metrics from recorded system outputs."""
from pathlib import Path

import pandas as pd

from experiments.MessageTypesTest import TestExecCodeMessage
from experiments.maf_evaluator_common import (
    evaluator_docker,
    close_models, debugging_workflow, model_mapping, module_entry, ROOT,
)


async def main(args):
    src = (Path(args.source) if args.source else
           ROOT / 'experiments/system/system_results' /
           f'experiment_{args.experiment_number}' / 'results_maf.parquet')
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_parquet(src)
    names, clients = model_mapping(args, ('test_executor', 'debugger'))
    docker = evaluator_docker()
    count = 0
    try:
        await docker.start()
        for idx, row in df.iterrows():
            if not row['debugged']:
                continue
            answer = await debugging_workflow(
                clients['test_executor'], clients['debugger'], docker,
                TestExecCodeMessage(row['prompt'], row['signature'],
                                    row['original_function'], row['tests']),
                (names['test_executor'], names['debugger']),
            )
            df.at[idx, 'token_debugger'] = answer.tokens
            count += 1
            if args.limit > 0 and count >= args.limit:
                break
        target = Path(args.output) if args.output else src
        target.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(target, index=False)
    finally:
        await docker.stop()
        await close_models(clients)


if __name__ == '__main__':
    module_entry(main, 'Re-measure MAF experimental debugging usage')
