"""Re-check existing system results against HumanEval+ without remote infrastructure."""
from pathlib import Path

import pandas as pd

from experiments.maf_evaluator_common import (
    evaluator_docker,
    dataset, legacy_passed, module_entry, run_in_docker, ROOT,
)


async def main(args):
    src = (Path(args.source) if args.source else
           ROOT / 'experiments/system/system_results' /
           f'experiment_{args.experiment_number}' / 'results_maf.parquet')
    if not src.exists():
        raise FileNotFoundError(src)
    frame = pd.read_parquet(src)
    human = dataset(args).set_index('task_id')
    docker = evaluator_docker()
    count = 0
    try:
        await docker.start()
        for index, row in frame.iterrows():
            task = str(row['task_id'])
            reference = human.loc[task]
            original, _ = await run_in_docker(
                docker, str(row['original_function']), reference['test'],
                reference['entry_point'])
            final, _ = await run_in_docker(
                docker, str(row['final_function']), reference['test'],
                reference['entry_point'])
            frame.at[index, 'original_function_correct'] = legacy_passed(original)
            frame.at[index, 'final_function_correct'] = legacy_passed(final)
            count += 1
            if args.limit > 0 and count >= args.limit:
                break
        target = Path(args.output) if args.output else src
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
    finally:
        await docker.stop()


async def total_tokens(args):
    src = (Path(args.source) if args.source else
           ROOT / 'experiments/system/system_results' /
           f'experiment_{args.experiment_number}' / 'results_maf.parquet')
    if not src.exists():
        raise FileNotFoundError(src)
    df = pd.read_parquet(src)
    token_cols = [f'token_{role}' for role in (
        'assistant','entry_point','coder','designer','executor','debugger','deployer')]
    time_cols = [f'time_{role}' for role in (
        'assistant','entry_point','coder','designer','executor','debugger','deployer')]
    df['total_tokens'] = df[token_cols].sum(axis=1)
    df['total_time'] = df[time_cols].sum(axis=1)
    target = Path(args.output) if args.output else src
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False)


if __name__ == '__main__':
    # The original script's entry point computed aggregate time/token columns.
    module_entry(total_tokens, 'Aggregate MAF system token and time columns')
