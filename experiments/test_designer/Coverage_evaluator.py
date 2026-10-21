"""Postprocess designer results with Docker-based coverage, no AutoGen imports."""
import re
from pathlib import Path

import pandas as pd

from app.agents.coding_agents.utils.Code_Extractors import CodeBlock
from experiments.maf_evaluator_common import (
    evaluator_docker,
    dataset, module_entry, legacy_passed, ROOT, run_in_docker,
)


def coverage_percent(text):
    match = re.search(r'(\d+)\s*%', text)
    return int(match.group(1)) if match else 0


async def run_coverage(executor, code, tests):
    import_script = f'{code}\n\n{tests}'
    script = (
        'from pathlib import Path\n'
        f'Path("solution.py").write_text({import_script!r})\n'
        'import subprocess\n'
        'p = subprocess.run(["coverage", "run", "--source=.", "solution.py"], capture_output=True, text=True)\n'
        'r = subprocess.run(["coverage", "report", "-m"], capture_output=True, text=True)\n'
        'print(p.stdout, p.stderr, r.stdout, r.stderr)\n'
    )
    result = await executor.execute_code_blocks([
        CodeBlock(code=script, language='python'),
    ])
    if not legacy_passed(result):
        return 0
    return coverage_percent(result.output)


async def main(args):
    result_file = (Path(args.source) if args.source else
                   ROOT / 'experiments/test_designer/designer_results' /
                   (args.model + '_maf.parquet'))
    if not result_file.exists():
        raise FileNotFoundError(f'No designer results at {result_file}')
    frame = pd.read_parquet(result_file)
    human = dataset(args).set_index('task_id')
    docker = evaluator_docker()
    count = 0
    try:
        await docker.start()
        for idx, row in frame.iterrows():
            if not row['passed']:
                continue
            task = str(row['task_id'])
            if task not in human.index:
                raise ValueError(f'Missing HumanEval task {task}')
            original = human.loc[task]
            code = original['prompt'] + original['canonical_solution']
            output, _ = await run_in_docker(docker, code, str(row['tests']), entry_point=None)
            passed = legacy_passed(output)
            frame.at[idx, 'passed'] = passed
            frame.at[idx, 'coverage'] = (await run_coverage(docker, code, str(row['tests']))) if passed else 0
            count += 1
            if args.limit > 0 and count >= args.limit:
                break
        if args.output:
            target = Path(args.output)
        else:
            target = result_file
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
    finally:
        await docker.stop()


if __name__ == '__main__':
    module_entry(main, 'MAF TestDesigner coverage postprocessing')
