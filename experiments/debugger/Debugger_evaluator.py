"""Isolated debugging experiment using the MAF TestExecutor <-> Debugger workflow."""
from pathlib import Path

import pandas as pd

from experiments.MessageTypesTest import TestExecCodeMessage
from experiments.maf_evaluator_common import (
    evaluator_docker,
    Results, close_models, complexity, dataset, debugging_workflow, model_mapping,
    module_entry, new_items, output_path, ROOT,
)

COLUMNS = [
    'task_id','passed','passed_after_debugging','debugging_tokens','total_tokens',
    'generation_time','debugging_time','attempts',
    'CC_generation','CC_canonical','CoG_generation','CoG_canonical',
    'Original_function','Debugged_function','CC_debugged','CoG_debugged',
    'first_error','last_error',
]


def modify_tests(test):
    # Preserve original debugging diagnostic injection.
    fragment = ('        out = candidate(*inp)\n'
                '        if out != exp:\n'
                '            print("input: " + str(inp))\n'
                '            print("expected output: " + str(exp))\n'
                '            print("actual output: " + str(out))')
    lines = test.split('\n')
    idx = next((i for i, line in enumerate(lines)
                if line.strip().startswith('for i, (inp, exp) in enumerate(zip(inputs, results)):')), -1)
    assertion = next((i for i, line in enumerate(lines)
                      if line.strip().startswith('assertion(')), -1)
    if idx >= 0:
        lines.insert(idx + 1, fragment)
        if assertion > idx:
            assertion += 1
    if assertion >= 0:
        lines[assertion] = '        ' + lines[assertion].lstrip()
    return '\n'.join(lines)


async def main(args):
    llm, clients = model_mapping(args, ('test_executor', 'debugger'))
    coder_name = args.model
    debugger_name = args.debugger_model or args.model
    suffix = '' if args.prompt == 'yes' else '_no_prompt'
    results = Results(output_path(args, 'debugger/debugger_results',
                                  f'{coder_name}_{debugger_name}{suffix}'), COLUMNS)
    source = (Path(args.coder_results) if args.coder_results else
              ROOT / 'experiments/coder/coder_results' /
              (coder_name + suffix + '_maf.parquet'))
    coder_frame = pd.read_parquet(source).set_index('task_id')
    docker = evaluator_docker()
    try:
        await docker.start()
        for row in new_items(dataset(args), results, args.limit):
            task_id = str(row.task_id)
            if task_id not in coder_frame.index:
                raise ValueError(f'Missing matching Coder result for {task_id}')
            previous = coder_frame.loc[task_id]
            # NEVER zip two dataframes: align by task_id, not iteration position.
            code = str(previous['function'])
            initial_passed = bool(previous['passed'])
            answer = None
            if not initial_passed:
                answer = await debugging_workflow(
                    clients['test_executor'], clients['debugger'], docker,
                    TestExecCodeMessage(row.prompt, row.entry_point, code,
                                        modify_tests(row.test)),
                    (llm['test_executor'], llm['debugger']),
                )
            final = code if answer is None else answer.final_function
            cc_debugged, cog_debugged = complexity(final) if answer else (0, 0)
            debug_tokens = answer.tokens if answer else 0
            results.append({
                'task_id': task_id, 'passed': initial_passed,
                'passed_after_debugging': True if answer is None else answer.passed,
                'debugging_tokens': debug_tokens,
                'total_tokens': float(previous['tokens']) + debug_tokens,
                'generation_time': previous['generation_time'],
                'debugging_time': 0 if answer is None else answer.time,
                'attempts': 0 if answer is None else answer.attempts,
                'CC_generation': previous['CC_generation'],
                'CC_canonical': previous['CC_canonical'],
                'CoG_generation': previous['CoG_generation'],
                'CoG_canonical': previous['CoG_canonical'],
                'Original_function': code, 'Debugged_function': final,
                'CC_debugged': cc_debugged, 'CoG_debugged': cog_debugged,
                'first_error': '' if answer is None else answer.first_error,
                'last_error': '' if answer is None else answer.last_error,
            })
    finally:
        await docker.stop()
        await close_models(clients)


if __name__ == '__main__':
    module_entry(main, 'MAF Debugger HumanEval+ experiment')
