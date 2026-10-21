"""Isolated TestDesigner; preserve original Parquet schema and coverage placeholder."""
from app.agents.coding_agents.TestDesigner import TestDesigner
from app.agents.coding_agents.utils.Code_Extractors import extract_markdown_code_blocks
from experiments.MessageTypesTest import TestCodeMessage
from experiments.maf_evaluator_common import (
    evaluator_docker,
    Results, close_models, dataset, legacy_passed, model_mapping,
    module_entry, new_items, output_path, run_in_docker, single_agent,
)

COLUMNS = ['task_id','passed','generation_time','tokens','execution_time','tests','coverage']


async def main(args):
    llm, clients = model_mapping(args, ('test_designer',))
    suffix = '' if args.prompt == 'yes' else '_no_prompt'
    results = Results(output_path(args, 'test_designer/designer_results',
                                  llm['test_designer'] + suffix), COLUMNS)
    designer = TestDesigner(llm['test_designer'], clients['test_designer'])
    docker = evaluator_docker()
    try:
        await docker.start()
        for row in new_items(dataset(args), results, args.limit):
            answer = await single_agent(
                designer, TestCodeMessage(row.prompt, row.entry_point, args.prompt == 'yes')
            )
            blocks = extract_markdown_code_blocks(answer.content)
            tests = blocks[0].code if blocks else ''
            elapsed = 0.0
            passed = False
            if blocks:
                executed, elapsed = await run_in_docker(
                    docker, row.prompt + row.canonical_solution, tests,
                    entry_point=None,
                )
                passed = legacy_passed(executed)
            results.append({
                'task_id': str(row.task_id), 'passed': passed,
                'generation_time': answer.time, 'tokens': answer.tokens,
                'execution_time': elapsed, 'tests': tests, 'coverage': 0,
            })
    finally:
        await docker.stop()
        await close_models(clients)


if __name__ == '__main__':
    module_entry(main, 'MAF TestDesigner HumanEval+ experiment')
