"""Isolated Coder experiment; HumanEval+ and original complexity metrics."""
from app.agents.coding_agents.Coder import Coder
from experiments.MessageTypesTest import TestCodeMessage
from experiments.maf_evaluator_common import (
    evaluator_docker,
    Results, close_models, complexity, dataset, first_code, legacy_passed,
    model_mapping, module_entry, new_items, output_path, run_in_docker, single_agent,
)

COLUMNS = [
    'task_id', 'passed', 'generation_time', 'tokens', 'execution_time',
    'execution_time_canonical', 'CC_generation', 'CC_canonical',
    'CoG_generation', 'CoG_canonical', 'function', 'error',
]


async def main(args):
    llm, clients = model_mapping(args, ('coder',))
    suffix = '' if args.prompt == 'yes' else '_no_prompt'
    result_file = output_path(args, 'coder/coder_results', llm['coder'] + suffix)
    results = Results(result_file, COLUMNS)
    docker = evaluator_docker()
    coder = Coder(llm['coder'], clients['coder'])
    try:
        await docker.start()
        for row in new_items(dataset(args), results, args.limit):
            answer = await single_agent(
                coder, TestCodeMessage(row.prompt, row.entry_point, args.prompt == 'yes')
            )
            code = first_code(answer.content)
            generated, execution_time = await run_in_docker(docker, code, row.test, row.entry_point)
            canonical = row.prompt + row.canonical_solution
            _, canonical_time = await run_in_docker(docker, canonical, row.test, row.entry_point)
            cc, cog = complexity(code)
            cc_original, cog_original = complexity(canonical)
            results.append({
                'task_id': str(row.task_id), 'passed': legacy_passed(generated),
                'generation_time': answer.time, 'tokens': answer.tokens,
                'execution_time': execution_time,
                'execution_time_canonical': canonical_time,
                'CC_generation': cc, 'CC_canonical': cc_original,
                'CoG_generation': cog, 'CoG_canonical': cog_original,
                'function': code,
                'error': '' if legacy_passed(generated) else generated.output,
            })
    finally:
        await docker.stop()
        await close_models(clients)


if __name__ == '__main__':
    module_entry(main, 'MAF Coder HumanEval+ experiment')
