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
        full_dataset = dataset(args)
        positions = {
            str(task_id): index
            for index, task_id in enumerate(full_dataset['task_id'], start=1)
        }
        total = len(full_dataset)
        completed = sum(results.seen(task_id) for task_id in positions)
        print(
            f"[Q1] Dataset: {total} task; gia' salvati: {completed}; "
            f"da valutare: {total - completed}.",
            flush=True,
        )

        for row in new_items(full_dataset, results, args.limit):
            task_id = str(row.task_id)
            print(f"[Q1] Task {positions[task_id]}/{total}: {task_id}", flush=True)
            answer = await single_agent(
                coder, TestCodeMessage(row.prompt, row.entry_point, args.prompt == 'yes')
            )
            code = first_code(answer.content)
            empty_response = not code.strip()
            if empty_response:
                print(
                    f"[Q1] {task_id}: il modello non ha restituito codice; "
                    "pass@1=False (generazione vuota).",
                    flush=True,
                )
            generated, execution_time = await run_in_docker(docker, code, row.test, row.entry_point)
            canonical = row.prompt + row.canonical_solution
            _, canonical_time = await run_in_docker(docker, canonical, row.test, row.entry_point)
            cc, cog = complexity(code)
            cc_original, cog_original = complexity(canonical)
            # Nel paper originale anche il codice vuoto veniva passato
            # all'esecutore. Nessun codice = nessuna funzione corretta.
            passed = False if empty_response else legacy_passed(generated)
            error = '' if passed else generated.output
            if empty_response:
                error = 'EMPTY_LLM_RESPONSE: nessun codice generato\n' + error
            results.append({
                'task_id': task_id, 'passed': passed,
                'generation_time': answer.time, 'tokens': answer.tokens,
                'execution_time': execution_time,
                'execution_time_canonical': canonical_time,
                'CC_generation': cc, 'CC_canonical': cc_original,
                'CoG_generation': cog, 'CoG_canonical': cog_original,
                'function': code,
                'error': error,
            })
            completed += 1
            print(
                f"[Q1] Salvato {completed}/{total}: {task_id} | "
                f"passed={passed} | tokens={answer.tokens} | "
                f"tempo LLM={answer.time:.2f}s",
                flush=True,
            )
    finally:
        await docker.stop()
        await close_models(clients)


if __name__ == '__main__':
    module_entry(main, 'MAF Coder HumanEval+ experiment')
