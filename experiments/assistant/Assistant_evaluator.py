"""Isolated Assistant experiment: preserve two classifications per HumanEval+ task."""
from app.agents.Assistant import Assistant
from experiments.MessageTypesTest import TestMessage
from experiments.maf_evaluator_common import (
    Results, close_models, dataset, model_mapping, module_entry, output_path,
    single_agent,
)

COLUMNS = ['task_id', 'actual_type', 'type', 'time', 'tokens', 'response']


async def main(args):
    llm, clients = model_mapping(args, ('assistant',))
    results = Results(output_path(args, 'assistant/assistant_results', llm['assistant']), COLUMNS)
    assistant = Assistant(llm['assistant'], clients['assistant'])
    count = 0
    try:
        for row in dataset(args).itertuples(index=False):
            for category, content in (
                ('translation', row.prompt),
                ('deployment', row.prompt + row.canonical_solution),
            ):
                if results.seen(row.task_id, category=category):
                    continue
                answer = await single_agent(assistant, TestMessage(content))
                results.append({
                    'task_id': str(row.task_id), 'actual_type': category,
                    'type': answer.type, 'time': answer.time,
                    'tokens': answer.tokens, 'response': answer.content,
                })
            count += 1
            if args.limit > 0 and count >= args.limit:
                break
    finally:
        await close_models(clients)


if __name__ == '__main__':
    module_entry(main, 'MAF Assistant classification')
