"""Read-only regression audit over the frozen Q5 Parquet.

The 164 functions are NOT re-run. No Serverledge/Ollama interaction occurs.
"""
import ast
from collections import Counter
from pathlib import Path

import pandas as pd

from app.agents.FaaSDeployer import validate_deployment_source

PARQUET = Path(
    "experiments/system/system_results/experiment_3/"
    "q5_maf_b_openweight.parquet"
)


def main():
    df = pd.read_parquet(PARQUET)
    assert len(df) == 164 and df.task_id.nunique() == 164
    failed = df.loc[df.deployed.eq(True) & df.correctly_executed.eq(False)]
    passed = df.loc[df.deployed.eq(True) & df.correctly_executed.eq(True)]
    tally = Counter()
    examples = {}

    for _, row in failed.iterrows():
        raw = str(row["deployed_function"])
        name = str(row["task_id"])
        try:
            ast.parse(raw)
        except SyntaxError:
            before_valid = False
        else:
            before_valid = True

        try:
            prepared = validate_deployment_source(raw)
        except (ValueError, TypeError) as exc:
            group = "REJECTED_INVALID"
            examples[name] = str(exc)
        else:
            assert ast.parse(prepared) is not None
            if before_valid:
                assert prepared == raw.lstrip("\n")
                group = "UNCHANGED_VALID"
            else:
                group = "NORMALIZED_VALID"
                assert prepared != raw
        tally[group] += 1

    passing_changed = []
    passing_rejected = []
    for _, row in passed.iterrows():
        raw = str(row["deployed_function"])
        try:
            prepared = validate_deployment_source(raw)
        except (ValueError, TypeError):
            passing_rejected.append(str(row["task_id"]))
            continue
        if prepared != raw.lstrip("\n"):
            passing_changed.append(str(row["task_id"]))

    print("===== Q5 DEPLOYER ESCAPING REGRESSION =====")
    print("Failed invocations:", len(failed))
    print("Success invocations:", len(passed))
    for k in ["NORMALIZED_VALID", "UNCHANGED_VALID", "REJECTED_INVALID"]:
        print(f"{k:20s}: {tally[k]}")
    print("Previously successful sources changed:", passing_changed)
    print("Previously successful sources rejected:", passing_rejected)
    print("Invalid sources, samples:")
    for task_id, error in list(examples.items())[:25]:
        print(f"  {task_id}: {error}")
    assert len(failed) == 71
    assert len(passed) == 68
    assert not passing_changed and not passing_rejected, (
        "Regression: previously successful deployment source changed!"
    )
    assert sum(tally.values()) == 71
    print("PASS — read-only regression audit completed")


if __name__ == "__main__":
    main()
