"""Single-task, LLM-free Serverledge smoke test of the escaping fix.

NOTE: Creates ONE clearly named diagnostic function in the current registry.
The frozen Q5 Parquet is not modified. No full Q5 workflow is run.
"""
import asyncio
import ast
from datetime import datetime, timezone
from pathlib import Path
import os

import aiohttp
import pandas as pd

from app.agents.FaaSDeployer import (
    create_json_serverledge,
    validate_deployment_source,
)

PARQUET = Path(
    "experiments/system/system_results/experiment_3/"
    "q5_maf_b_openweight.parquet"
)


async def main():
    base = os.environ.get("SERVERLEDGE_URL", "http://127.0.0.1:1423").rstrip("/")
    df = pd.read_parquet(PARQUET)
    row = df.loc[df.task_id.eq("HumanEval/3")]
    assert len(row) == 1
    raw = row.iloc[0]["deployed_function"]
    assert isinstance(raw, str)
    assert "\\n" in raw and "\n" not in raw

    code = validate_deployment_source(raw)
    ast.parse(code)
    assert code != raw

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    name = f"spec2faas_escape_probe_{stamp}"
    print("===== ONE-TASK SERVERLEDGE SMOKE =====")
    print("Historic task: HumanEval/3")
    print("New diagnostic function:", name)
    print("Serverledge:", base)
    print("Original escaped separators:", raw.count("\\n"))
    print("Valid Python lines:", code.count("\n") + 1)
    print("NOTE: creates ONE function in the currently connected registry.")

    os.environ["SERVERLEDGE_URL"] = base
    result = await create_json_serverledge(
        code=raw,  # Intentionally use raw Q5 code: fixed wrapper repairs it.
        name=name,
        runtime="python310",
        memoryMB=128,
        CPUDemand=1,
    )
    print("Deployment result:", result)
    if isinstance(result, dict) and result.get("status", 200) >= 400:
        raise RuntimeError(f"Serverledge rejected the diagnostic function: {result}")

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        async with session.post(
            f"{base}/invoke/{name}",
            json={"Params": {"operations": [1, -2]}, "ReturnOutput": True},
        ) as response:
            body = await response.text()
            print("Invocation HTTP status:", response.status)
            print("Invocation body:", body[:6000])
            assert response.status == 200, "Inspect the Serverledge terminal log"
            payload = await response.json()
            assert payload.get("Success") is True

    print("PASS — repaired Python handler deployed and invoked with Success=true")
    print("Frozen Q5 Parquet unchanged.")


if __name__ == "__main__":
    asyncio.run(main())
