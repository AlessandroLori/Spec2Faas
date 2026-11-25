import contextlib
import copy
import json
import os
import sys


def _error(exc: BaseException) -> dict:
    """Restituisce un errore sintetico senza includere il codice eseguito."""
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc)[:400],
    }


def _equivalent_refaas(actual, expected) -> bool:
    """Confronta il body come JSON, non come ordine lessicale delle chiavi."""
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        return actual == expected
    actual = actual.copy()
    expected = expected.copy()
    if isinstance(actual.get("body"), str) and isinstance(expected.get("body"), str):
        try:
            actual["body"] = json.loads(actual["body"])
            expected["body"] = json.loads(expected["body"])
        except ValueError:
            pass
    return actual == expected


def _run(payload: dict) -> dict:
    """Esegue i casi utilizzando l'asserzione originale HumanEval+."""
    namespace = {"__name__": "__redataset_reference__",
                 "__file__": os.path.join(os.getcwd(), "function.py")}
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                exec(compile(payload["code"], "<reference.py>", "exec"), namespace)
                if payload.get("comparator") not in ("refaas_expected_json", "sebs_original_validate_output", "sebs_derived_explicit_json"):
                    exec(compile(payload["tests"], "<tests.py>", "exec"), namespace)
        function = namespace[payload["entry_point"]]
        assertion = namespace.get("assertion")
        if not callable(function) or (payload.get("comparator") == "humaneval_plus_assertion"
                                      and not callable(assertion)):
            raise TypeError("Entry point o comparatore non richiamabile")
        if payload.get("comparator") == "sebs_original_validate_output":
            tests_namespace = {"__name__": "__sebs_original_validator__"}
            exec(compile(payload["tests"], "<sebs_input.py>", "exec"), tests_namespace)
            if not callable(tests_namespace.get("validate_output")):
                raise TypeError("Validatore SeBS originale mancante")
    except BaseException as exc:
        return {"status": "setup_error", "checked_cases": 0, **_error(exc)}

    checked = 0
    for case in payload["cases"]:
        try:
            # Ogni invocazione riceve una copia indipendente degli argomenti.
            args = copy.deepcopy(case["args"])
            expected = copy.deepcopy(case["expected"])
            with open(os.devnull, "w", encoding="utf-8") as sink:
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    value = function(*args)
                    if payload.get("comparator") == "sebs_original_validate_output":
                        feedback = tests_namespace["validate_output"](
                            None, args[0], value, "python")
                        if feedback is not None:
                            raise AssertionError(str(feedback))
                    elif payload.get("comparator") == "refaas_expected_json":
                        if not _equivalent_refaas(value, expected):
                            raise AssertionError("Output ReFaaS diverso dall'atteso")
                    elif payload.get("comparator") == "sebs_derived_explicit_json":
                        if (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
                                != json.dumps(expected, sort_keys=True, ensure_ascii=False, allow_nan=False)):
                            raise AssertionError("Output SeBS diverso dai test DERIVATI (non originali)")
                    else:
                        assertion(value, expected, case["atol"])
        except BaseException as exc:
            return {
                "status": "failed",
                "checked_cases": checked,
                "failed_case_id": case["case_id"],
                **_error(exc),
            }
        checked += 1

    return {"status": "passed", "checked_cases": checked}


if __name__ == "__main__":
    try:
        result = _run(json.load(sys.stdin))
    except BaseException as exc:
        result = {"status": "setup_error", "checked_cases": 0, **_error(exc)}
    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
