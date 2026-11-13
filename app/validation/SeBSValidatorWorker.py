"""Esegue esclusivamente il validatore originale SeBS in un processo separato.

Il chiamante verifica preventivamente l'hash del codice originale; il processo
separato non e' un sandbox di sicurezza per codice non fidato.
Il valore None di validate_output significa PASS secondo il protocollo SeBS.
"""

import contextlib
import json
import os
import sys


def _check(payload: dict) -> dict:
    if (not isinstance(payload, dict)
            or not isinstance(payload.get("test_code"), str)
            or not isinstance(payload.get("event"), dict)
            or not isinstance(payload.get("output"), dict)
            or payload.get("language") not in ("python", "go")):
        raise ValueError("Payload del validatore SeBS non valido")

    namespace = {"__name__": "__sebs_original_validator__"}
    # Il validatore puo' stampare diagnostica: non deve contaminare lo stdout
    # strutturato del protocollo utilizzato dal comparatore.
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(compile(payload["test_code"], "<sebs_input.py>", "exec"), namespace)
            validate = namespace.get("validate_output")
            if not callable(validate):
                raise TypeError("Validatore originale SeBS non richiamabile")
            feedback = validate(None, payload["event"], payload["output"],
                                payload["language"])
    if feedback is None:
        return {"valid": True}
    # Un risultato diverso da None e' un fallimento della verifica originale.
    return {"valid": False, "message": str(feedback)[:400]}


if __name__ == "__main__":
    try:
        answer = _check(json.load(sys.stdin))
    except BaseException as exc:
        answer = {"error_type": type(exc).__name__, "error_message": str(exc)[:400]}
    json.dump(answer, sys.stdout, ensure_ascii=False, allow_nan=False)
    sys.stdout.write("\n")
