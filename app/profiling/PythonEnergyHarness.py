"""Harness Python per invocazioni JSON-lines nei container per la misura energetica."""

import importlib.util
import json
import math
import sys
import time
from contextlib import redirect_stdout


def load_function(path: str, entry: str):
    """Carica l'entry point una sola volta: l'import appartiene allo startup."""
    spec = importlib.util.spec_from_file_location("python_energy_function", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Sorgente Python non caricabile")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
    function = getattr(module, entry, None)
    if not callable(function):
        raise RuntimeError("Entry point Python inesistente o non richiamabile")
    return function



TIMER_RESULT_PREFIX = "__SPEC2FAAS_TIMER_V1__"


def timed_stream(function, *, seconds: float, max_invocations: int) -> int:
    """Ripete gli stessi input JSON fino alla durata richiesta; registra N effettivo.

    Il timer parte dopo l'import del modulo e la preparazione dei template.
    Ogni chiamata deserializza nuovamente gli argomenti e serializza l'output.
    """
    templates = [line for line in sys.stdin if line.strip()]
    if not templates:
        raise ValueError("Timer: almeno un caso JSON e' necessario")
    for template_index, line in enumerate(templates):
        request = json.loads(line)
        if request.get("index") != template_index or not isinstance(request.get("args"), list):
            raise ValueError("Timer: template JSON non valido")

    started = time.monotonic()
    count = 0
    for index in range(max_invocations):
        request = json.loads(templates[index % len(templates)])
        request["index"] = index
        with redirect_stdout(sys.stderr):
            value = function(*request["args"])
        sys.stdout.write(json.dumps({"index": index, "value": value},
                                    ensure_ascii=False, allow_nan=False) + "\n")
        count += 1
        if time.monotonic() - started >= seconds:
            break
    elapsed = time.monotonic() - started
    sys.stdout.flush()
    reason = "duration_reached" if elapsed >= seconds else "invocation_cap_reached"
    print(TIMER_RESULT_PREFIX + json.dumps({"invocations": count,
           "workload_seconds": elapsed, "stop_reason": reason}), file=sys.stderr)
    return 0


def main() -> int:
    """Esegue un evento per riga, replicando il protocollo Go di validazione."""
    timer_mode = len(sys.argv) == 7 and sys.argv[3] == "--timer-seconds" and sys.argv[5] == "--timer-max-invocations"
    if len(sys.argv) != 3 and not timer_mode:
        return 2
    try:
        function = load_function(sys.argv[1], sys.argv[2])
        if timer_mode:
            seconds = float(sys.argv[4])
            max_invocations = int(sys.argv[6])
            if not math.isfinite(seconds) or seconds <= 0 or max_invocations < 1:
                raise ValueError("Parametri Timer non validi")
            return timed_stream(function, seconds=seconds, max_invocations=max_invocations)
        for index, line in enumerate(sys.stdin):
            request = json.loads(line)
            if request["index"] != index or not isinstance(request["args"], list):
                raise ValueError("Indice o argomenti JSON invalidi")
            with redirect_stdout(sys.stderr):
                value = function(*request["args"])
            sys.stdout.write(json.dumps({"index": index, "value": value},
                                        ensure_ascii=False, allow_nan=False) + "\n")
        sys.stdout.flush()
        return 0
    except Exception as exc:
        print(f"PythonEnergyHarness: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
