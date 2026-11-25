"""Worker isolato per reference Python con storage provider reale.

Esegue il generatore e validatore ORIGINALI, senza alterare i sorgenti SeBS.
Una fixture verificata puo' sostituire esclusivamente un URL di download:
non intercetta ne' modifica le richieste boto3 al provider emulato.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import os
import sys
import urllib.request
from pathlib import Path
from unittest.mock import patch


SOURCE_ROOT = Path("/opt/reliq-source")


def _error(exc: BaseException) -> dict:
    return {"status": "setup_error", "checked_cases": 0,
            "error_type": type(exc).__name__, "error_message": str(exc)[:500]}


def _run(payload: dict) -> dict:
    if not isinstance(payload, dict) or payload.get("protocol") != "sebs_provider_reference_v1":
        raise ValueError("Protocollo della reference provider non supportato")
    importlib.invalidate_caches()
    sys.path.insert(0, str(SOURCE_ROOT))
    # Configurazione SDK standard, senza modificare il wrapper AWS originale:
    # forziamo indirizzamento S3 path-style per il DNS della rete Docker.
    sdk_config = SOURCE_ROOT / "aws_config.ini"
    if sdk_config.is_file():
        os.environ["AWS_CONFIG_FILE"] = str(sdk_config)
    generator = importlib.import_module("benchmark_input")
    args = payload.get("generator_args")
    if not isinstance(args, list):
        raise ValueError("Argomenti del generatore SeBS mancanti")
    event = generator.generate_input(*args)
    if not isinstance(event, dict):
        raise TypeError("Il generatore originale non ha restituito un evento JSON")
    if "expected_event" in payload and event != payload["expected_event"]:
        raise ValueError("Input del generatore originale diverso dalla suite provider attestata")
    mode = payload.get("mode", "reference")
    if mode not in ("reference", "validate"):
        raise ValueError("Modalita' worker provider non supportata")
    if mode == "reference":
        handler_module = importlib.import_module(payload["handler_module"])
        handler = getattr(handler_module, payload["entry_point"])
    storage_module = importlib.import_module(payload["storage_module"])
    storage = getattr(storage_module, "storage").get_instance()
    # Fallire esplicitamente se un wrapper boto3 non viene indirizzato a Floci.
    # L'isolamento di rete impedirebbe comunque l'accesso ad AWS, ma il
    # controllo rende diagnosticabile una configurazione SDK errata.
    local_s3 = os.environ.get("AWS_ENDPOINT_URL_S3")
    client = getattr(storage, "client", None)
    if local_s3 and client is not None and hasattr(client, "meta"):
        actual_endpoint = getattr(client.meta, "endpoint_url", "")
        if actual_endpoint.rstrip("/") != local_s3.rstrip("/"):
            raise ValueError(
                f"Endpoint S3 del wrapper non emulato: {actual_endpoint!r}"
            )

    fixture_name = payload.get("fixture_file")
    if fixture_name is not None:
        fixture_file = SOURCE_ROOT / fixture_name
        if fixture_file.resolve().parent != SOURCE_ROOT.resolve():
            raise ValueError("File fixture non valido")
        fixture_bytes = fixture_file.read_bytes()
        if hashlib.sha256(fixture_bytes).hexdigest() != payload["fixture_sha256"]:
            raise ValueError("Hash della fixture HTTP diverso dall'atteso")
        url = payload["fixture_url"]
        # L'evento deve mantenere esattamente l'URL generato da input.py.
        if event.get("object", {}).get("url") != url:
            raise ValueError("La fixture non corrisponde all'URL del generatore originale")

        def offline_urlopen(request, *args, **kwargs):
            actual = request.full_url if isinstance(request, urllib.request.Request) else request
            if actual != url:
                raise RuntimeError("Download HTTP non previsto nella reference isolata")
            return io.BytesIO(fixture_bytes)

        request_context = patch("urllib.request.urlopen", side_effect=offline_urlopen)
    else:
        # Niente rete pubblica disponibile nel container: download non consentiti.
        request_context = contextlib.nullcontext()

    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            if mode == "reference":
                with request_context:
                    output = handler(event)
            else:
                output = payload.get("output")
                if not isinstance(output, dict):
                    raise TypeError("Output Go da validare non valido")
            # Il validatore usa il client storage ORIGINALE, non None.
            # Il parametro language riproduce l'oracolo Python originale.
            try:
                feedback = generator.validate_output(None, event, output, "python", storage)
            except Exception as exc:
                # Un oggetto assente e' un errore del risultato Go, non
                # un guasto di Floci. Errori di connessione/configurazione
                # restano setup_error e non consumano tentativi LLM.
                response = getattr(exc, "response", None)
                error = response.get("Error", {}) if isinstance(response, dict) else {}
                code = error.get("Code") if isinstance(error, dict) else None
                if mode == "validate" and code in ("NoSuchKey", "404", "NotFound"):
                    return {"status": "failed", "checked_cases": 0,
                            "error_type": type(exc).__name__,
                            "error_message": "Oggetto richiesto dal Go assente nello storage S3"}
                raise
    if feedback is not None:
        return {"status": "failed", "checked_cases": 0,
                "error_type": "OriginalSeBSValidator", "error_message": str(feedback)[:500]}
    if not isinstance(output, dict) or not isinstance(output.get("result"), dict):
        raise TypeError("Output della reference non JSON")
    result = output["result"]
    return {"status": "passed", "checked_cases": 1,
            "result_key": result.get("key"), "result_bucket": result.get("bucket"),
            "result_url": result.get("url"), "storage_checked": True}


def main() -> None:
    try:
        answer = _run(json.load(sys.stdin))
    except BaseException as exc:
        answer = _error(exc)
    json.dump(answer, sys.stdout, ensure_ascii=False, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
