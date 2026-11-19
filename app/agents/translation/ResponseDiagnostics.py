"""Metadati di una risposta LLM, senza conservare il contenuto reasoning."""

from collections.abc import Mapping


def response_diagnostics(response) -> dict:
    """Legge campi MAF disponibili, senza inventare conteggi o finish reason."""
    usage = getattr(response, "usage_details", None)
    def field(name):
        if isinstance(usage, Mapping):
            value = usage.get(name)
        else:
            value = getattr(usage, name, None)
        return value if type(value) is int and value >= 0 else None

    reasoning_chars = 0
    reasoning_present = False
    for message in getattr(response, "messages", ()) or ():
        for content in getattr(message, "contents", ()) or ():
            if getattr(content, "type", None) == "text_reasoning":
                reasoning_present = True
                reasoning_chars += len(getattr(content, "text", "") or "")

    reason = getattr(response, "finish_reason", None)
    return {
        "input_tokens": field("input_token_count"),
        "output_tokens": field("output_token_count"),
        "finish_reason": (str(reason) if reason is not None else None),
        "reasoning_present": reasoning_present,
        "reasoning_characters": reasoning_chars,
        "final_text_characters": len(getattr(response, "text", "") or ""),
    }
