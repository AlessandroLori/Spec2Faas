"""Varianti sperimentali indipendenti dei prompt Python->Go.

La baseline e' un'identita': nessuna trasformazione del prompt storico.
Le varianti sono versionate e non modificano strategie, test o codice Go.
"""

TRANSLATOR_CHOICES = ("baseline", "format-v1")
DEBUGGER_CHOICES = ("baseline", "semantics-v1")
FEEDBACK_CHOICES = ("baseline", "feedback-v1", "feedback-history-v1")

FORMAT_V1 = (
    "\n\nMANDATORY GO OUTPUT CONTRACT (check before replying):\n"
    "1. Return exactly one COMPLETE Go source file, not only a function snippet.\n"
    "2. Its first non-comment, non-empty line MUST be: package main\n"
    "3. Declare the exact required entry-point function identifier.\n"
    "4. Do NOT define func main(); the harness supplies it.\n"
    "5. Include only actually used imports; omit the import declaration if unnecessary.\n"
    "6. Return the complete source in exactly one ```go fenced block, without prose.\n"
    "A function snippet without package main is INVALID, even if its logic is correct."
)

SEMANTICS_V1 = (
    "\n\nPYTHON SEMANTIC COMPATIBILITY (apply when relevant to the observed failure):\n"
    "- The Python source and original-test oracle define the required behavior.\n"
    "- In Python, bool is a subclass of int and can participate in integer arithmetic.\n"
    "- Python integers may exceed the fixed ranges of Go int and int64.\n"
    "- Preserve input/output types, edge cases, and observable error behavior.\n"
    "- Correct the reported failure without breaking already correct behavior.\n"
    "- Do not invent requirements unsupported by the Python source or test evidence."
)


_DESCRIPTIONS = (
    ("Translator", "baseline", "Prompt legacy della campagna Balanced90 (linear)."),
    ("Translator", "format-v1", "Contratto esplicito di sorgente Go completo e package main."),
    ("Debugger", "baseline", "Prompt legacy di debugging LLM (linear)."),
    ("Debugger", "semantics-v1", "Attenzione mirata alla semantica Python/Go."),
    ("Feedback", "baseline", "Primo fallimento reale, senza cronologia."),
    ("Feedback", "feedback-v1", "Primo fallimento + massimo uno distinto e reale, corrente."),
    ("Feedback", "feedback-history-v1", "Primo fallimento + massimo un caso storico verificato."),
)


def profile_help() -> str:
    """Testo auto-documentante per --list-prompt-profiles."""
    return "\n".join(f"{group:10} {name:22} {description}" for group, name, description in _DESCRIPTIONS)


def validate_profiles(*, translator: str, debugger: str, feedback: str,
                      strategy: str = "linear") -> None:
    """Rifiuta configurazioni inesistenti o applicate al checkpoint congelato."""
    for label, value, choices in (
        ("--prompt-translator", translator, TRANSLATOR_CHOICES),
        ("--prompt-debugger", debugger, DEBUGGER_CHOICES),
        ("--prompt-feedback", feedback, FEEDBACK_CHOICES),
    ):
        if value not in choices:
            raise ValueError(f"{label}: '{value}' non valido; disponibili: {', '.join(choices)}")
    if strategy != "linear" and any(v != "baseline" for v in (translator, debugger, feedback)):
        raise ValueError("I prompt sperimentali sono disponibili solo con --debug-strategy linear; "
                         "checkpoint rimane invariato")


def translator_system_prompt(base: str, variant: str) -> str:
    if variant not in TRANSLATOR_CHOICES:
        raise ValueError("Profilo Translator non valido")
    return base if variant == "baseline" else base + FORMAT_V1


def debugger_system_prompt(base: str, variant: str) -> str:
    if variant not in DEBUGGER_CHOICES:
        raise ValueError("Profilo Debugger non valido")
    return base if variant == "baseline" else base + SEMANTICS_V1
