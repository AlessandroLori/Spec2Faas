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


def profile_help(*, verbose: bool = False) -> str:
    """Elenca le varianti oppure mostra i prompt completi e le parti dinamiche."""
    brief = "\n".join(f"{group:10} {name:22} {description}"
                      for group, name, description in _DESCRIPTIONS)
    if not verbose:
        return brief

    # Import locali: Translator e Debugger importano a loro volta PromptProfiles.
    # Recuperiamo le costanti e i template reali, senza duplicare i system prompt.
    from .Translator import Translator
    from .Debugger import TranslationDebugger
    from .Messages import TranslationRequest
    from .DebugMessages import DebugRequest
    from .FeedbackProfiles import FeedbackProfiles

    translation_template = Translator.build_user_prompt(TranslationRequest(
        candidate_id="<TASK_ID>", source_code="<SORGENTE_PYTHON>",
        source_language="python", target_language="go", entry_point="<ENTRY_POINT>"))
    debug_template = TranslationDebugger._legacy_user_prompt(DebugRequest(
        candidate_id="<TASK_ID>", source_code="<SORGENTE_PYTHON>",
        current_code="<GO_CORRENTE>", entry_point="<ENTRY_POINT>",
        failure_kind="<CATEGORIA_ERRORE>", failure_details="<FEEDBACK_REALE>",
        attempt=1, failed_case_id="<ID_CASO_FALLITO>"))
    checkpoint_template = TranslationDebugger._prompt(DebugRequest(
        candidate_id="<TASK_ID>", source_code="<SORGENTE_PYTHON>",
        current_code="<GO_CORRENTE>", entry_point="<ENTRY_POINT>",
        failure_kind="<CATEGORIA_ERRORE>", failure_details="<FEEDBACK_REALE>",
        attempt=1, failed_case_id="<ID_CASO_FALLITO>"))

    sections = [
        "CATALOGO DEI PROFILI\n" + brief,
        "NOTA: le sezioni seguenti mostrano i prompt completi per linear; "
        "i segnaposto <...> indicano valori inseriti in esecuzione.",
    ]
    for name in TRANSLATOR_CHOICES:
        sections.append(f"TRANSLATOR system [linear, {name}]\n"
                        + translator_system_prompt(Translator.LEGACY_SYSTEM_PROMPT, name))
    sections.append("TRANSLATOR user [linear, tutti i profili; TEMPLATE]\n"
                    + translation_template)
    for name in DEBUGGER_CHOICES:
        sections.append(f"DEBUGGER system [linear, {name}]\n"
                        + debugger_system_prompt(TranslationDebugger.LEGACY_SYSTEM_PROMPT, name))
    sections.append("DEBUGGER user [linear, tutti i profili; TEMPLATE]\n" + debug_template)
    sections.append("FEEDBACK baseline [linear; REGOLE DINAMICHE]\n"
                    "Il feedback non e' un agente e non ha un system prompt. "
                    "DebugPipeline._feedback_legacy() usa esclusivamente gli errori verificati.\n"
                    "- traduzione assente: 'Invalid Go translation: <MOTIVO>'\n"
                    "- compilazione/runtime falliti: testo dell'errore reale\n"
                    "- confronto fallito: 'Validation: <ESITO>.', 'Failed case: <ID>.', "
                    "errore, input JSON, output atteso e osservato e stato del caso.\n"
                    "Tutto il feedback appare dentro <feedback> nello user prompt del Debugger.")
    sections.append("FEEDBACK feedback-v1 [linear; APPENDICE CONDIZIONALE]\n"
                    "Aggiunge al feedback baseline, solo se un'altra firma di errore distinta "
                    "e' stata verificata nell'esecuzione Go corrente:\n"
                    "ADDITIONAL DISTINCT FAILED CASE (current Go revision; verified):\n"
                    "Case ID: <ID>\nInput arguments: <JSON>\nExpected output: <JSON>\n"
                    "Actual output: <JSON>\nExecution status: <STATO>")
    sections.append("FEEDBACK feedback-history-v1 [linear; APPENDICE CONDIZIONALE]\n"
                    "Aggiunge al feedback baseline un caso verificato relativo alla cronologia:\n"
                    "REGRESSION (passed in an earlier tested revision, FAILED now):\n"
                    "<CASO_VERIFICATO>\nOPPURE\n"
                    "PREVIOUSLY FIXED CASE (do not regress this behavior):\n"
                    "Case ID: <ID>\nInput arguments: <JSON>\nExpected output: <JSON>\n"
                    "Status: PASSED in the CURRENT tested Go revision.\n"
                    f"Massimo {FeedbackProfiles.MAX_ADDITION_CHARS} caratteri per l'estratto di caso.")
    sections.append("CHECKPOINT [strategia checkpoint; profili solo baseline]\n"
                    "Translator system:\n" + Translator.SYSTEM_PROMPT +
                    "\nDebugger system:\n" + TranslationDebugger.SYSTEM_PROMPT +
                    "\nDebugger user [TEMPLATE]:\n" + checkpoint_template +
                    "\nNota: con dipendenze esterne abilitate, i system prompt di checkpoint "
                    "sono adattati dai rispettivi costruttori. In ogni esecuzione reale "
                    "gli esatti system/user prompt sono salvati negli artefatti.")
    return "\n\n".join(sections)


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
