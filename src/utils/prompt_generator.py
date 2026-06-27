"""
AI Prompt Generator.

Turns a rough idea (written casually OR formally) into a clean, structured prompt
for an LLM, using a selectable prompt framework, plus an optional critique-and-
refine pass that scores and improves the draft.

Rule basis: the openclaw prompt-quality rubric (Prompt-Input-Mock-QC.md Part A) plus
a mined set of transferable prompt-engineering techniques. The generator enforces
three layers:

  STRUCTURE (A.1)  role -> context -> input -> task -> constraints -> exact output
                   format, with an objective, verifiable success condition.
  CLARITY (A.3)    exact thresholds, explicit conditionals, clear scope, atomic
                   one-ask-per-line requirements, affirmative phrasing, no hollow
                   quality words ('correctly', 'properly'), no escape-hatches, and
                   no over-prescribed output format the idea never asked for.
  HYGIENE (A.4)    zero em dash / en-dash-as-punctuation / smart quotes, no AI-tell
                   or filler words, no placeholder tokens.

Hygiene is enforced twice: as generation instructions AND as a deterministic
post-processing pass (_sanitize) that GUARANTEES typographic compliance regardless
of the model. The sanitizer is region-aware: it masks out fenced code blocks and
inline-code spans before rewriting, so backticked code is preserved exactly.
(Unfenced code is treated as prose; the generation rules tell the model to fence code.)

Short ideas are expanded with gated conditional clauses (temporal anchoring,
report-missing/do-not-fabricate, numeric precision, choice tie-breakers) so a terse
input still yields a complete prompt. The refine pass runs a bounded, fail-loud
repair loop and re-sanitizes every rewrite. Routes through the shared Groq/Ollama
gateway; degrades gracefully with no AI backend.
"""

import re
import datetime
from .gemini_helper import generate_text, is_unavailable_response

# Selectable prompt frameworks → the structure the generated prompt should follow.
PROMPT_FRAMEWORKS = {
    "R-T-F (Role · Task · Format)": (
        "Structure the prompt in three clear parts: (1) ROLE — the expertise the AI "
        "should adopt; (2) TASK — the specific job to do; (3) FORMAT — the exact "
        "output format expected."
    ),
    "CRISPE": (
        "Structure as CRISPE: Capacity/Role, Insight (context & background), "
        "Statement (the precise task), Personality (tone/style of the response), and "
        "Experiment (ask for one or two alternative versions)."
    ),
    "Chain-of-Thought": (
        "Structure so the AI reasons before answering: a clear role, the task, an "
        "explicit instruction to think through the problem step by step, then the "
        "final answer in a stated output format."
    ),
    "Few-shot (with examples)": (
        "Structure with a role, the task, one or two short illustrative input→output "
        "examples that demonstrate the desired pattern, then the actual request and "
        "the output format."
    ),
    "Contract (deliverable + acceptance)": (
        "Structure as an explicit contract with these headed sections: ROLE, "
        "OBJECTIVE (the deliverable), CONTEXT/INPUTS, CONSTRAINTS (hard rules), "
        "OUTPUT FORMAT (exact), and ACCEPTANCE CRITERIA (how to tell it was done well)."
    ),
}

TARGET_MODELS = ["Any LLM", "Claude", "ChatGPT (GPT)", "Gemini", "Llama / Groq"]

# --- Shared banned-term constants (define-once: used by BOTH the generation -------
# --- instructions and the deterministic detector, so the two can never drift) -----

# Escape-hatch phrases that let the model wriggle out of a requirement (QC A.3).
_ESCAPE_HATCHES = (
    "if applicable", "as needed", "where possible", "when possible",
    "use your best judgment", "use your discretion", "if you can", "infer if necessary",
    "as appropriate",
)

# "Hollow quality" adverbs that sound like a constraint but specify nothing testable.
# Detection uses only this high-precision subset (rarely precise as a bare modifier);
# the generation instruction warns against the broader family too.
_HOLLOW_ADVERBS = (
    "correctly", "properly", "appropriately", "adequately", "sufficiently",
    "suitably", "accurately", "thoroughly", "effectively", "consistently", "plainly",
)
# Broader family mentioned to the model (not deterministically flagged, since several
# are legitimate next to a concrete number, e.g. "exactly 5").
_HOLLOW_ADVERBS_SOFT = (
    "exactly", "fully", "completely", "clearly", "reasonably", "relevant",
    "mostly", "typically", "often", "roughly", "generally", "usually",
    "preferably", "ideally",
)

# Vague / unquantified amount words (QC A.3 thresholds rule).
_VAGUE_QUANTIFIERS = ("a few", "approximately", "roughly", "around", "a number of", "several")

# Banned AI-tell + filler words/phrases (QC A.4). Verb stems carry their common
# inflections (-e/-es/-ed/-ing). "landscape" lives only in the generation instruction
# (a literal/geographic "landscape" is legitimate and a blind match would misfire).
_BANNED_RE = re.compile(
    r"\b(?:"
    r"it'?s important to note|it is important to note|it'?s worth noting|"
    r"it is worth noting|this ensures|this allows|in order to|needless to say|"
    r"it should be noted|as previously mentioned|moving forward|essentially|"
    r"basically|fundamentally|arguably|comprehensive(?:ly)?|"
    r"leverag(?:e|es|ed|ing)|utiliz(?:e|es|ed|ing)|delv(?:e|es|ed|ing)|"
    r"facilitat(?:e|es|ed|ing)|streamlin(?:e|es|ed|ing)"
    r")\b",
    re.I,
)

# Placeholder / fill-in tokens (QC A.4). Tightened to genuine placeholder shapes:
# UPPER-CASE angle/brace tokens, ${VAR}, TBD/XXX, [INSERT...]. It deliberately does
# NOT match real HTML/XML tags (<html>), generics (List<String>), comparison prose
# ('a < b'), or lowercase template tokens ({json}).
_PLACEHOLDER_RE = re.compile(
    r"<[A-Z][A-Z0-9_]{1,38}>"        # <TOKEN>, <FILE_NAME>
    r"|\$\{[^}]{1,40}\}"             # ${VAR}
    r"|\{[A-Z][A-Z0-9_]{1,38}\}"    # {VALUE}
    r"|\bTBD\b|\bXXX\b"             # TBD / XXX
    r"|\[(?i:insert)[^\]]*\]"       # [INSERT ...]
)
# Standalone literal "..." used as a fill-in (anchored both sides so code spread is safe).
_LITERAL_ELLIPSIS_RE = re.compile(r"(?:^|\s)\.\.\.+(?:\s|$)")

# Relative-time words that need anchoring to a concrete date.
_RELATIVE_TIME_RE = re.compile(
    r"\b(recent(?:ly)?|lately|soon|latest|current(?:ly)?|now|today|yesterday|tomorrow|"
    r"upcoming|expired|this (?:week|month|year|quarter)|last (?:week|month|year|quarter)|"
    r"next (?:week|month|year|quarter))\b",
    re.I,
)

# Intent keyword sets that gate the conditional clauses.
_EXTRACTION_KW = (
    "extract", "find", "look up", "lookup", "analyze", "analyse", "audit", "pull",
    "identify", "summarize", "summarise", "review", "search", "retrieve", "parse",
)
_NUMERIC_KW = (
    "average", "total", "sum", "percentage", "percent", "ratio", "rate", "count",
    "mean", "median", "growth", "metric", "number of", "how many", "calculate", "compute",
)
_CHOICE_KW = (
    "classify", "categorize", "categorise", "rate", "score", "rank", "pick", "select",
    "choose", "label", "sort into", "bucket", "tag each",
)


def _today() -> str:
    return datetime.date.today().isoformat()


def _build_clarity_rules() -> str:
    return (
        "CLARITY (write a CRISP prompt two careful readers would interpret identically):\n"
        "- Be specific and concrete. Resolve vagueness by making reasonable, explicit choices.\n"
        "- State exact thresholds and quantities; never 'a few', 'some', 'several', or 'approximately'.\n"
        "- Make any conditional logic explicit (if X then Y, otherwise Z).\n"
        "- Define scope boundaries (which items, which range, which cases).\n"
        "- Give an objective, verifiable success condition and state what 'done' looks like; do not "
        "settle for subjective goals like 'write a good summary' or 'give your thoughts'.\n"
        "- Make each requirement atomic: one obligation per line. Split lines that join obligations "
        "with 'and', 'while', or 'including' into separate lines.\n"
        "- Name referents instead of bare pronouns (avoid a stray 'it', 'they', 'this' whose subject "
        "sits in an earlier sentence).\n"
        "- Phrase format, length, and style requirements as positive targets ('limit to 200 words or "
        "fewer') rather than negations, while keeping genuine prohibitions.\n"
        "- Do NOT use hollow quality words that specify nothing testable (" + ", ".join(
            _HOLLOW_ADVERBS[:6]) + ", " + ", ".join(_HOLLOW_ADVERBS_SOFT[:4]) + "); replace each with "
        "the concrete, checkable observable it implies (for example 'accurately' becomes 'every figure "
        "matches the source'). A word like 'exactly' is fine only next to a concrete number.\n"
        "- Do NOT use escape-hatch phrases (" + ", ".join(_ESCAPE_HATCHES[:5]) + ").\n"
        "- Include only constraints that change what counts as a correct output; drop any that restate "
        "or contradict another, and omit a section the idea gives no material for (do not pad).\n"
        "- Only specify concrete output-format details (column names, section titles, filenames) that "
        "trace to the idea or the chosen framework; otherwise give a general shape and invent nothing.\n"
        "- Keep it self-contained: it must make sense without seeing this instruction.\n"
        "- Address the AI in the second person ('You are...', 'Write...')."
    )


def _build_hygiene_rules() -> str:
    return (
        "HYGIENE (hard rules, zero tolerance):\n"
        "- Use NO em dashes and NO en dashes as punctuation; use commas, periods, or parentheses. "
        "(An en dash is allowed only inside a number range like 2020-2024.)\n"
        "- Use straight quotes ' and \" only; no curly/smart quotes and no ellipsis character.\n"
        "- Do NOT use these AI-tell words/phrases: it's important to note, this ensures, this allows, "
        "delve, leverage, comprehensive, streamline, utilize, facilitate, in order to, needless to say, "
        "it should be noted, as previously mentioned, moving forward, landscape (used metaphorically). "
        "Prefer plain words (use, not utilize; to, not in order to).\n"
        "- No filler or hedging: essentially, basically, fundamentally, arguably.\n"
        "- No placeholder or fill-in tokens (<TOKEN>, {VALUE}, ${VAR}, XXX, TBD, [INSERT...], '...'); "
        "make a concrete choice instead."
    )


# QC Part A.1: every prompt must carry this skeleton in order, whatever the framework's labels.
_STRUCTURE_RULES = (
    "STRUCTURE (include all of these, in this order, whatever labels the framework uses):\n"
    "1. Persona/role — open by giving the AI a specific expert role.\n"
    "2. Background/context — the situation and any domain context it needs.\n"
    "3. Input — what the AI will be given or should assume as its input.\n"
    "4. Task — the precise, single deliverable or action to perform.\n"
    "5. Constraints — hard rules, scope boundaries, and what to avoid.\n"
    "6. Output format — the exact shape of the expected output."
)

_SELF_CHECK_TAIL = (
    "\n\nBefore you finalize, list each requirement above and confirm your response satisfies it. "
    "Revise any gaps, then deliver only the final result."
)


def _mode_instruction(mode: str) -> str:
    if (mode or "").lower().startswith("proc"):
        return (
            "MODE (procedure-oriented): the user wants the method spelled out. Include explicit, "
            "ordered steps the AI should follow to reach the deliverable."
        )
    return (
        "MODE (outcome-oriented): specify WHAT the result must be and its acceptance criteria. Do not "
        "script a step-by-step method; let the AI choose its own approach unless the framework requires "
        "step-by-step reasoning."
    )


def _conditional_clauses(idea: str, task_type: str) -> list:
    """Gated clauses that expand a short idea into a complete prompt.

    Each is appended only when the idea's intent calls for it, so a terse input still
    yields temporal anchoring, missing-input handling, numeric precision, or a choice
    tie-breaker as appropriate. Mis-triggering only adds a sensible extra instruction.
    """
    text = f"{idea or ''} {task_type or ''}".lower()
    clauses = []
    if _RELATIVE_TIME_RE.search(text):
        clauses.append(
            f"The request uses relative time references. Anchor them to a concrete date. Treat today "
            f"as {_today()} and convert words like 'recent', 'latest', or 'now' into an explicit date "
            f"or range rather than leaving them floating."
        )
    if _kw_hit(text, _EXTRACTION_KW):
        clauses.append(
            "If a value the task needs is not present in the provided input, state that it is missing. "
            "Do not infer, guess, or invent it."
        )
    if _kw_hit(text, _NUMERIC_KW):
        clauses.append(
            "For any numeric result, state the units, the number of decimal places, and the rounding "
            "rule. Carry full precision through intermediate steps and round only the final value."
        )
    if _kw_hit(text, _CHOICE_KW):
        clauses.append(
            "If the task assigns items to categories or rates them, give a default for the residual "
            "case where no option cleanly applies or two apply equally (for example, choose the stricter)."
        )
    return clauses


# --- Deterministic scanning / sanitizing ------------------------------------------

def _wb_matches(low: str, terms) -> list:
    """Whole-word/phrase matches of `terms` within already-lowercased `low`.

    Word-boundary matching avoids substring false positives like 'around' inside
    'turnaround', 'roughly' inside 'thoroughly', or 'if you can' inside 'if you cannot'.
    """
    return sorted({t for t in terms if re.search(r"\b" + re.escape(t) + r"\b", low)})


def _kw_hit(text: str, terms) -> bool:
    """True if any term appears as a whole word/phrase in `text` (same boundary safety)."""
    return any(re.search(r"\b" + re.escape(t) + r"\b", text) for t in terms)


def _scan(text: str) -> dict:
    """Categorize hygiene/clarity violations in `text` (the source of truth for both
    the displayed issues and the repair loop)."""
    t = text or ""
    low = t.lower()
    vague = set(_wb_matches(low, _VAGUE_QUANTIFIERS))
    if re.search(r"\bsome\b", low):
        vague.add("some")
    # A space-flanked en dash is banned punctuation EXCEPT inside a numeric range
    # (2020 - 2024), which the QC explicitly permits; drop ranges before testing.
    no_ranges = re.sub(r"\d\s*–\s*\d", "", t)
    return {
        "em_dash": "—" in t,
        "en_dash": bool(re.search(r"\s–|–\s", no_ranges)),
        "ai_tells": sorted({m.group(0).lower() for m in _BANNED_RE.finditer(low)}),
        "placeholders": bool(_PLACEHOLDER_RE.search(t) or _LITERAL_ELLIPSIS_RE.search(t)),
        "escape_hatches": _wb_matches(low, _ESCAPE_HATCHES),
        "hollow": _wb_matches(low, _HOLLOW_ADVERBS),
        "vague": sorted(vague),
    }


def _hygiene_issues(text: str) -> list:
    """Human-readable issue list for the refine pass (built from _scan)."""
    s = _scan(text)
    issues = []
    if s["em_dash"]:
        issues.append("Contains em dash(es); replace with commas or periods.")
    if s["en_dash"]:
        issues.append("Contains en dash used as punctuation; use commas or periods "
                      "(an en dash is fine only inside a number range).")
    if s["ai_tells"]:
        issues.append("Uses AI-tell / filler words: " + ", ".join(s["ai_tells"][:8]) + ".")
    if s["escape_hatches"]:
        issues.append("Uses escape-hatch phrasing: " + ", ".join(s["escape_hatches"][:8])
                      + "; make a concrete choice.")
    if s["hollow"]:
        issues.append("Uses hollow quality words: " + ", ".join(s["hollow"][:8])
                      + "; replace each with the concrete, checkable observable it implies.")
    if s["placeholders"]:
        issues.append("Contains placeholder tokens; replace them with concrete values.")
    if s["vague"]:
        issues.append("Vague / unquantified phrasing: " + ", ".join(s["vague"][:8])
                      + "; state exact amounts.")
    return issues


def _hard_terms(text: str) -> list:
    """Zero-tolerance violations that justify a repair pass (soft issues like hollow
    adverbs and vague quantifiers are reported but never force a rewrite)."""
    s = _scan(text)
    terms = []
    if s["em_dash"]:
        terms.append("em dash")
    if s["en_dash"]:
        terms.append("en dash used as punctuation")
    terms += s["ai_tells"]
    if s["placeholders"]:
        terms.append("placeholder token")
    terms += s["escape_hatches"]
    return sorted(set(terms))


def _sanitize_prose(t: str) -> str:
    """Typographic normalization for a non-code text segment (no final strip)."""
    t = re.sub(r"[   ]", " ", t)          # nbsp / thin spaces
    t = re.sub(r"[‘’‚‛]", "'", t)     # smart single quotes / apostrophes
    t = re.sub(r"[“”„‟]", '"', t)     # smart double quotes

    # Unicode ellipsis (U+2026): drop at a line edge, else collapse to '. '.
    t = re.sub(r"(?m)^[ \t]*…[ \t]*", "", t)
    t = re.sub(r"(?m)[ \t]*…[ \t]*$", ".", t)
    t = re.sub(r"[ \t]*…[ \t]*", ". ", t)
    # Literal ASCII ellipsis used as a fill-in -> period (kept in lockstep with the
    # placeholder detector). It only fires when followed by whitespace/end, so code
    # spread like "...args" is left alone, and real code spans are masked out first.
    t = re.sub(r"[ \t]?\.\.\.+(?=\s|$)", ".", t)

    # Em dash (banned everywhere): drop at a line edge, else -> comma.
    t = re.sub(r"(?m)^[ \t]*—[ \t]*", "", t)
    t = re.sub(r"(?m)[ \t]*—[ \t]*$", "", t)
    t = re.sub(r"[ \t]*—[ \t]*", ", ", t)

    # En dash: a numeric range (2020-2024 or 2020 - 2024) -> ASCII hyphen; a closed-up
    # word compound (cost–benefit) -> hyphen; space-flanked punctuation -> comma.
    t = re.sub(r"(?<=\d)[ \t]*–[ \t]*(?=\d)", "-", t)
    t = re.sub(r"(?<=[A-Za-z])–(?=[A-Za-z])", "-", t)
    t = re.sub(r"(?m)^[ \t]*–[ \t]+", "", t)
    t = re.sub(r"(?m)[ \t]+–[ \t]*$", "", t)
    t = re.sub(r"[ \t]+–[ \t]+", ", ", t)

    # Drop trailing horizontal whitespace a substitution may have left on a line.
    t = re.sub(r"[ \t]+(?=\n)", "", t)
    return t


def _sanitize(text: str) -> str:
    """Apply the deterministic QC A.4 typographic bans, but ONLY to prose regions.

    Fenced code blocks (```...```) and inline-code spans (`...`) are MASKED out before
    rewriting and restored verbatim afterward, so a literal code sample is preserved
    exactly. Masking (rather than splitting + classifying by a leading backtick) means
    an unbalanced or fused backtick can never cause a prose region to be skipped: any
    text that is not a balanced code span is sanitized as prose. Guarantees zero em
    dashes / en-dash-as-punctuation / smart quotes in every prose region.
    """
    text = text or ""
    spans = []

    def _mask(m):
        spans.append(m.group(0))
        return f"\x00{len(spans) - 1}\x00"

    masked = re.sub(r"```.*?```|`[^`\n]+`", _mask, text, flags=re.S)
    cleaned = _sanitize_prose(masked)
    cleaned = re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], cleaned)
    return cleaned.strip()


def _strip_fences(text: str) -> str:
    """Unwrap markdown code fences ONLY when the whole string is one fenced block.

    (A blanket trailing-``` strip would eat the closing fence of a legitimate
    example block inside the prompt — common for the Few-shot/Contract frameworks.)
    """
    t = (text or "").strip()
    m = re.match(r"^```[a-zA-Z0-9]*\s*\n(.*)\n```$", t, re.S)
    return m.group(1).strip() if m else t


def _strip_preamble(text: str) -> str:
    """Drop a leading conversational lead-in line (e.g. 'Sure! Here is the prompt:')."""
    return re.sub(
        r"^(sure|here(?:\s+is|'s| you go)|certainly|okay|of course|absolutely)[^\n]*:?\s*\n+",
        "", (text or ""), count=1, flags=re.I,
    ).strip()


def _safe(s: str) -> str:
    """Neutralize the triple-quote delimiter so user text can't break out of it."""
    return (s or "").replace('"""', "'''")


def generate_prompt(idea: str, framework: str, target_model: str = "Any LLM",
                    task_type: str = "", mode: str = "outcome",
                    self_check: bool = False) -> dict:
    """Convert an idea (any tone) into a structured LLM prompt. Returns
    {prompt, framework, error}."""
    idea = (idea or "").strip()
    if not idea:
        return {"prompt": "", "framework": framework, "error": "Please describe your idea first."}

    fw_instr = PROMPT_FRAMEWORKS.get(framework) or next(iter(PROMPT_FRAMEWORKS.values()))
    task_line = f"Task type / domain: {task_type.strip()}\n" if (task_type or "").strip() else ""
    clauses = _conditional_clauses(idea, task_type)
    clause_block = ("CONTEXT-SPECIFIC REQUIREMENTS (fold these into the prompt):\n"
                    + "\n".join(f"- {c}" for c in clauses) + "\n\n") if clauses else ""

    meta = (
        "You are an expert prompt engineer. Convert the user's idea into a high-quality, "
        f"ready-to-use prompt for {target_model}.\n\n"
        "The user's idea may be written casually/informally or formally — interpret their "
        "intent either way:\n"
        f'"""{_safe(idea)}"""\n\n'
        f"{task_line}"
        f"Framework to follow:\n{fw_instr}\n\n"
        f"{_mode_instruction(mode)}\n\n"
        f"{_STRUCTURE_RULES}\n\n"
        f"{_build_clarity_rules()}\n\n"
        f"{_build_hygiene_rules()}\n\n"
        f"{clause_block}"
        "Output ONLY the final prompt text, ready to paste. No preamble, no explanation, "
        "no surrounding code fences."
    )

    result = generate_text(meta)
    if is_unavailable_response(result):
        return {"prompt": "", "framework": framework, "error": result or "AI features are not available."}
    cleaned = _sanitize(_strip_preamble(_strip_fences(result)))
    if not cleaned.strip():
        return {"prompt": "", "framework": framework, "error": "The model returned an empty prompt. Please try again."}
    if self_check:
        cleaned = cleaned + _SELF_CHECK_TAIL
    return {"prompt": cleaned, "framework": framework, "error": None}


def _repair(prompt: str, terms: list, max_passes: int = 2):
    """Bounded, fail-loud repair loop: rewrite away zero-tolerance violations, re-running
    the deterministic scan after each pass. Returns (improved, warning)."""
    improved = prompt
    residual = terms
    passes = 0
    while residual and passes < max_passes:
        passes += 1
        repair_meta = (
            "Rewrite the prompt below to REMOVE the banned items listed, changing nothing else and "
            "keeping every requirement intact.\n"
            f"Banned items still present: {', '.join(residual)}\n\n"
            f'Prompt:\n"""{_safe(improved)}"""\n\n'
            "Hard rules: no em dashes, no en-dash punctuation, no smart quotes, none of the listed "
            "AI-tell words, no escape-hatch phrases, no placeholder tokens. Output ONLY the rewritten "
            "prompt, no preamble or code fences."
        )
        rep = generate_text(repair_meta)
        if is_unavailable_response(rep) or not (rep or "").strip():
            break
        cand = _sanitize(_strip_preamble(_strip_fences(rep)))
        if cand.strip():
            improved = cand
        residual = _hard_terms(improved)
    warning = None
    if residual:
        warning = "Could not fully remove: " + ", ".join(residual[:8]) + "."
    return improved, warning


def refine_prompt(prompt: str, idea: str = "", framework: str = "") -> dict:
    """Critique a prompt against the QC quality rubric and return an improved version.
    Returns {improved, issues, score, warning, error}."""
    prompt = (prompt or "").strip()
    if not prompt:
        return {"improved": "", "issues": [], "score": 0, "warning": None, "error": "Nothing to refine."}

    idea_line = f'Original idea: """{_safe(idea.strip())}"""\n' if (idea or "").strip() else ""
    meta = (
        "You are a prompt-quality reviewer. Critique and improve the LLM prompt below.\n\n"
        f"{idea_line}"
        f'Current prompt:\n"""{_safe(prompt)}"""\n\n'
        "Evaluate it on: structural completeness (role, context/background, input, task, constraints, "
        "and an exact output format all present); clarity (exact thresholds, explicit conditionals, "
        "clear scope, an objective and verifiable success condition, atomic one-obligation-per-line "
        "requirements, affirmative rather than negated phrasing, no hollow quality words like "
        "'correctly'/'properly'/'clearly'); a two-reader test (could two careful readers disagree on "
        "whether an output passes? if so, sharpen it); economy (remove any constraint that does not "
        "change the set of acceptable outputs, restates another, or contradicts another; do not pad); "
        "and hygiene (no em dashes, no smart quotes, no AI-tell words like 'utilize'/'leverage'/'in "
        "order to', no escape-hatch phrases, no placeholder tokens). Then rewrite it to fix every "
        "issue, adding a hard constraint or concrete detail where it helps.\n"
        "GUARDRAIL: your rewrite may only better-satisfy these rules. Never relax, remove, or trade "
        "away a requirement to improve flow, and never reintroduce a banned item.\n\n"
        "Respond EXACTLY in this format:\n"
        "SCORE: <0-100 integer for the ORIGINAL prompt>\n"
        "ISSUES:\n- <issue>\n- <issue>\n"
        "IMPROVED:\n<the improved prompt, ready to paste, no code fences>"
    )

    out = generate_text(meta)
    if is_unavailable_response(out):
        return {"improved": "", "issues": [], "score": 0, "warning": None,
                "error": out or "AI features are not available."}

    # Read SCORE only from the header (before IMPROVED), case-insensitively.
    score = 0
    head = re.split(r"(?im)^\s*IMPROVED:", out, maxsplit=1)[0]
    m = re.search(r"SCORE:\s*(\d{1,3})", head, re.I)
    if m:
        score = max(0, min(100, int(m.group(1))))

    issues = []
    block = re.search(r"ISSUES:\s*(.*?)(?:\n\s*IMPROVED:|\Z)", out, re.S | re.I)
    if block:
        for line in block.group(1).splitlines():
            ls = line.strip()
            if not ls or ls.lower().startswith("issues"):
                continue
            if ls.startswith(("-", "•", "*")) or re.match(r"^\d+[.)]\s", ls):
                issues.append(re.sub(r"^([-•*]|\d+[.)])\s*", "", ls).strip())
            else:
                issues.append(ls)

    # Augment with deterministic findings on the ORIGINAL prompt, so a rule violation
    # is always reported even if the model's critique missed it.
    det = _hygiene_issues(prompt)
    issues = det + [i for i in issues if i]
    seen, deduped = set(), []
    for i in issues:
        k = i.lower()
        if k not in seen:
            seen.add(k)
            deduped.append(i)

    improved = prompt
    mp = re.search(r"IMPROVED:\s*(.*)", out, re.S | re.I)
    if mp and mp.group(1).strip():
        improved = _strip_fences(mp.group(1))
    improved = _sanitize(improved)  # also covers the fallback (raw original) path

    # Bounded, fail-loud repair: re-run the scan and rewrite away any residual
    # zero-tolerance violation the refine pass left in, capping at 2 extra passes.
    warning = None
    residual = _hard_terms(improved)
    if residual:
        improved, warning = _repair(improved, residual)

    return {"improved": improved, "issues": deduped[:12], "score": score,
            "warning": warning, "error": None}
