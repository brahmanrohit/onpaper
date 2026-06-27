"""
Paraphraser / Rewriter.

Rewrites text for different goals (reduce plagiarism, improve clarity, change
tone, simplify, shorten, expand) using the shared AI gateway in gemini_helper
(which routes to Ollama or Gemini). Requires a working AI backend; if none is
available, generate_text() returns an actionable error string which is passed
straight back to the UI.
"""

from typing import Dict
from .gemini_helper import generate_text, is_unavailable_response

# Each mode maps to a specific instruction for the language model.
MODES: Dict[str, str] = {
    "Standard rewrite": "Rewrite the text in your own words while keeping the original meaning.",
    "Reduce plagiarism": (
        "Rewrite the text so it is substantially different in wording and sentence "
        "structure from the original, while preserving the exact meaning. Avoid copying "
        "any phrase of more than four consecutive words."
    ),
    "Improve clarity": "Rewrite the text to be clearer, more concise, and easier to read, without changing the meaning.",
    "Academic tone": "Rewrite the text in a formal, academic tone suitable for a research paper.",
    "Simplify": "Rewrite the text in plain, simple language that a general audience can easily understand.",
    "Shorten": "Rewrite the text to be significantly shorter while keeping all key information.",
    "Expand": "Rewrite the text with more detail, examples, and explanation, expanding its length.",
}

DEFAULT_MODE = "Standard rewrite"


def paraphrase_text(text: str, mode: str = DEFAULT_MODE) -> Dict:
    """Paraphrase/rewrite text according to the chosen mode.

    Returns {"result": str, "error": Optional[str]}.
    """
    text = (text or "").strip()
    if not text:
        return {"result": "", "error": "Please enter some text to rewrite."}

    instruction = MODES.get(mode, MODES[DEFAULT_MODE])
    prompt = (
        f"{instruction}\n\n"
        "Return only the rewritten text, with no preamble, labels, quotes, or commentary.\n\n"
        f"Text:\n{text}"
    )

    result = generate_text(prompt)
    if is_unavailable_response(result):
        return {"result": "", "error": result}
    return {"result": result.strip(), "error": None}


def get_modes() -> list:
    """Return the list of available rewrite modes (for the UI)."""
    return list(MODES.keys())
