"""
Academic Tone Auditor (offline).

Counts hedging, overclaiming, filler/weasel words and first-person usage, with
per-1000-word rates and flagged example sentences — helping writers calibrate
claim strength (a common, teachable weakness). Pure regex + NLTK; no AI needed.
This is the offline tone counterpart to the ZeroGPT phrase detector.
"""

import re
from nltk.tokenize import sent_tokenize

LEXICONS = {
    "Hedging": [
        "may", "might", "could", "possibly", "perhaps", "potentially", "arguably",
        "somewhat", "relatively", "fairly", "seems", "appears", "suggests",
        "it is possible", "to some extent", "in some cases", "tends to",
    ],
    "Overclaiming": [
        "clearly", "obviously", "undoubtedly", "undeniably", "proves", "proven",
        "always", "never", "every", "all of", "without doubt", "certainly",
        "definitely", "unquestionably", "guarantees", "perfect", "completely",
    ],
    "Filler / weasel": [
        "very", "really", "quite", "basically", "actually", "literally", "simply",
        "in order to", "due to the fact that", "a lot of", "kind of", "sort of",
        "it is important to note", "needless to say", "as we all know",
    ],
    "First person": [
        "i ", "we ", "our ", "us ", "my ", "i'm", "we're", "i've", "we've",
    ],
}


def _count(text_padded: str, phrase: str) -> int:
    """Count phrase occurrences with word boundaries (case-insensitive)."""
    if phrase.endswith(" ") or phrase.startswith(" "):
        # Space-padded token (e.g. "i "): the trailing space is the right
        # boundary, but we still need a LEFT word boundary so "i " doesn't match
        # as the suffix of "sushi", "chili", etc. (the bare .count() bug).
        return len(re.findall(r"(?<![a-z])" + re.escape(phrase), text_padded))
    return len(re.findall(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", text_padded))


def analyze_tone(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {"error": "No text provided."}
    sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    words = len(text.split())
    if words < 20:
        return {"error": "Please provide at least ~20 words for a meaningful tone audit."}

    padded = " " + re.sub(r"\s+", " ", text.lower()) + " "
    categories = {}
    for cat, phrases in LEXICONS.items():
        hits, found = 0, []
        for p in phrases:
            c = _count(padded, p)
            if c:
                hits += c
                found.append(p.strip())
        categories[cat] = {
            "count": hits,
            "per_1000_words": round(hits / words * 1000, 1),
            "terms": found[:12],
        }

    # Example sentences: overclaim term with no citation-ish support, or 2+ hedges.
    overclaim_terms = [t for t in categories["Overclaiming"]["terms"]]
    examples = {"overclaim": [], "over_hedged": []}
    hedge_rx = re.compile(r"(?<![a-z])(" + "|".join(re.escape(h) for h in LEXICONS["Hedging"]) + r")(?![a-z])", re.I)
    for s in sentences:
        low = s.lower()
        if overclaim_terms and any(re.search(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", low) for t in overclaim_terms):
            if len(examples["overclaim"]) < 8:
                examples["overclaim"].append(s)
        if len(hedge_rx.findall(low)) >= 2 and len(examples["over_hedged"]) < 8:
            examples["over_hedged"].append(s)

    return {
        "word_count": words,
        "sentence_count": len(sentences),
        "categories": categories,
        "examples": examples,
        "error": None,
    }
