"""
Readability & Writing Quality analyzer.

Fully offline (no AI key, no network). Implements the standard, well-defined
readability formulas (Flesch Reading Ease, Flesch-Kincaid Grade, Gunning Fog)
plus simple writing-quality heuristics (passive voice, long sentences,
hard words). Tokenization uses NLTK, which is already a project dependency.
"""

import re
from typing import Dict, List
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Ensure NLTK tokenizer data (punkt_tab/punkt) is available.
from .nltk_setup import ensure_nltk_data
ensure_nltk_data()

_VOWELS = "aeiouy"

# Irregular past participles commonly used in academic writing, for the
# passive-voice heuristic (regular ones ending in "ed" are matched separately).
_IRREGULAR_PARTICIPLES = {
    "been", "done", "made", "shown", "found", "given", "taken", "seen",
    "written", "known", "held", "built", "kept", "left", "told", "led",
    "set", "put", "read", "run", "begun", "drawn", "grown", "proven",
    "chosen", "driven", "spoken", "broken", "tested",
}

_BE_FORMS = {"am", "is", "are", "was", "were", "be", "been", "being", "get", "got", "gets"}


def count_syllables(word: str) -> int:
    """Estimate the syllable count of a word using a vowel-group heuristic."""
    word = word.lower().strip()
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 0
    if len(word) <= 3:
        return 1

    # Count vowel groups
    syllables = len(re.findall(r"[aeiouy]+", word))

    # Silent trailing 'e' (but not 'le' which usually forms a syllable)
    if word.endswith("e") and not word.endswith("le"):
        syllables -= 1
    # Common silent endings
    if word.endswith("es") or word.endswith("ed"):
        # only subtract when the 'e' is silent (heuristic)
        if not re.search(r"[aeiouy][^aeiouy]?(es|ed)$", word):
            syllables -= 0  # keep, edge case

    return max(1, syllables)


def _interpret_flesch(score: float) -> str:
    """Map a Flesch Reading Ease score to a human-readable difficulty band."""
    if score >= 90:
        return "Very easy (5th grade)"
    if score >= 80:
        return "Easy (6th grade)"
    if score >= 70:
        return "Fairly easy (7th grade)"
    if score >= 60:
        return "Standard (8th-9th grade)"
    if score >= 50:
        return "Fairly difficult (10th-12th grade)"
    if score >= 30:
        return "Difficult (college)"
    return "Very difficult (graduate)"


def _find_passive_sentences(sentences: List[str]) -> List[str]:
    """Heuristically flag sentences that appear to use passive voice.

    Heuristic: a 'be' form (is/was/were/been/...) followed within a couple of
    words by a past participle (regular '-ed' or a common irregular form).
    This is an approximation, not a grammatical parse.
    """
    passive = []
    pattern = re.compile(
        r"\b(" + "|".join(_BE_FORMS) + r")\b\s+(\w+ly\s+)?(\w+ed|"
        + "|".join(_IRREGULAR_PARTICIPLES) + r")\b",
        re.IGNORECASE,
    )
    for sent in sentences:
        if pattern.search(sent):
            passive.append(sent.strip())
    return passive


def analyze_text(text: str, long_sentence_words: int = 25) -> Dict:
    """Analyze readability and writing quality of the given text.

    Returns a dict of metrics. All values are computed offline.
    """
    text = (text or "").strip()
    if not text:
        return {"error": "No text provided."}

    sentences = [s for s in sent_tokenize(text) if s.strip()]
    words = [w for w in word_tokenize(text) if re.search(r"[A-Za-z0-9]", w)]

    sentence_count = len(sentences)
    word_count = len(words)

    if sentence_count == 0 or word_count == 0:
        return {"error": "Not enough text to analyze."}

    syllable_count = sum(count_syllables(w) for w in words)
    hard_words = [w for w in words if count_syllables(w) >= 3]
    hard_word_count = len(hard_words)
    char_count = sum(len(re.sub(r"[^A-Za-z]", "", w)) for w in words)

    words_per_sentence = word_count / sentence_count
    syllables_per_word = syllable_count / word_count

    # --- Standard readability formulas ---
    flesch_reading_ease = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
    flesch_kincaid_grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59
    gunning_fog = 0.4 * (words_per_sentence + 100 * (hard_word_count / word_count))

    # --- Writing-quality heuristics ---
    long_sentences = []
    for sent in sentences:
        n = len([w for w in word_tokenize(sent) if re.search(r"[A-Za-z0-9]", w)])
        if n > long_sentence_words:
            long_sentences.append({"sentence": sent.strip(), "words": n})
    long_sentences.sort(key=lambda x: x["words"], reverse=True)

    passive_sentences = _find_passive_sentences(sentences)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "character_count": char_count,
        "syllable_count": syllable_count,
        "avg_words_per_sentence": round(words_per_sentence, 1),
        "avg_syllables_per_word": round(syllables_per_word, 2),
        "flesch_reading_ease": round(flesch_reading_ease, 1),
        "flesch_reading_ease_label": _interpret_flesch(flesch_reading_ease),
        "flesch_kincaid_grade": round(flesch_kincaid_grade, 1),
        "gunning_fog": round(gunning_fog, 1),
        "reading_time_min": round(word_count / 200, 1),  # ~200 wpm
        "speaking_time_min": round(word_count / 130, 1),  # ~130 wpm
        "hard_word_count": hard_word_count,
        "hard_word_percent": round(100 * hard_word_count / word_count, 1),
        "passive_count": len(passive_sentences),
        "passive_percent": round(100 * len(passive_sentences) / sentence_count, 1),
        "passive_sentences": passive_sentences[:10],
        "long_sentence_count": len(long_sentences),
        "long_sentences": long_sentences[:10],
        "long_sentence_threshold": long_sentence_words,
    }


def analyze_sentences(text: str) -> List[Dict]:
    """Per-sentence difficulty, for a readability heat-strip.

    Each sentence is bucketed easy / medium / hard from its length and the share
    of long/complex words. Fully offline.
    """
    sentences = [s for s in sent_tokenize(text or "") if s.strip()]
    out = []
    for s in sentences:
        words = [w for w in word_tokenize(s) if re.search(r"[A-Za-z0-9]", w)]
        n = len(words)
        if n == 0:
            continue
        syllables = sum(count_syllables(w) for w in words)
        spw = syllables / n
        hard = sum(1 for w in words if count_syllables(w) >= 3)
        if n > 30 or spw > 1.9:
            level = "hard"
        elif n > 22 or spw > 1.6:
            level = "medium"
        else:
            level = "easy"
        out.append({"sentence": s.strip(), "words": n,
                    "syllables_per_word": round(spw, 2), "hard_words": hard, "level": level})
    return out


# Target Flesch-Kincaid grade band per audience (used to color the metrics).
AUDIENCE_PRESETS = {
    "Academic / journal (grade 13–16)": (13, 16),
    "Graduate (grade 16–20)": (16, 20),
    "General / undergraduate (grade 8–12)": (8, 12),
    "Broad public (grade 6–10)": (6, 10),
}


def grade_band_status(grade: float, low: int, high: int) -> str:
    """Where a Flesch-Kincaid grade falls relative to a target band."""
    if grade < low:
        return "simpler than your target"
    if grade > high:
        return "more complex than your target"
    return "on target"


def compare_to_target(word_count: int, target_low: int, target_high: int) -> Dict:
    """Compare an actual word count against a target range."""
    if word_count < target_low:
        status = "under"
        message = f"{target_low - word_count} words below the target minimum."
    elif word_count > target_high:
        status = "over"
        message = f"{word_count - target_high} words above the target maximum."
    else:
        status = "ok"
        message = "Within the target range."
    return {"status": status, "message": message,
            "target_low": target_low, "target_high": target_high}
