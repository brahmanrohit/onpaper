"""
ZeroGPT-style AI vs Human text detector  (EXPERIMENTAL / heuristic).

This is NOT a trained classifier and cannot match commercial detectors. It is a
transparent, fully-offline heuristic that estimates how "AI-like" a passage is
from well-known statistical signals:

  1. Burstiness  - humans vary sentence length a lot; AI text is more uniform.
  2. Vocabulary diversity (type-token ratio) - AI text often repeats wording.
  3. AI-giveaway phrases - clichés LLMs overuse ("delve into", "moreover", ...).

Each signal yields a 0-1 sub-score; the blended score is reported as a
percentage with a verdict. Treat the result as an estimate, not proof.
"""

import os
import pickle
import re
import statistics
from typing import Dict
from nltk.tokenize import sent_tokenize, word_tokenize
from .config import AI_DETECTOR_MODEL_PATH

# Lazily-loaded trained classifier (HC3-trained TF-IDF + LogisticRegression).
_ml_pipeline = None
_ml_loaded = False


def _get_ml_pipeline():
    """Load the trained AI-detector pipeline once, if it exists."""
    global _ml_pipeline, _ml_loaded
    if not _ml_loaded:
        _ml_loaded = True
        path = str(AI_DETECTOR_MODEL_PATH)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    _ml_pipeline = pickle.load(f).get("pipeline")
            except Exception:
                _ml_pipeline = None
    return _ml_pipeline


def _ml_ai_probability(text: str):
    """Return P(AI) in 0-100 from the trained model, or None if unavailable."""
    pipe = _get_ml_pipeline()
    if pipe is None:
        return None
    try:
        return round(float(pipe.predict_proba([text])[0][1]) * 100, 1)
    except Exception:
        return None

# Words / phrases that large language models tend to over-use.
AI_GIVEAWAY_PHRASES = [
    "delve into", "delve", "moreover", "furthermore", "in conclusion",
    "it is important to note", "it is worth noting", "in today's world",
    "in the realm of", "realm of", "navigating the", "tapestry", "underscore",
    "leverage", "leveraging", "seamless", "seamlessly", "robust", "holistic",
    "multifaceted", "paradigm", "pivotal", "intricate", "myriad",
    "a testament to", "plays a crucial role", "plays a significant role",
    "it is essential", "on the other hand", "as a result", "in summary",
    "cutting-edge", "ever-evolving", "foster", "facilitate", "encompass",
    "notably", "consequently", "additionally", "however, it is",
]

MIN_WORDS = 25  # below this, the estimate is unreliable


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def detect_ai_text(text: str, use_perplexity: bool = False) -> Dict:
    """Estimate whether a passage is AI- or human-written.

    Returns a dict with ai_score (0-100), verdict, signals, and flagged phrases.
    If use_perplexity is True and the perplexity model is available, a
    GPTZero-style perplexity signal is added and used as the primary score.
    """
    text = (text or "").strip()
    if not text:
        return {"error": "Please enter some text to analyze."}

    sentences = [s for s in sent_tokenize(text) if s.strip()]
    words = [w.lower() for w in word_tokenize(text) if re.search(r"[a-zA-Z]", w)]
    word_count = len(words)
    sentence_count = len(sentences)

    if word_count < MIN_WORDS or sentence_count < 2:
        return {
            "error": f"Need at least {MIN_WORDS} words and 2 sentences for a reliable estimate.",
            "word_count": word_count,
        }

    # --- Signal 1: Burstiness (variation in sentence length) ---
    lengths = [len([w for w in word_tokenize(s) if re.search(r"[a-zA-Z]", w)]) for s in sentences]
    lengths = [n for n in lengths if n > 0]
    mean_len = statistics.mean(lengths)
    stdev_len = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
    cv = (stdev_len / mean_len) if mean_len else 0.0  # coefficient of variation
    # Human writing usually has cv ~0.5+; very uniform (low cv) looks AI-like.
    uniformity_score = _clamp(1.0 - (cv / 0.6))

    # --- Signal 2: Vocabulary diversity (type-token ratio) ---
    unique_words = len(set(words))
    ttr = unique_words / word_count
    # Lower diversity -> more AI-like. Normalize around a 0.45 reference.
    diversity_score = _clamp((0.6 - ttr) / 0.4)

    # --- Signal 3: AI-giveaway phrase density ---
    lowered = " " + re.sub(r"\s+", " ", text.lower()) + " "
    found_phrases = []
    hits = 0
    # Longest-first + skip phrases contained in an already-counted longer phrase,
    # so a span like "delve into" isn't ALSO counted as "delve" (double-counting).
    for phrase in sorted(AI_GIVEAWAY_PHRASES, key=len, reverse=True):
        if any(phrase != fp and phrase in fp for fp in found_phrases):
            continue
        c = lowered.count(" " + phrase + " ") + lowered.count(" " + phrase + ",") + lowered.count(" " + phrase + ".")
        if c > 0:
            hits += c
            found_phrases.append(phrase)
    phrase_density = hits / sentence_count
    phrase_score = _clamp(phrase_density / 0.8)

    # --- Heuristic blend ---
    # Vocabulary diversity is unreliable on short text, so it is weighted lightly;
    # sentence uniformity and AI-phrase density carry most of the signal.
    heuristic_score = 100 * (0.45 * uniformity_score + 0.10 * diversity_score + 0.45 * phrase_score)
    heuristic_score = round(heuristic_score, 1)

    # --- Trained model (if available) ---
    ml_score = _ml_ai_probability(text)  # 0-100 or None

    # --- Perplexity signal (GPTZero-style, optional) ---
    perplexity_score = None
    perplexity_info = None
    if use_perplexity:
        from .perplexity_detector import detect_perplexity
        pr = detect_perplexity(text)
        if pr.get("available") and pr.get("error") is None:
            perplexity_score = pr["ai_score"]
            perplexity_info = {"perplexity": pr["perplexity"], "burstiness": pr["burstiness"]}

    # --- Final score: weighted noisy-OR over available signals ---
    # Each signal is independent EVIDENCE of AI authorship, so we combine them
    # with a noisy-OR: any reliable signal can raise the score, but a weak signal
    # cannot cancel a strong one (unlike a plain average). Weights reflect each
    # signal's reliability. Text where ALL signals are low stays low (human).
    signal_list = []  # (fraction 0-1, reliability weight)
    used = []
    if perplexity_score is not None:
        signal_list.append((perplexity_score / 100.0, 1.0))
        used.append("perplexity")
    if ml_score is not None:
        signal_list.append((ml_score / 100.0, 0.9))
        used.append("trained model")
    # The lexical heuristic is the weakest signal and can over-fire on short text
    # (e.g. one common word like "robust"). When a stronger signal (trained model
    # or perplexity) is available, down-weight the heuristic so it can't override a
    # confident human verdict; otherwise it carries its normal weight.
    heuristic_weight = 0.5 if (ml_score is not None or perplexity_score is not None) else 0.7
    signal_list.append((heuristic_score / 100.0, heuristic_weight))
    used.append("heuristic")

    prod = 1.0
    for frac, weight in signal_list:
        frac = min(max(frac, 0.0), 1.0)
        prod *= (1.0 - frac) ** weight
    ai_score = round(100.0 * (1.0 - prod), 1)
    method = " + ".join(used)

    if ai_score >= 65:
        verdict = "Likely AI-generated"
    elif ai_score >= 35:
        verdict = "Mixed / Uncertain"
    else:
        verdict = "Likely Human-written"

    return {
        "ai_score": ai_score,
        "human_score": round(100 - ai_score, 1),
        "verdict": verdict,
        "method": method,
        "ml_score": ml_score,
        "heuristic_score": heuristic_score,
        "perplexity_score": perplexity_score,
        "perplexity_info": perplexity_info,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "signals": {
            "burstiness_cv": round(cv, 2),
            "uniformity_score": round(uniformity_score, 2),
            "vocabulary_diversity_ttr": round(ttr, 2),
            "diversity_score": round(diversity_score, 2),
            "ai_phrase_hits": hits,
            "phrase_score": round(phrase_score, 2),
        },
        "flagged_phrases": found_phrases,
        "error": None,
    }
