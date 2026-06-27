"""
Flow & Cohesion Map (offline).

Measures semantic similarity between consecutive paragraphs to surface abrupt
topic jumps / missing transitions — turning the vague reviewer complaint "your
paper doesn't flow" into a concrete, ordered list of weak boundaries.

Reuses the MiniLM embedder (same model as Chat-with-your-Paper); falls back to
TF-IDF cosine when the embedding model is unavailable. No AI backend needed.
"""

import re
import numpy as np
from . import embedder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Transition connectors whose presence at a boundary signals an intended link.
CONNECTORS = [
    "however", "moreover", "furthermore", "therefore", "thus", "consequently",
    "additionally", "in addition", "meanwhile", "nevertheless", "nonetheless",
    "similarly", "conversely", "in contrast", "on the other hand", "as a result",
    "for example", "for instance", "first", "second", "third", "next", "finally",
    "in conclusion", "overall", "specifically", "notably", "accordingly", "hence",
    "whereas", "although", "by contrast", "building on", "to summarize",
]


def _split_paragraphs(text: str):
    text = (text or "").strip()
    parts = re.split(r"\n\s*\n", text)
    paras = [p.strip() for p in parts if len(p.split()) >= 5]
    # Fallback: many pastes (from PDFs/Word) use single newlines between
    # paragraphs. If blank-line splitting found <2, retry on single newlines.
    if len(paras) < 2 and "\n" in text:
        paras = [p.strip() for p in text.split("\n") if len(p.split()) >= 5]
    return paras


def _starts_with_connector(paragraph: str) -> bool:
    # Compare whole tokens so "Firstly"/"Nextflow" don't match "first"/"next".
    head = " " + " ".join(w.strip(".,;:").lower() for w in paragraph.split()[:6]) + " "
    return any((" " + c + " ") in head for c in CONNECTORS)


def analyze_cohesion(text: str) -> dict:
    """Return per-boundary similarity + weak-transition flags."""
    paras = _split_paragraphs(text)
    if len(paras) < 2:
        return {"error": "Need at least 2 paragraphs (separated by a blank line) to map flow."}

    embs = embedder.embed(paras) if embedder.is_available() else None
    if embs is not None:
        mode = "semantic (MiniLM embeddings)"
        sims = [float(np.dot(embs[i], embs[i + 1])) for i in range(len(paras) - 1)]
    else:
        mode = "lexical (TF-IDF fallback)"
        try:
            m = TfidfVectorizer(stop_words="english").fit_transform(paras)
            sims = [float(cosine_similarity(m[i], m[i + 1])[0][0]) for i in range(len(paras) - 1)]
        except ValueError:
            return {"error": "Not enough distinct content to analyze flow."}

    avg = sum(sims) / len(sims)
    std = (sum((s - avg) ** 2 for s in sims) / len(sims)) ** 0.5
    # Relative threshold: flag boundaries notably below this paper's own average,
    # which beats a hard cutoff since absolute similarity varies by topic/length.
    floor = 0.15 if embs is not None else 0.05
    threshold = max(floor, avg - 0.75 * std)

    pairs = []
    for i, s in enumerate(sims):
        connector = _starts_with_connector(paras[i + 1])
        weak = s < threshold and not connector
        pairs.append({
            "from": i + 1, "to": i + 2,
            "similarity": round(s, 3),
            "has_connector": connector,
            "weak": weak,
            "preview": (paras[i + 1][:90] + "…") if len(paras[i + 1]) > 90 else paras[i + 1],
        })

    return {
        "paragraphs": len(paras),
        "avg_similarity": round(avg, 3),
        "threshold": round(threshold, 3),
        "weak_count": sum(1 for p in pairs if p["weak"]),
        "pairs": pairs,
        "mode": mode,
        "error": None,
    }
