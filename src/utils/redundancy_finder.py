"""
Intra-Document Self-Plagiarism & Repetition Finder (offline).

Finds near-duplicate sentences WITHIN one paper (recycled phrasing, a definition
pasted twice, the abstract restated in the conclusion) so the writer can cut
redundancy. Distinct from corpus-based plagiarism — it compares a document to
itself. Uses a single TF-IDF transform + cosine; no AI backend or network.
"""

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def find_repetition(text: str, threshold: float = 0.8, min_words: int = 8) -> dict:
    text = (text or "").strip()
    if not text:
        return {"error": "No text provided."}
    all_sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    # Keep substantial sentences, remembering each one's ORIGINAL position so the
    # "skip adjacent" rule uses true document order (not filtered-list indices).
    kept = [(idx, s) for idx, s in enumerate(all_sentences) if len(s.split()) >= min_words]
    if len(kept) < 2:
        return {"error": "Need at least two substantial sentences to compare."}

    orig_idx = [idx for idx, _ in kept]
    sentences = [s for _, s in kept]
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    except ValueError:
        return {"error": "Not enough distinct content to analyze."}

    sims = cosine_similarity(matrix)
    n = len(sentences)
    pairs = []
    for a in range(n):
        for b in range(a + 1, n):
            if abs(orig_idx[a] - orig_idx[b]) <= 1:  # truly adjacent in the document
                continue
            score = float(sims[a][b])
            if score >= threshold:
                pairs.append({"a": sentences[a], "b": sentences[b],
                              "similarity": round(score, 2),
                              "i": orig_idx[a] + 1, "j": orig_idx[b] + 1})
    pairs.sort(key=lambda p: p["similarity"], reverse=True)

    return {
        "sentence_count": n,
        "pairs": pairs[:25],
        "pair_count": len(pairs),
        "threshold": threshold,
        "error": None,
    }
