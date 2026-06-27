"""
Keyword & Contribution Extractor (offline).

Suggests candidate keywords (TF-IDF top terms over the draft's sentences) and
surfaces likely contribution/novelty statements via cue phrases. Helps authors
pick keywords and articulate contributions. Pure scikit-learn + regex.
"""

import re
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

CONTRIB_CUES = re.compile(
    r"\b(we (?:propose|present|introduce|develop|design|demonstrate|contribute)|"
    r"our (?:contribution|approach|method|model|framework)|"
    r"this (?:paper|study|work|article) (?:proposes|presents|introduces|develops|contributes)|"
    r"the (?:main|key|primary) contribution|for the first time|to the best of our knowledge|"
    r"novel (?:approach|method|methodology|framework|technique|architecture|algorithm|model|system|contribution)|"
    r"we argue that)", re.I)


def extract_keywords(text: str, top_k: int = 8) -> dict:
    text = (text or "").strip()
    if not text:
        return {"error": "No text provided."}
    sentences = [s for s in sent_tokenize(text) if s.strip()]
    docs = sentences if len(sentences) >= 2 else [text]

    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=2000)
        matrix = vec.fit_transform(docs)
    except ValueError:
        return {"error": "Not enough text to extract keywords."}

    scores = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    # Prefer longer terms on (near-)ties so phrases ("machine translation") aren't
    # pre-empted by their constituent unigrams ("machine").
    ranked = sorted(zip(terms, scores), key=lambda x: (x[1], len(x[0].split())), reverse=True)

    def _toks(s):
        return set(s.split())

    keywords = []
    for term, _ in ranked:
        if len(keywords) >= top_k:
            break
        # Whole-word, one-directional: drop a NEW term only if its words are a
        # subset of an already-chosen term (so a unigram can't block a phrase,
        # and char-substrings like "learn" in "learning" aren't merged).
        tt = _toks(term)
        if any(tt <= _toks(k) for k in keywords):
            continue
        keywords.append(term)

    contributions = [s.strip() for s in sentences if CONTRIB_CUES.search(s)][:10]

    return {
        "keywords": keywords,
        "contributions": contributions,
        "sentence_count": len(sentences),
        "error": None,
    }
