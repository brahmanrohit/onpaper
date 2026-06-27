"""
Citation Density & Uncited-Claim Detector (offline).

Counts in-text citation markers (APA author-year and numeric [n]) overall and
per section, and flags evidence-bearing sentences (reporting verbs, statistics,
comparatives) that carry no nearby citation — the single most common reviewer
complaint. Pure regex + NLTK; no AI backend or network needed.
"""

import re
from nltk.tokenize import sent_tokenize
from .section_splitter import split_sections

# In-text citation markers. "et al." is a terminal optional suffix before the
# year (real form is "Smith et al., 2020"), NOT part of the name-joining group.
CITATION_PATTERNS = [
    # (Author, 2020) / (Author & Author, 2020) / (Author et al., 2020) / (Smith, 2020a)
    re.compile(r"\([A-Z][A-Za-z\-]+(?:\s+(?:and|&)\s+[A-Za-z\-]+)*(?:\s+et al\.)?,?\s*\d{4}[a-z]?\)"),
    # numeric [12] / [1, 3] / [1-4]
    re.compile(r"\[\d+(?:\s*[-,]\s*\d+)*\]"),
    # "Author (2020)" / "Smith et al. (2020)" narrative form
    re.compile(r"\b[A-Z][A-Za-z\-]+(?:\s+(?:and|&)\s+[A-Za-z\-]+)*(?:\s+et al\.)?\s+\(\d{4}[a-z]?\)"),
]

# Strong cues that signal an evidence claim on their own.
STRONG_CUES = re.compile(
    r"\b(studies show|research (?:shows|suggests|indicates)|evidence (?:shows|indicates|suggests)|"
    r"it has been shown|demonstrate[sd]? that|prove[sd]? that|according to|"
    r"results? (?:show|indicate|suggest|reveal)|findings? (?:show|indicate|suggest)|"
    r"associated with|correlat|outperform)", re.I)
# Weak cues (bare comparatives/adverbs) only count as a claim if a statistic is
# also present — avoids flagging benign sentences like "We improved the layout."
WEAK_CUES = re.compile(
    r"\b(significantly|compared (?:to|with)|increase[sd]?|decrease[sd]?|improve[sd]?|reduce[sd]?)", re.I)
NUMBER_CLAIM = re.compile(r"\b\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?\s*(?:times|fold|percent|x)\b")


def _count_markers(text: str) -> int:
    return sum(len(p.findall(text)) for p in CITATION_PATTERNS)


def _has_citation(sentence: str) -> bool:
    return any(p.search(sentence) for p in CITATION_PATTERNS)


def analyze_claims(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {"error": "No text provided."}
    sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    if not sentences:
        return {"error": "Not enough text to analyze."}

    words = len(text.split())
    total_markers = _count_markers(text)

    uncited = []
    for s in sentences:
        if _has_citation(s):
            continue
        has_number = bool(NUMBER_CLAIM.search(s))
        if STRONG_CUES.search(s) or has_number or (WEAK_CUES.search(s) and has_number):
            uncited.append(s)

    # Per-section density (only if the text has detectable headings).
    per_section = []
    for name, body in split_sections(text).items():
        bw = len(body.split())
        if bw == 0:
            continue
        m = _count_markers(body)
        per_section.append({
            "section": name, "words": bw, "markers": m,
            "per_1000": round(m / bw * 1000, 1),
        })

    return {
        "word_count": words,
        "sentence_count": len(sentences),
        "citation_markers": total_markers,
        "citations_per_1000_words": round(total_markers / max(1, words) * 1000, 1),
        "uncited_claims": uncited[:25],
        "uncited_count": len(uncited),
        "per_section": per_section,
        "error": None,
    }
