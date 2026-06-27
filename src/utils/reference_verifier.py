"""
Verify-My-References (CrossRef).

Paste a reference list; each entry is matched against CrossRef to check whether
it corresponds to a real paper — catching AI-hallucinated / fabricated citations
(a top cause of desk-rejection). Each entry gets a verdict:

  * Verified       — a real paper whose title is well covered by the entry.
  * Possible match — a partial match; check it manually.
  * Not found      — no good match; likely fabricated or badly formatted.

Also flags a DOI mismatch (entry's DOI ≠ the matched paper's DOI). Network
required (same free CrossRef API as Reference Finder); no AI key needed.
"""

import re
from .reference_finder import search_references

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s,;\]\)]+", re.I)

_STOP = {
    "the", "a", "an", "of", "and", "in", "on", "for", "to", "with", "via", "by",
    "from", "using", "based", "study", "analysis", "approach", "is", "are", "at",
    "as", "review", "paper", "research", "journal", "vol", "no", "pp", "doi",
}


def _content_tokens(s: str):
    return {t for t in re.findall(r"[a-z0-9]+", (s or "").lower())
            if len(t) > 2 and t not in _STOP}


def _split_entries(text: str):
    """Split a pasted reference list into individual entries.

    Blank-line separated -> each block is one (possibly wrapped) reference.
    Otherwise -> one reference per line (matches the UI's "one per line" hint).
    """
    text = (text or "").strip()
    if not text:
        return []
    if re.search(r"\n\s*\n", text):
        candidates = [" ".join(b.split()) for b in re.split(r"\n\s*\n", text)]
    else:
        candidates = [ln.strip() for ln in text.split("\n")]
    cleaned = []
    for e in candidates:
        e = re.sub(r"^[\[\(]?\s*\d+\s*[\]\).:]\s*", "", e).strip()  # strip "[1]", "1.", "(1)"
        if len(e.split()) >= 4:
            cleaned.append(e)
    return cleaned


def verify_references(text: str, max_refs: int = 20) -> dict:
    entries = _split_entries(text)
    if not entries:
        return {"error": "No references found. Paste one reference per line."}

    results = []
    for entry in entries[:max_refs]:
        m = DOI_RE.search(entry)
        entry_doi = m.group(0).rstrip(".").lower() if m else None
        entry_tokens = _content_tokens(entry)

        res = search_references(entry, rows=3)
        refs = res.get("references", []) if not res.get("error") else []

        best, best_score, best_ntitle, matched_by_doi = None, 0.0, 0, False
        # 1) Prefer an exact DOI match — a DOI is a unique identifier.
        if entry_doi:
            for r in refs:
                if (r.get("doi") or "").lower() == entry_doi:
                    best, best_score, matched_by_doi = r, 1.0, True
                    best_ntitle = len(_content_tokens(r.get("title", "")))
                    break
        # 2) Otherwise score by TWO-WAY (F1) title overlap, so a generic real
        #    title fully contained in a fabricated entry no longer scores 100%.
        if best is None:
            for r in refs:
                t_tokens = _content_tokens(r.get("title", ""))
                if not t_tokens:
                    continue
                overlap = len(t_tokens & entry_tokens)
                recall = overlap / len(t_tokens)
                precision = overlap / max(1, len(entry_tokens))
                f1 = 0.0 if (recall + precision) == 0 else 2 * recall * precision / (recall + precision)
                if f1 > best_score:
                    best, best_score, best_ntitle = r, f1, len(t_tokens)

        # Require a reasonably specific title (>=4 content tokens) for a confident
        # verdict, so 2-3 word generic titles can't "verify" fabricated entries.
        if matched_by_doi or (best and best_score >= 0.7 and best_ntitle >= 4):
            verdict = "Verified"
        elif best and best_score >= 0.4:
            verdict = "Possible match"
        else:
            verdict = "Not found"

        found_doi = ((best.get("doi") if best else "") or "").lower()
        # Only flag a DOI mismatch on a confident title match (not when CrossRef
        # simply didn't return the entry's paper, and never when matched by DOI).
        doi_mismatch = bool(entry_doi and found_doi and entry_doi != found_doi
                            and verdict == "Verified" and not matched_by_doi)

        results.append({
            "entry": entry,
            "verdict": verdict,
            "confidence": round(best_score * 100),
            "matched_title": (best.get("title") if best else "") or "",
            "matched_doi": found_doi,
            "entry_doi": entry_doi or "",
            "doi_mismatch": doi_mismatch,
            "url": (f"https://doi.org/{found_doi}" if found_doi
                    else ((best.get("url") if best else "") or "")),
        })

    counts = {"Verified": 0, "Possible match": 0, "Not found": 0}
    for r in results:
        counts[r["verdict"]] += 1
    return {
        "results": results,
        "counts": counts,
        "checked": len(results),
        "truncated": len(entries) > max_refs,
        "total_found": len(entries),
        "error": None,
    }
