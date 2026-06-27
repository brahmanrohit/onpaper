"""
Real reference / citation finder.

Queries the free CrossRef API (no API key required) to find real academic
papers with DOIs for a search query, and formats them in APA, IEEE, or MLA
style. This replaces the mock data in citation_manager.py with real results.

Network is required. All calls have timeouts and degrade gracefully on error.
"""

import re
import requests
from typing import List, Dict

CROSSREF_URL = "https://api.crossref.org/works"
# CrossRef asks callers to identify themselves via User-Agent (the "polite pool").
_HEADERS = {"User-Agent": "OnPaper-ResearchAssistant/1.0 (mailto:onpaper@example.com)"}


def search_references(query: str, rows: int = 8) -> Dict:
    """Search CrossRef for references matching a query.

    Returns {"references": [...], "error": Optional[str]}.
    Each reference is a normalized dict with title/authors/year/journal/doi/url/type.
    """
    query = (query or "").strip()
    if not query:
        return {"references": [], "error": "Please enter a search query."}

    try:
        resp = requests.get(
            CROSSREF_URL,
            params={"query": query, "rows": max(1, min(rows, 25)), "select":
                    "title,author,issued,container-title,DOI,URL,type,publisher"},
            headers=_HEADERS,
            timeout=20,
        )
    except requests.exceptions.Timeout:
        return {"references": [], "error": "The reference service timed out. Please try again."}
    except requests.exceptions.RequestException as e:
        return {"references": [], "error": f"Could not reach the reference service: {e}"}

    if resp.status_code != 200:
        return {"references": [], "error": f"Reference service returned status {resp.status_code}."}

    try:
        items = resp.json().get("message", {}).get("items", [])
    except ValueError:
        return {"references": [], "error": "Received an invalid response from the reference service."}

    references = [_parse_crossref(item) for item in items]
    references = [r for r in references if r.get("title")]
    if not references:
        return {"references": [], "error": "No references found for that query."}
    return {"references": references, "error": None}


def _parse_crossref(item: Dict) -> Dict:
    """Normalize a CrossRef item into a simple reference dict."""
    title_list = item.get("title") or []
    title = title_list[0].strip() if title_list else ""

    authors = []
    for a in item.get("author", []) or []:
        family = (a.get("family") or "").strip()
        given = (a.get("given") or "").strip()
        if family or given:
            authors.append({"given": given, "family": family})

    year = ""
    issued = item.get("issued", {}).get("date-parts", [])
    if issued and issued[0] and issued[0][0]:
        year = str(issued[0][0])

    journal_list = item.get("container-title") or []
    journal = journal_list[0].strip() if journal_list else ""

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "journal": journal,
        "publisher": (item.get("publisher") or "").strip(),
        "doi": (item.get("DOI") or "").strip(),
        "url": (item.get("URL") or "").strip(),
        "type": (item.get("type") or "").strip(),
    }


def _initials(given: str) -> str:
    """Turn a given name into APA-style initials, e.g. 'John Paul' -> 'J. P.'."""
    parts = [p for p in given.replace(".", " ").split() if p]
    return " ".join(f"{p[0].upper()}." for p in parts)


def _authors_apa(authors: List[Dict]) -> str:
    names = [f"{a['family']}, {_initials(a['given'])}".strip().rstrip(",") for a in authors if a.get("family")]
    if not names:
        return ""
    if len(names) > 6:
        names = names[:6] + ["et al."]
    if len(names) == 1:
        return names[0]
    if names[-1] == "et al.":
        return ", ".join(names[:-1]) + ", " + names[-1]
    return ", ".join(names[:-1]) + ", & " + names[-1]


def _authors_ieee(authors: List[Dict]) -> str:
    names = []
    for a in authors:
        if a.get("family"):
            init = _initials(a.get("given", ""))
            names.append(f"{init} {a['family']}".strip())
    if not names:
        return ""
    if len(names) > 6:
        names = names[:6] + ["et al."]
    return ", ".join(names)


def _authors_mla(authors: List[Dict]) -> str:
    if not authors:
        return ""
    first = authors[0]
    lead = f"{first.get('family','')}, {first.get('given','')}".strip().rstrip(",")
    if len(authors) == 1:
        return lead
    if len(authors) == 2:
        second = authors[1]
        second_name = " ".join(p for p in [second.get('given', ''), second.get('family', '')] if p).strip()
        return f"{lead}, and {second_name}".strip()
    return f"{lead}, et al."


def _norm_title(title: str) -> str:
    """Normalized title key for de-duplication."""
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def dedupe_and_rank(references: List[Dict], query: str = None):
    """Collapse near-duplicate references (e.g. preprint vs published — same title)
    and, if the embedding model is available, re-rank by semantic closeness to the
    query. Keeps the richer record of each duplicate (prefers one with a DOI, then
    more authors).

    Returns (deduped_references, reranked: bool) — `reranked` is True only when
    semantic re-ranking actually ran, so callers don't over-claim it.
    """
    if not references:
        return references, False

    # De-duplicate by normalized title, keeping the best record per group.
    best_by_title = {}
    order = []
    for ref in references:
        key = _norm_title(ref.get("title", ""))
        if not key:
            order.append(ref)  # keep untitled (rare) as-is
            continue
        if key not in best_by_title:
            best_by_title[key] = ref
            order.append(key)
        else:
            cur = best_by_title[key]
            better = (bool(ref.get("doi")) and not cur.get("doi")) or (
                bool(ref.get("doi")) == bool(cur.get("doi"))
                and len(ref.get("authors") or []) > len(cur.get("authors") or [])
            )
            if better:
                best_by_title[key] = ref
    deduped = [item if isinstance(item, dict) else best_by_title[item] for item in order]

    # Optional semantic re-ranking against the query.
    reranked = False
    if query and len(deduped) > 1:
        try:
            from . import embedder
            if embedder.is_available():
                titles = [r.get("title", "") for r in deduped]
                q_emb = embedder.embed([query])
                t_emb = embedder.embed(titles)
                if q_emb is not None and t_emb is not None:
                    sims = embedder.cosine_sim(q_emb[0], t_emb)
                    deduped = [r for _, r in sorted(zip(sims, deduped),
                                                    key=lambda p: p[0], reverse=True)]
                    reranked = True
        except Exception:
            pass  # ranking is best-effort; keep dedup order on any failure
    return deduped, reranked


def format_reference(ref: Dict, style: str = "APA") -> str:
    """Format a normalized reference dict in APA, IEEE, or MLA style."""
    title = ref.get("title", "")
    year = ref.get("year", "")
    journal = ref.get("journal", "")
    doi = ref.get("doi", "")
    publisher = ref.get("publisher", "")
    source = journal or publisher
    doi_url = f"https://doi.org/{doi}" if doi else ref.get("url", "")

    style = (style or "APA").upper()

    if style == "IEEE":
        authors = _authors_ieee(ref.get("authors", []))
        parts = []
        if authors:
            parts.append(authors + ",")
        # End the title with a comma only if something follows it, else a period
        # (avoids a dangling comma inside the quotes when there's no source/year).
        parts.append(f'"{title},"' if (source or year) else f'"{title}."')
        if source:
            parts.append(f"{source}," if year else f"{source}.")
        if year:
            parts.append(f"{year}.")
        line = " ".join(parts).strip()
        if doi:
            line += f" doi: {doi}."
        elif ref.get("url"):
            line += f" [Online]. Available: {ref['url']}."
        return line

    if style == "MLA":
        authors = _authors_mla(ref.get("authors", []))
        parts = []
        if authors:
            parts.append(f"{authors}.")
        parts.append(f'"{title}."')
        if source:
            parts.append(f"{source},")
        if year:
            parts.append(f"{year}.")
        line = " ".join(parts).strip()
        if doi_url:
            line += f" {doi_url}."
        return line

    # Default: APA
    authors = _authors_apa(ref.get("authors", []))
    parts = []
    if authors:
        parts.append(f"{authors}")
    if year:
        parts.append(f"({year}).")
    parts.append(f"{title}.")
    if source:
        parts.append(f"{source}.")
    line = " ".join(parts).strip()
    if doi_url:
        line += f" {doi_url}"
    return line
