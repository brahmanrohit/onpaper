"""
Terminology & Acronym Consistency Checker (offline).

Flags the mechanical inconsistencies reviewers dislike:
  * acronyms used before they're defined, or never expanded, or defined with
    two different expansions;
  * single-token term variants (spelling/hyphenation), e.g. "dataset" vs
    "data-set", "color" vs "colour".

Pure regex + stdlib; no AI backend or network needed.
"""

import re
from collections import defaultdict, Counter

# "ACR" or "ACRs" — 2–6 uppercase letters/digits starting with a letter.
ACRONYM_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,5})s?\b")
# "Expanded Form (ACR)" definition pattern.
DEF_RE = re.compile(r"\b([A-Z][A-Za-z0-9][A-Za-z0-9\-\s]{2,60}?)\s*\(([A-Z][A-Z0-9]{1,5})s?\)")

# All-caps tokens that aren't really acronyms to flag.
_IGNORE_ACR = {"I", "A", "OK", "TV", "US", "UK", "EU", "PM", "AM", "CEO", "ID", "FAQ"}

# Common all-caps words / headings that get matched by the acronym regex but
# should not be reported as "undefined acronyms" (when they have no definition).
_COMMON_CAPS_WORDS = {
    "DATA", "NOTE", "NOTES", "END", "FIG", "FIGS", "TAB", "TABS", "REF", "REFS", "EQ", "EQS",
    "THE", "AND", "FOR", "NOT", "ALL", "NEW", "USE", "SEE", "TWO", "ONE", "MAY", "CAN",
    "FIGURE", "TABLE", "ITEM", "STEP", "PART", "WHO", "HOW", "WHY", "ITS", "OUR", "YOU",
    "ABSTRACT", "METHODS", "RESULTS", "INTRO", "AIM", "AIMS",
}


def analyze_consistency(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {"error": "No text provided."}

    # --- Acronyms ---
    defs = defaultdict(list)            # acronym -> list of (expansion, position)
    for m in DEF_RE.finditer(text):
        defs[m.group(2)].append((m.group(1).strip(), m.start()))

    usages = defaultdict(list)          # acronym -> list of positions
    for m in ACRONYM_RE.finditer(text):
        usages[m.group(1)].append(m.start())

    acronym_issues = []
    for acr, positions in sorted(usages.items()):
        if acr in _IGNORE_ACR or len(positions) == 0:
            continue
        def_list = defs.get(acr, [])
        first_use = min(positions)
        if not def_list:
            # Don't flag ordinary all-caps words/headings as undefined acronyms.
            if acr in _COMMON_CAPS_WORDS:
                continue
            acronym_issues.append({"acronym": acr, "issue": "never expanded / defined",
                                   "uses": len(positions)})
            continue
        first_def = min(p for _, p in def_list)
        if first_use < first_def:
            acronym_issues.append({"acronym": acr, "issue": "used before it is defined",
                                   "uses": len(positions)})
        expansions = {e for e, _ in def_list}
        if len(expansions) > 1:
            acronym_issues.append({"acronym": acr,
                                   "issue": "defined with different expansions: " + "; ".join(sorted(expansions)),
                                   "uses": len(positions)})

    # --- Single-token term variants (spelling / hyphenation) ---
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]+[A-Za-z]", text)
    groups = defaultdict(Counter)
    for t in tokens:
        key = re.sub(r"[\-\s]", "", t.lower())
        groups[key][t] += 1
    term_variants = []
    for key, surfaces in groups.items():
        if len(key) < 4:
            continue
        # Only flag when the variants differ by more than capitalization.
        if len({s.lower() for s in surfaces}) > 1 and sum(surfaces.values()) >= 3:
            term_variants.append({"forms": dict(surfaces), "total": sum(surfaces.values())})
    term_variants.sort(key=lambda x: x["total"], reverse=True)

    return {
        "acronym_issues": acronym_issues,
        "acronyms_found": len([a for a in usages if a not in _IGNORE_ACR]),
        "term_variants": term_variants[:20],
        "error": None,
    }
