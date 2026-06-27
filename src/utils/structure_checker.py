"""
Section Completeness & Word-Budget Checklist (offline).

Detects the sections present in a draft and checks them against the chosen
paper type's expected structure (missing / out-of-order), plus each present
section's word count vs the template's recommended range. No AI backend needed.
"""

from .section_splitter import split_sections
from .content_generator import content_generator
from .readability_analyzer import compare_to_target

# Template section names that mean the same thing as a detected canonical name.
# (section_splitter canonicalizes a "Methods" heading to "Methodology", but the
# Technical Report template lists the section as "Methods".)
_EQUIV = {"Methods": "Methodology"}


def _canon(name: str) -> str:
    return _EQUIV.get(name, name)


def get_paper_types():
    """List of (key, display name) for the UI selector."""
    return [(k, t["name"]) for k, t in content_generator.research_templates.items()]


def check_structure(text: str, paper_type: str = "empirical") -> dict:
    text = (text or "").strip()
    if not text:
        return {"error": "No text provided."}

    templates = content_generator.research_templates
    tpl = templates.get(paper_type, templates["empirical"])
    # Expected content sections (Title is not a prose section a user pastes).
    expected = [s for s in tpl["structure"] if s != "Title"]
    word_limits = tpl.get("word_limits", {})

    found = split_sections(text)
    found_names = list(found.keys())
    if not found_names:
        return {"error": "No section headings detected. Add headings like 'Introduction', "
                         "'Methodology', etc. (one per line) so sections can be checked."}

    found_set = set(found_names)
    expected_canon = {_canon(e) for e in expected}

    def _present(s):
        return s in found_set or _canon(s) in found_set

    def _body_for(s):
        return found.get(s) or found.get(_canon(s)) or ""

    present = [s for s in expected if _present(s)]
    missing = [s for s in expected if not _present(s)]
    extra = [s for s in found_names if s not in expected and s not in expected_canon]

    rows = []
    for s in present:
        wc = len(_body_for(s).split())
        wl = word_limits.get(s)
        status, target_str = "n/a", (wl or "—")
        if wl:
            try:
                lo, hi = [int(x) for x in str(wl).split("-")]
                status = compare_to_target(wc, lo, hi)["status"]
            except (ValueError, AttributeError):
                pass
        rows.append({"section": s, "words": wc, "target": target_str, "status": status})

    # Order check on canonical names (so Methods≡Methodology doesn't false-flag).
    found_canon_seq = [_canon(f) for f in found_names if _canon(f) in expected_canon]
    present_canon = set(found_canon_seq)
    expected_present_seq = [c for c in (_canon(e) for e in expected) if c in present_canon]
    order_ok = found_canon_seq == expected_present_seq

    return {
        "paper_type": paper_type,
        "paper_name": tpl["name"],
        "expected": expected,
        "present": present,
        "missing": missing,
        "extra": extra,
        "rows": rows,
        "order_ok": order_ok,
        "completeness_pct": round(100 * len(present) / max(1, len(expected))),
        "error": None,
    }
