"""
Paper Health Check — one click, every offline analyzer, one report card.

Runs the existing offline checkers (readability, citations, flow, tone,
terminology, repetition, structure, abstract, keywords) on a single draft and
turns their raw output into 0-100 sub-scores, an overall score, and a ranked
"fix these first" list. No AI backend or network needed.

Scoring is deliberately simple and explainable (each rule is one line below) so
a writer can see exactly why a dimension lost points.
"""

from .readability_analyzer import analyze_text
from .cohesion_analyzer import analyze_cohesion
from .claim_checker import analyze_claims
from .structure_checker import check_structure
from .abstract_scorecard import score_abstract
from .tone_auditor import analyze_tone
from .consistency_checker import analyze_consistency
from .redundancy_finder import find_repetition
from .keyword_extractor import extract_keywords

# Academic prose usually lands at Flesch-Kincaid grade 12-16.
GRADE_LOW, GRADE_HIGH = 12, 16


def _clamp(x):
    return int(max(0, min(100, round(x))))


def _ok(r):
    return isinstance(r, dict) and not r.get("error")


def score_dimensions(raw):
    """Turn raw analyzer outputs into dimensions. Pure function (easy to test).

    raw: {"read", "claim", "coh", "tone", "term", "rep", "struct", "abs"} -> analyzer dicts.
    Returns a list of {name, icon, score|None, summary, issues:[(severity, text)]}.
    A dimension with score None could not be measured and is excluded from the total.
    """
    dims = []

    r = raw.get("read")
    if _ok(r):
        g = r["flesch_kincaid_grade"]
        dist = max(0, GRADE_LOW - g, g - GRADE_HIGH)
        issues = [("Major" if len(r["long_sentences"]) > 3 else "Minor",
                   f"Long sentence ({s['words']} words): {s['sentence']}") for s in r["long_sentences"][:3]]
        if dist:
            issues.insert(0, ("Major" if dist > 4 else "Minor",
                              f"Grade level {g} is outside the academic sweet spot ({GRADE_LOW}-{GRADE_HIGH})."))
        dims.append(dict(name="Readability", icon="📖",
                         score=_clamp(100 - 8 * dist - 3 * r["long_sentence_count"]),
                         summary=f"Grade {g} · {r['flesch_reading_ease_label']} · {r['long_sentence_count']} long sentence(s)",
                         issues=issues))

    c = raw.get("claim")
    if _ok(c):
        s = _clamp(100 - 12 * c["uncited_count"])
        if c["word_count"] >= 300 and c["citation_markers"] == 0:
            s = min(s, 40)  # a paper-length draft with zero citations is a red flag
        issues = [("Major", f"Claim with no citation: {x}") for x in c["uncited_claims"][:4]]
        if c["citation_markers"] == 0 and c["word_count"] >= 300:
            issues.insert(0, ("Critical", "No in-text citations found anywhere in the draft."))
        dims.append(dict(name="Evidence & Citations", icon="📑", score=s,
                         summary=f"{c['citation_markers']} citation(s) · {c['uncited_count']} uncited claim(s)",
                         issues=issues))

    h = raw.get("coh")
    if _ok(h) and h.get("pairs"):
        weak = [p for p in h["pairs"] if p["weak"]]
        dims.append(dict(name="Flow", icon="🔀",
                         score=_clamp(100 * (1 - len(weak) / len(h["pairs"]))),
                         summary=f"{h['paragraphs']} paragraphs · {len(weak)} abrupt jump(s)",
                         issues=[("Minor", f"Abrupt jump ¶{p['from']}→¶{p['to']}; ¶{p['to']} starts “{p['preview']}”")
                                 for p in weak[:3]]))

    t = raw.get("tone")
    if _ok(t):
        over, hedge = t["examples"]["overclaim"], t["examples"]["over_hedged"]
        dims.append(dict(name="Tone", icon="🎚️", score=_clamp(100 - 10 * (len(over) + len(hedge))),
                         summary=f"{len(over)} overclaim(s) · {len(hedge)} over-hedged sentence(s)",
                         issues=[("Major", f"Overclaim: {x}") for x in over[:3]]
                         + [("Minor", f"Over-hedged: {x}") for x in hedge[:2]]))

    m = raw.get("term")
    if _ok(m):
        n = len(m["acronym_issues"]) + len(m["term_variants"])
        dims.append(dict(name="Terminology", icon="🔤", score=_clamp(100 - 8 * n),
                         summary=f"{len(m['acronym_issues'])} acronym issue(s) · {len(m['term_variants'])} spelling variant(s)",
                         issues=[("Minor", f"{a['acronym']}: {a['issue']}") for a in m["acronym_issues"][:3]]
                         + [("Minor", "Inconsistent spelling: " + ", ".join(v["forms"])) for v in m["term_variants"][:2]]))

    p = raw.get("rep")
    if _ok(p):
        dims.append(dict(name="Originality of phrasing", icon="♻️", score=_clamp(100 - 15 * p["pair_count"]),
                         summary=f"{p['pair_count']} near-duplicate sentence pair(s)",
                         issues=[("Minor", f"Sentences {x['i']} and {x['j']} say nearly the same thing: {x['a']}")
                                 for x in p["pairs"][:2]]))

    st_ = raw.get("struct")
    if _ok(st_):
        dims.append(dict(name="Structure", icon="🧱", score=_clamp(st_["completeness_pct"]),
                         summary=f"{st_['completeness_pct']}% of expected sections present",
                         issues=[("Critical" if sec in ("Abstract", "Introduction", "Conclusion") else "Major",
                                  f"Missing section: {sec}") for sec in st_["missing"]]))

    a = raw.get("abs")
    if _ok(a) and a.get("had_abstract_heading"):
        dims.append(dict(name="Abstract", icon="📋", score=_clamp(a["score"]),
                         summary=f"{a['moves_present']}/{a['moves_total']} expected moves · length {a['length_status']}",
                         issues=[("Major", f"Abstract is missing the '{mv}' move.") for mv, ok in a["moves"].items() if not ok]
                         + [("Minor", tip) for tip in a["tips"][:2]]))
    return dims


_SEV_RANK = {"Critical": 0, "Major": 1, "Minor": 2}


def summarize(dims):
    """Overall score + the issues ranked by severity, then by how weak their dimension is."""
    scored = [d for d in dims if d["score"] is not None]
    overall = _clamp(sum(d["score"] for d in scored) / len(scored)) if scored else None
    fixes = [(sev, d["score"], d["icon"], d["name"], text) for d in dims for sev, text in d["issues"]]
    fixes.sort(key=lambda f: (_SEV_RANK.get(f[0], 3), f[1]))
    return overall, fixes


def grade(overall):
    if overall is None:
        return "—", "Not enough text to judge"
    for cutoff, letter, note in ((90, "A", "Ready for a reviewer"), (80, "B", "Strong — polish the flagged spots"),
                                 (65, "C", "Solid draft — a revision pass will lift it"),
                                 (50, "D", "Needs work on the critical items"), (0, "E", "Early draft — start with the top fixes")):
        if overall >= cutoff:
            return letter, note


def run_health_check(text, paper_type="empirical"):
    """Run every offline analyzer on `text`. Returns {overall, grade, note, dims, fixes, keywords, contributions, error}."""
    text = (text or "").strip()
    if len(text.split()) < 50:
        return {"error": "Paste at least ~50 words so the checks have something to measure."}

    def safe(fn, *args, **kw):
        try:
            return fn(*args, **kw)
        except Exception as e:  # one broken analyzer must not sink the whole report
            return {"error": str(e)}

    raw = {
        "read": safe(analyze_text, text), "claim": safe(analyze_claims, text), "coh": safe(analyze_cohesion, text),
        "tone": safe(analyze_tone, text), "term": safe(analyze_consistency, text), "rep": safe(find_repetition, text),
        "struct": safe(check_structure, text, paper_type), "abs": safe(score_abstract, text),
    }
    dims = score_dimensions(raw)
    overall, fixes = summarize(dims)
    letter, note = grade(overall)
    kw = safe(extract_keywords, text)
    return {"overall": overall, "grade": letter, "note": note, "dims": dims, "fixes": fixes,
            "keywords": kw.get("keywords", []) if _ok(kw) else [],
            "contributions": kw.get("contributions", []) if _ok(kw) else [],
            "word_count": len(text.split()), "error": None}


def report_markdown(res):
    """A downloadable Markdown report of a health check."""
    lines = [f"# Paper Health Check — {res['overall']}/100 (grade {res['grade']})", "", f"_{res['note']}_", "",
             "## Scores", ""]
    lines += [f"- {d['icon']} **{d['name']}**: {d['score']}/100 — {d['summary']}" for d in res["dims"]]
    lines += ["", "## Fix these first", ""]
    lines += [f"{i}. **[{sev}] {name}** — {text}" for i, (sev, _, _, name, text) in enumerate(res["fixes"], 1)] or ["Nothing flagged."]
    if res["keywords"]:
        lines += ["", "## Suggested keywords", "", ", ".join(res["keywords"])]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    # Self-check of the scoring rules (no analyzers needed).
    fake = {
        "read": {"flesch_kincaid_grade": 22, "flesch_reading_ease_label": "x", "long_sentence_count": 1,
                 "long_sentences": [{"words": 40, "sentence": "s"}]},
        "claim": {"uncited_count": 2, "citation_markers": 0, "word_count": 500, "uncited_claims": ["a", "b"]},
        "rep": {"pair_count": 0, "pairs": []},
        "struct": {"completeness_pct": 50, "missing": ["Conclusion"]},
        "abs": {"error": "x"},
    }
    d = {x["name"]: x for x in score_dimensions(fake)}
    assert d["Readability"]["score"] == 100 - 8 * 6 - 3          # grade 22 is 6 above 16
    assert d["Evidence & Citations"]["score"] == 40              # capped: no citations in 500 words
    assert d["Originality of phrasing"]["score"] == 100
    assert "Abstract" not in d                                   # errored analyzer is skipped
    overall, fixes = summarize(list(d.values()))
    assert fixes[0][0] == "Critical"                             # critical issues rank first
    assert overall == round((49 + 40 + 100 + 50) / 4)
    assert grade(95)[0] == "A" and grade(10)[0] == "E" and grade(None)[0] == "—"
    assert run_health_check("too short")["error"]
    print("health_check self-check passed")
