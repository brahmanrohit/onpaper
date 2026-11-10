"""
Paper Health Check — one click, every offline analyzer, one report card.

Runs the existing offline checkers (readability, citations, flow, tone,
terminology, repetition, structure, abstract, keywords) on a single draft and
turns their raw output into 0-100 sub-scores, an overall score, a ranked
"fix these first" list, and an annotated copy of the draft with every flagged
sentence highlighted. No AI backend or network needed.

Scoring is deliberately simple and explainable (each rule is one line below) so
a writer can see exactly why a dimension lost points.
"""

import html
import re
from collections import namedtuple

from .readability_analyzer import analyze_text
from .cohesion_analyzer import analyze_cohesion
from .claim_checker import analyze_claims
from .structure_checker import check_structure
from .abstract_scorecard import score_abstract
from .tone_auditor import analyze_tone
from .consistency_checker import analyze_consistency
from .redundancy_finder import find_repetition
from .keyword_extractor import extract_keywords
from .limits import MAX_TEXT_CHARS, clip_text

# Academic prose usually lands at Flesch-Kincaid grade 12-16.
GRADE_LOW, GRADE_HIGH = 12, 16

# sentence: the exact draft sentence the issue is about (for highlighting), or None.
# action: what the UI can offer — "cite" (find a source), a Paraphraser mode name
#         (one-click rewrite), or None.
Issue = namedtuple("Issue", "severity text sentence action", defaults=(None, None))
Fix = namedtuple("Fix", "severity dim_score dim text sentence action")


def _clamp(x):
    return int(max(0, min(100, round(x))))


def _ok(r):
    return isinstance(r, dict) and not r.get("error")


def score_dimensions(raw):
    """Turn raw analyzer outputs into dimensions. Pure function (easy to test).

    raw: {"read", "claim", "coh", "tone", "term", "rep", "struct", "abs"} -> analyzer dicts.
    Returns a list of {name, score, summary, issues:[Issue]}; an analyzer
    that errored or doesn't apply is simply left out (and so not averaged in).
    """
    dims = []

    r = raw.get("read")
    if _ok(r):
        g = r["flesch_kincaid_grade"]
        dist = max(0, GRADE_LOW - g, g - GRADE_HIGH)
        issues = [Issue("Major" if len(r["long_sentences"]) > 3 else "Minor",
                        f"Long sentence ({s['words']} words) — split it.", s["sentence"], "Improve clarity")
                  for s in r["long_sentences"][:5]]
        if dist:
            issues.insert(0, Issue("Major" if dist > 4 else "Minor",
                                   f"Grade level {g} is outside the academic sweet spot ({GRADE_LOW}-{GRADE_HIGH})."))
        dims.append(dict(name="Readability",
                         score=_clamp(100 - 8 * dist - 3 * r["long_sentence_count"]),
                         summary=f"Grade {g} · {r['flesch_reading_ease_label']} · {r['long_sentence_count']} long sentence(s)",
                         issues=issues))

    c = raw.get("claim")
    if _ok(c):
        s = _clamp(100 - 12 * c["uncited_count"])
        no_cites = c["word_count"] >= 300 and c["citation_markers"] == 0
        if no_cites:
            s = min(s, 40)  # a paper-length draft with zero citations is a red flag
        issues = [Issue("Major", "Claim with no citation — back it with a source.", x, "cite")
                  for x in c["uncited_claims"][:6]]
        if no_cites:
            issues.insert(0, Issue("Critical", "No in-text citations found anywhere in the draft."))
        dims.append(dict(name="Evidence & Citations", score=s,
                         summary=f"{c['citation_markers']} citation(s) · {c['uncited_count']} uncited claim(s)",
                         issues=issues))

    h = raw.get("coh")
    if _ok(h) and h.get("pairs"):
        weak = [p for p in h["pairs"] if p["weak"]]
        dims.append(dict(name="Flow",
                         score=_clamp(100 * (1 - len(weak) / len(h["pairs"]))),
                         summary=f"{h['paragraphs']} paragraphs · {len(weak)} abrupt jump(s)",
                         issues=[Issue("Minor", f"Abrupt jump from ¶{p['from']} to ¶{p['to']} — add a transition "
                                                f"(¶{p['to']} starts “{p['preview']}”).") for p in weak[:3]]))

    t = raw.get("tone")
    if _ok(t):
        over, hedge = t["examples"]["overclaim"], t["examples"]["over_hedged"]
        dims.append(dict(name="Tone", score=_clamp(100 - 10 * (len(over) + len(hedge))),
                         summary=f"{len(over)} overclaim(s) · {len(hedge)} over-hedged sentence(s)",
                         issues=[Issue("Major", "Overclaim — absolute wording a reviewer will challenge.", x, "Calibrate claim")
                                 for x in over[:4]]
                         + [Issue("Minor", "Over-hedged — commit to the claim.", x, "Calibrate claim") for x in hedge[:3]]))

    m = raw.get("term")
    if _ok(m):
        n = len(m["acronym_issues"]) + len(m["term_variants"])
        dims.append(dict(name="Terminology", score=_clamp(100 - 8 * n),
                         summary=f"{len(m['acronym_issues'])} acronym issue(s) · {len(m['term_variants'])} spelling variant(s)",
                         issues=[Issue("Minor", f"{a['acronym']}: {a['issue']}") for a in m["acronym_issues"][:4]]
                         + [Issue("Minor", "Inconsistent spelling: " + ", ".join(v["forms"])) for v in m["term_variants"][:3]]))

    p = raw.get("rep")
    if _ok(p):
        dims.append(dict(name="Originality of phrasing", score=_clamp(100 - 15 * p["pair_count"]),
                         summary=f"{p['pair_count']} near-duplicate sentence pair(s)",
                         issues=[Issue("Minor", f"Repeats sentence {x['i']} almost word for word — cut or merge.",
                                       x["b"], "Standard rewrite") for x in p["pairs"][:3]]))

    st_ = raw.get("struct")
    if _ok(st_):
        dims.append(dict(name="Structure", score=_clamp(st_["completeness_pct"]),
                         summary=f"{st_['completeness_pct']}% of expected sections present",
                         issues=[Issue("Critical" if sec in ("Abstract", "Introduction", "Conclusion") else "Major",
                                       f"Missing section: {sec}") for sec in st_["missing"]]))

    a = raw.get("abs")
    if _ok(a) and a.get("had_abstract_heading"):
        dims.append(dict(name="Abstract", score=_clamp(a["score"]),
                         summary=f"{a['moves_present']}/{a['moves_total']} expected moves · length {a['length_status']}",
                         issues=[Issue("Major", f"Abstract is missing the '{mv}' move.") for mv, ok in a["moves"].items() if not ok]
                         + [Issue("Minor", tip) for tip in a["tips"][:2]]))
    return dims


_SEV_RANK = {"Critical": 0, "Major": 1, "Minor": 2}


def summarize(dims):
    """Overall score + every issue as a Fix, ranked by severity, then by how weak its dimension is."""
    overall = _clamp(sum(d["score"] for d in dims) / len(dims)) if dims else None
    # dict.fromkeys de-duplicates: a sentence repeated in the draft is one fix, not two.
    fixes = list(dict.fromkeys(Fix(i.severity, d["score"], d["name"], i.text, i.sentence, i.action)
                               for d in dims for i in d["issues"]))
    fixes.sort(key=lambda f: (_SEV_RANK.get(f.severity, 3), f.dim_score))
    return overall, fixes


def grade(overall):
    if overall is None:
        return "—", "Not enough text to judge"
    for cutoff, letter, note in ((90, "A", "Ready for a reviewer"), (80, "B", "Strong — polish the flagged spots"),
                                 (65, "C", "Solid draft — a revision pass will lift it"),
                                 (50, "D", "Needs work on the critical items"), (0, "E", "Early draft — start with the top fixes")):
        if overall >= cutoff:
            return letter, note


def locate_sentence(text, sentence):
    """(start, end) of `sentence` in `text`, tolerant of whitespace differences; None if absent."""
    words = (sentence or "").split()
    if not words:
        return None
    m = re.search(r"\s+".join(map(re.escape, words)), text)
    return m.span() if m else None


def annotate_html(text, fixes):
    """The draft as HTML with each flagged sentence wrapped in <mark class="op-sev-…" title="issue">.

    One highlight per sentence (its most severe issue wins; `fixes` is already
    severity-sorted), overlaps skipped, everything HTML-escaped.
    """
    spans, notes = [], {}
    for f in fixes:
        loc = locate_sentence(text, f.sentence)
        if not loc:
            continue
        if loc in notes:
            notes[loc].append(f.text)
            continue
        if any(loc[0] < e and s < loc[1] for s, e, _ in spans):
            continue
        notes[loc] = [f.text]
        spans.append((loc[0], loc[1], f.severity))
    out, pos = [], 0
    for s, e, sev in sorted(spans):
        tip = html.escape(" · ".join(dict.fromkeys(notes[(s, e)])), quote=True)
        out += [html.escape(text[pos:s]), f'<mark class="op-sev-{sev.lower()}" title="{tip}">',
                html.escape(text[s:e]), "</mark>"]
        pos = e
    out.append(html.escape(text[pos:]))
    return "".join(out).replace("\n", "<br>")


def run_health_check(text, paper_type="empirical"):
    """Run every offline analyzer on `text`.

    Returns {overall, grade, note, dims, fixes, annotated, keywords, contributions, word_count,
    truncated, error}. Text beyond MAX_TEXT_CHARS is not analysed (truncated=True).
    """
    text, truncated = clip_text((text or "").strip(), MAX_TEXT_CHARS)
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
            "annotated": annotate_html(text, fixes),
            "keywords": kw.get("keywords", []) if _ok(kw) else [],
            "contributions": kw.get("contributions", []) if _ok(kw) else [],
            "word_count": len(text.split()), "truncated": truncated, "error": None}


def report_markdown(res):
    """A downloadable Markdown report of a health check."""
    lines = [f"# Paper Health Check — {res['overall']}/100 (grade {res['grade']})", "", f"_{res['note']}_", "",
             "## Scores", ""]
    lines += [f"- **{d['name']}**: {d['score']}/100 — {d['summary']}" for d in res["dims"]]
    lines += ["", "## Fix these first", ""]
    for i, f in enumerate(res["fixes"], 1):
        lines.append(f"{i}. **[{f.severity}] {f.dim}** — {f.text}")
        if f.sentence:
            lines.append(f"   > {' '.join(f.sentence.split())}")
    if not res["fixes"]:
        lines.append("Nothing flagged.")
    if res["keywords"]:
        lines += ["", "## Suggested keywords", "", ", ".join(res["keywords"])]
    return "\n".join(lines) + "\n"
