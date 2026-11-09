"""Paper Health Check: scoring rules, ranking, annotation, and a real run on the sample draft.

Run: python -m pytest tests/test_health_check.py   (or plain: python tests/test_health_check.py)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.utils.health_check import (Fix, annotate_html, grade, report_markdown,  # noqa: E402
                                    run_health_check, score_dimensions, summarize)

SAMPLE = open(os.path.join(ROOT, "main", "sample_draft.txt"), encoding="utf-8").read()


def test_scoring_rules():
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
    assert fixes[0].severity == "Critical"                       # critical issues rank first
    assert overall == round((49 + 40 + 100 + 50) / 4)
    assert any(f.action == "cite" and f.sentence == "a" for f in fixes)


def test_grades():
    assert grade(95)[0] == "A" and grade(80)[0] == "B" and grade(10)[0] == "E" and grade(None)[0] == "—"


def test_annotate_escapes_and_tolerates_whitespace():
    text = "Intro <b>here</b>.\nThis   claim\nspans lines. Fine."
    fixes = [Fix("Major", 50, "Tone", 'Overclaim "x"', "This claim spans lines.", None),
             Fix("Minor", 60, "Flow", "dup", "This claim spans lines.", None)]
    out = annotate_html(text, fixes)
    assert "&lt;b&gt;" in out and "<b>" not in out               # draft text is escaped
    assert out.count("<mark") == 1                               # one highlight per sentence
    assert 'class="op-sev-major"' in out                         # most severe issue wins
    assert "&quot;x&quot; · dup" in out                          # all issues listed in the tooltip
    assert annotate_html("abc", [Fix("Minor", 1, "", "t", "not there", None)]) == "abc"


def test_real_run_on_sample_draft():
    res = run_health_check(SAMPLE)
    assert res["error"] is None and 0 <= res["overall"] <= 100
    names = {d["name"] for d in res["dims"]}
    assert {"Readability", "Evidence & Citations", "Tone", "Structure", "Abstract"} <= names
    assert any("Overclaim" in f.text for f in res["fixes"])      # "clearly proves ... always destroy"
    assert any("Over-hedged" in f.text for f in res["fixes"])    # "might perhaps possibly"
    assert "<mark" in res["annotated"]
    assert report_markdown(res).startswith("# Paper Health Check")
    assert run_health_check("too short")["error"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
