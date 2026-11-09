"""End-to-end UI tests: drive the real Streamlit app headlessly (streamlit.testing).

Network and AI calls are faked, so these are fast and deterministic and can gate
every deploy. Run: python -m pytest tests/
"""
import os
import sys

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
APP = os.path.join(ROOT, "main", "main.py")

import src.utils.gemini_helper as gateway  # noqa: E402
import src.utils.paraphraser as paraphraser  # noqa: E402
import src.utils.reference_finder as refs  # noqa: E402

PAGES = ["Home", "Paper Health Check", "Readability & Quality", "Plagiarism Detection",
         "ZeroGPT — Writing Analysis Suite", "AI Peer Reviewer", "Content Generation", "Paraphraser",
         "Grammar Check", "AI Prompt Generator", "Reference Finder", "Verify References",
         "Chat with your Paper", "Paper Analysis"]


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    """No network, a fake 'connected' AI backend, and a fresh cache per test."""
    monkeypatch.setattr(gateway, "get_backend_status",
                        lambda: {"groq": {"available": True, "model": "fake"}, "ollama": {"available": False}})
    monkeypatch.setattr(paraphraser, "paraphrase_text",
                        lambda text, mode="x": {"result": "Late-evening phone use was linked to shorter sleep.",
                                                "error": None})
    monkeypatch.setattr(refs, "search_references", lambda q, rows=8: {
        "references": [{"title": f"A study on {q[:20]}", "authors": [{"given": "A", "family": "Author"}],
                        "year": "2020", "journal": "J", "doi": "10.1/x", "url": "", "type": "journal-article"}],
        "error": None})
    st.cache_data.clear()


def app(page="Home"):
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["feature_menu"] = page
    return at.run()


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    at = app(page)
    assert not at.exception, at.exception[0].value


def test_sample_draft_runs_health_check_and_follows_you():
    at = app("Home")
    at.button[[b.label for b in at.button].index("See it on a sample draft")].click().run()
    assert at.session_state["feature_menu"] == "Paper Health Check"
    res = at.session_state["hc_result"]
    assert res["error"] is None and res["overall"] > 0
    assert any("op-doc" in m.value for m in at.markdown)            # annotated draft rendered
    # the same draft is already waiting in another tool
    at.session_state["feature_menu"] = "Readability & Quality"
    at.run()
    assert at.text_area(key="read_text").value.startswith("Abstract")
    assert not at.exception


def test_rewrite_then_apply_updates_draft_and_score_history():
    at = app("Home")
    at.button[[b.label for b in at.button].index("See it on a sample draft")].click().run()
    fixes = at.session_state["hc_result"]["fixes"]
    i = next(n for n, f in enumerate(fixes) if f.action == "Calibrate claim" and "clearly proves" in f.sentence)
    at.button(key=f"hc_rw_{i}").click().run()
    at.button(key=f"hc_apply_{i}").click().run()
    assert "clearly proves" not in at.session_state["draft"]
    assert "linked to shorter sleep" in at.session_state["draft"]
    assert len(at.session_state["hc_history"]) == 2                 # re-checked automatically
    assert not at.exception


def test_find_a_source_opens_reference_finder_with_results():
    at = app("Home")
    at.button[[b.label for b in at.button].index("See it on a sample draft")].click().run()
    fixes = at.session_state["hc_result"]["fixes"]
    i = next(n for n, f in enumerate(fixes) if f.action == "cite")
    at.button(key=f"hc_cite_{i}").click().run()
    assert at.session_state["feature_menu"] == "Reference Finder"
    assert at.text_input(key="ref_query").value
    assert any("Found 1 reference" in s.value for s in at.success)  # searched without a click
    assert not at.exception


def test_send_to_paraphraser_from_readability():
    at = app("Readability & Quality")
    at.text_area(key="read_text").set_value(" ".join(["the participants reported that"] * 8) + ". Short.").run()
    at.button(key="read_btn").click().run()
    at.button(key="para_long_0").click().run()
    assert at.header[0].value.startswith("Paraphraser")
    assert at.text_area(key="para_text").value.startswith("the participants")


def test_clear_draft():
    at = app("Home")
    at.button[[b.label for b in at.button].index("See it on a sample draft")].click().run()
    at.button(key="draft_clear").click().run()
    assert not at.session_state["draft"] if "draft" in at.session_state else True
    assert at.text_area(key="hc_text").value == ""
