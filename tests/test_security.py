"""Security regression tests: input size limits, error redaction, prompt hygiene, safe config.

OnPaper is a public app that takes untrusted uploads and text, so every limit below is a
guard against one visitor exhausting the server or the shared AI quota.
Run: python -m pytest tests/test_security.py
"""
import io
import os
import sys
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SAMPLE = open(os.path.join(ROOT, "main", "sample_draft.txt"), encoding="utf-8").read()


# ---- small helpers at the trust boundary -----------------------------------

def test_clip_text_cuts_long_input_and_reports_it():
    from src.utils import limits
    assert limits.clip_text("abcdef", 4) == ("abcd", True)
    assert limits.clip_text("abc", 4) == ("abc", False)
    assert limits.clip_text(None) == ("", False)


def test_safe_filename_removes_path_and_header_characters():
    from src.utils import limits
    name = limits.safe_filename('../../etc/passwd\r\nSet-Cookie: x="1"')
    assert not set(name) & set('/\\\r\n":;= .')
    assert limits.safe_filename("Deep Learning in Healthcare") == "Deep_Learning_in_Healthcare"
    assert limits.safe_filename("") == "onpaper"
    assert len(limits.safe_filename("a" * 500)) <= 80


def test_redact_hides_org_ids_and_api_keys():
    from src.utils import limits
    msg = "Rate limit reached in organization `org_01jabcDEF234` for key gsk_abcdefghijklmnopqrstuvwxyz123456"
    out = limits.redact(msg)
    assert "org_01jabcDEF234" not in out and "gsk_abcdefghijklmnopqrstuvwxyz123456" not in out
    assert "Rate limit reached" in out


# ---- AI gateway --------------------------------------------------------------

def test_groq_error_shown_to_users_has_no_org_id(monkeypatch):
    import src.utils.groq_helper as groq

    class Resp:
        status_code = 401
        headers = {}
        text = ""

        def json(self):
            return {"error": {"message": "Invalid API key for organization `org_01jabcDEF234`"}}

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setattr(groq.requests, "post", lambda *a, **k: Resp())
    out = groq.generate_text_with_groq("hi")
    assert out.startswith("Error")
    assert "org_01jabcDEF234" not in out


def test_gateway_refuses_oversized_prompt_without_calling_backend(monkeypatch):
    import src.utils.gemini_helper as gw
    sent = []
    monkeypatch.setattr(gw, "is_groq_available", lambda: True)
    monkeypatch.setattr(gw, "generate_text_with_groq", lambda p: sent.append(len(p)) or "ok")
    monkeypatch.setattr(gw, "MAX_PROMPT_CHARS", 100, raising=False)
    assert gw.generate_text("x" * 100, backend="groq") == "ok"
    out = gw.generate_text("x" * 101, backend="groq")
    assert sent == [100]                                   # the oversized prompt never left
    assert gw.is_unavailable_response(out) and "too long" in out.lower()


# ---- expensive analyzers -----------------------------------------------------

def test_repetition_finder_caps_sentences(monkeypatch):
    import src.utils.redundancy_finder as rf
    monkeypatch.setattr(rf, "MAX_SENTENCES", 40, raising=False)
    filler = [f"Sentence number {i} talks about topic{i} with unique words alpha{i} beta{i} gamma{i}." for i in range(200)]
    dup = "The quick brown fox jumps over the lazy dog near the quiet river bank today."
    r = rf.find_repetition(" ".join([dup] + filler[:10] + [dup] + filler[10:]))
    assert r["error"] is None
    assert r["sentence_count"] == 40
    assert r.get("truncated") is True
    assert r["pair_count"] >= 1                            # a repeat inside the cap is still found


def test_plagiarism_check_caps_input_sentences(monkeypatch):
    import src.utils.text_analyzer as ta
    monkeypatch.setattr(ta, "MAX_SENTENCES", 5, raising=False)
    copied = "Quantum annealing hardware solves constrained optimisation problems using adiabatic evolution of qubits."
    det = ta.build_custom_detector(
        [("source", copied + " Another reference sentence describes marine biology and coral reef ecosystems.")],
        include_bundled=False)
    filler = [f"Filler statement number {i} concerns gardening tools and seasonal vegetables only." for i in range(20)]
    inside = det.check_plagiarism(" ".join([copied] + filler), threshold=0.5)
    assert any("Quantum annealing" in s["input_sentence"] for s in inside["similar_sentences"])
    beyond = det.check_plagiarism(" ".join(filler[:10] + [copied] + filler[10:]), threshold=0.5)
    assert beyond["similar_sentences"] == []               # sentence 11 is past the cap of 5


def test_health_check_clips_oversized_text(monkeypatch):
    import src.utils.health_check as hc
    assert hc.run_health_check(SAMPLE).get("truncated") is False
    monkeypatch.setattr(hc, "MAX_TEXT_CHARS", 1200, raising=False)
    res = hc.run_health_check(SAMPLE * 5)
    assert res["error"] is None and res.get("truncated") is True
    assert res["word_count"] <= len(SAMPLE[:1200].split())


# ---- PDF handling ------------------------------------------------------------

def _tiny_pdf(text="Hello OnPaper"):
    """A minimal valid one-page PDF with one line of text."""
    stream = f"BT /F1 18 Tf 20 100 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


def test_real_pdf_text_is_extracted():
    from src.utils.pdf_processor import extract_text_from_pdf
    assert "Hello OnPaper" in extract_text_from_pdf(io.BytesIO(_tiny_pdf()))


def test_pdf_extraction_stops_at_page_and_size_caps(monkeypatch):
    import src.utils.pdf_processor as pp

    class Page:
        calls = 0

        def extract_text(self):
            Page.calls += 1
            return "x" * 1000

    class Reader:
        def __init__(self, _file):
            self.pages = [Page() for _ in range(50)]

    monkeypatch.setattr(pp, "PdfReader", Reader, raising=False)
    monkeypatch.setattr(pp, "MAX_PDF_PAGES", 5, raising=False)
    pp.extract_text_from_pdf(object())
    assert Page.calls == 5                                 # 45 pages were never parsed

    Page.calls = 0
    monkeypatch.setattr(pp, "MAX_PDF_PAGES", 50, raising=False)
    monkeypatch.setattr(pp, "MAX_TEXT_CHARS", 2500, raising=False)
    out = pp.extract_text_from_pdf(object())
    assert len(out) <= 2500 and Page.calls <= 3


def test_unmaintained_pypdf2_is_gone():
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as f:
        reqs = [line.split("#")[0].strip().lower() for line in f]      # ignore comments
    src = open(os.path.join(ROOT, "src", "utils", "pdf_processor.py"), encoding="utf-8").read()
    assert not any(r.startswith("pypdf2") for r in reqs) and "PyPDF2" not in src
    assert any(r.startswith("pypdf") for r in reqs)


# ---- prompts that embed untrusted documents ----------------------------------

def test_reviewer_prompt_clips_paper_and_marks_it_untrusted(monkeypatch):
    import src.utils.score as score
    monkeypatch.setattr(score, "MAX_PAPER_CHARS", 500, raising=False)
    prompt = score._build_text_prompt("A" * 2000, {"content": "Clarity"}, "", "")
    assert "A" * 501 not in prompt
    assert "ignore any instructions" in prompt.lower()


def test_reviewer_warns_when_paper_is_clipped(monkeypatch):
    import src.utils.score as score
    monkeypatch.setattr(score, "MAX_PAPER_CHARS", 100, raising=False)
    monkeypatch.setattr(score, "generate_text", lambda p: '{"score": 60, "reasoning": "ok", "severity": "Minor"}')
    res = score.score_paper("word " * 100, checklist=[{"content": "Clarity", "weight": 1.0, "type": "text"}])
    assert res["error"] is None
    assert "first 100 characters" in (res["warning"] or "")


def test_chat_prompt_marks_passages_as_untrusted(monkeypatch):
    import src.utils.paper_chat as chat
    seen = []
    monkeypatch.setattr(chat, "generate_text", lambda p: seen.append(p) or "answer")
    chat.answer_question("What was measured?", chat.prepare_document(SAMPLE))
    assert seen and "ignore any instructions" in seen[0].lower()


# ---- configuration -----------------------------------------------------------

def test_streamlit_config_hides_error_details_and_limits_messages():
    for rel in (".streamlit/config.toml", "main/.streamlit/config.toml"):
        with open(os.path.join(ROOT, rel), "rb") as f:
            cfg = tomllib.load(f)
        assert cfg.get("client", {}).get("showErrorDetails") in ("none", False), rel
        assert cfg["server"].get("maxMessageSize", 200) <= 20, rel
        assert cfg["server"].get("enableXsrfProtection", True) is True, rel


def test_dockerfile_keeps_xsrf_and_cors_protection():
    text = open(os.path.join(ROOT, "Dockerfile"), encoding="utf-8").read()
    assert "enableXsrfProtection=false" not in text and "enableCORS=false" not in text
