"""Groq retires models a few times a year. When the configured one is gone, the app must move to
the next model instead of failing every AI feature (this happened: llama-3.3-70b-versatile was
shut down on 2026-08-16 and the live site answered 404 on every request).

Run: python -m pytest tests/test_groq_models.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import src.utils.groq_helper as groq  # noqa: E402


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body, self.headers, self.text = status, body, {}, ""

    def json(self):
        return self._body


def gone(model):
    return Resp(404, {"error": {"message": f"The model `{model}` does not exist or you do not have access to it.",
                                "code": "model_not_found"}})


def answer(text):
    return Resp(200, {"choices": [{"message": {"content": text}}]})


def setup(monkeypatch, dead, defaults=("gone-model", "live-model")):
    """Fake Groq: models in `dead` answer 404, every other model answers 'hello'. Returns the call log."""
    calls = []

    def post(url, headers=None, json=None, timeout=None):
        calls.append(json["model"])
        return gone(json["model"]) if json["model"] in dead else answer("hello")

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.setattr(groq, "DEFAULT_MODELS", defaults, raising=False)
    monkeypatch.setattr(groq, "_working_model", None, raising=False)
    monkeypatch.setattr(groq.requests, "post", post)
    return calls


def test_moves_to_the_next_model_when_one_is_shut_down(monkeypatch):
    calls = setup(monkeypatch, dead={"gone-model"})
    assert groq.generate_text_with_groq("hi") == "hello"
    assert calls == ["gone-model", "live-model"]
    assert groq.get_groq_model() == "live-model"            # the sidebar names the model that works
    assert groq.generate_text_with_groq("again") == "hello"
    assert calls == ["gone-model", "live-model", "live-model"]   # the dead model is not tried again


def test_a_model_the_owner_chose_is_tried_first_and_also_falls_back(monkeypatch):
    calls = setup(monkeypatch, dead={"my-old-model"})
    monkeypatch.setenv("GROQ_MODEL", "my-old-model")
    assert groq.generate_text_with_groq("hi") == "hello"
    assert calls[0] == "my-old-model" and calls[-1] == "gone-model"


def test_reports_the_provider_error_when_every_model_is_gone(monkeypatch):
    calls = setup(monkeypatch, dead={"gone-model", "live-model"})
    out = groq.generate_text_with_groq("hi")
    assert out.startswith("Error") and "404" in out
    assert calls == ["gone-model", "live-model"]


def test_a_bad_key_fails_fast_without_trying_other_models(monkeypatch):
    calls = setup(monkeypatch, dead=set())
    monkeypatch.setattr(groq.requests, "post",
                        lambda url, headers=None, json=None, timeout=None: (calls.append(json["model"]),
                                                                            Resp(401, {"error": {"message": "Invalid API Key"}}))[1])
    out = groq.generate_text_with_groq("hi")
    assert out.startswith("Error") and "401" in out
    assert len(calls) == 1


def test_shipped_defaults_do_not_include_retired_models():
    retired = {"llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3.6-27b"}   # Groq deprecations page
    assert not retired & set(groq.DEFAULT_MODELS)
