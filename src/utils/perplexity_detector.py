"""
Perplexity-based AI text detector (GPTZero-style).

Requires NO training data. It uses a pretrained language model (distilGPT-2)
to measure how "predictable" text is:

  * Perplexity  - lower perplexity means the text is more predictable, which is
                  characteristic of AI-generated writing.
  * Burstiness  - humans vary surprise sentence-to-sentence; AI is steadier.
                  Measured as the variation of per-sentence perplexity.

The model and tokenizer are loaded lazily on first use (a few hundred MB,
downloaded once and cached by the transformers library). All inference runs on
CPU. If torch/transformers are unavailable, the module reports that gracefully.
"""

import math
from typing import Dict, Optional
from nltk.tokenize import sent_tokenize

MODEL_NAME = "distilbert/distilgpt2"  # canonical HF id (the bare "distilgpt2" alias is legacy)

# Calibration anchors for mapping perplexity -> AI-likelihood. Below LOW_PPL the
# text looks strongly AI-like; above HIGH_PPL it looks strongly human. These are
# rough, model-dependent heuristics, not precise thresholds.
LOW_PPL = 10.0
HIGH_PPL = 60.0

_tokenizer = None
_model = None
_load_error: Optional[str] = None
_loaded = False


def _load_model():
    """Lazily load distilGPT-2 once. Returns (tokenizer, model) or (None, None)."""
    global _tokenizer, _model, _load_error, _loaded
    if _loaded:
        return _tokenizer, _model
    _loaded = True
    try:
        import torch  # noqa: F401
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        _tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
        _model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        _model.eval()
    except Exception as e:  # ImportError or download/runtime failure
        _tokenizer, _model = None, None
        _load_error = str(e)
    return _tokenizer, _model


def is_available() -> bool:
    """True if the perplexity model can be loaded."""
    tok, mdl = _load_model()
    return tok is not None and mdl is not None


def _text_perplexity(text: str) -> Optional[float]:
    """Compute the perplexity of a text under distilGPT-2 (CPU)."""
    import torch
    tok, mdl = _load_model()
    if tok is None or mdl is None:
        return None
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"]
    if input_ids.shape[1] < 2:
        return None
    with torch.no_grad():
        out = mdl(input_ids, labels=input_ids)
    # out.loss is mean cross-entropy (nats); perplexity = exp(loss)
    return float(torch.exp(out.loss).item())


def _ppl_to_ai_score(ppl: float) -> float:
    """Map perplexity to a 0-100 AI-likelihood (lower perplexity -> more AI)."""
    if ppl <= LOW_PPL:
        return 100.0
    if ppl >= HIGH_PPL:
        return 0.0
    # Linear interpolation between the anchors.
    return round(100.0 * (HIGH_PPL - ppl) / (HIGH_PPL - LOW_PPL), 1)


def detect_perplexity(text: str) -> Dict:
    """Score text for AI-likelihood using perplexity + burstiness.

    Returns {"available": bool, "ai_score": float, "perplexity": float,
             "burstiness": float, "error": Optional[str]}.
    """
    text = (text or "").strip()
    if not text:
        return {"available": True, "error": "Please enter some text to analyze."}

    if not is_available():
        return {"available": False,
                "error": "Perplexity model unavailable. Install torch + transformers.",
                "detail": _load_error}

    overall_ppl = _text_perplexity(text)
    if overall_ppl is None:
        return {"available": True, "error": "Text too short for perplexity analysis."}

    # Per-sentence perplexity for burstiness.
    sentences = [s for s in sent_tokenize(text) if len(s.split()) >= 4]
    sent_ppls = []
    for s in sentences:
        p = _text_perplexity(s)
        if p is not None and math.isfinite(p):
            sent_ppls.append(p)

    burstiness = 0.0
    if len(sent_ppls) > 1:
        mean = sum(sent_ppls) / len(sent_ppls)
        var = sum((p - mean) ** 2 for p in sent_ppls) / len(sent_ppls)
        burstiness = math.sqrt(var)

    ai_score = _ppl_to_ai_score(overall_ppl)

    return {
        "available": True,
        "ai_score": ai_score,
        "perplexity": round(overall_ppl, 1),
        "burstiness": round(burstiness, 1),
        "sentences_scored": len(sent_ppls),
        "error": None,
    }
