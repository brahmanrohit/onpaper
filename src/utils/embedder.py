"""
Sentence-embedding helper for semantic search.

Produces dense vector embeddings using a small pretrained model
(all-MiniLM-L6-v2) via the already-installed `transformers` library — no extra
pip dependency. Embeddings are mean-pooled over tokens and L2-normalized, so a
plain dot product equals cosine similarity.

The model is loaded lazily on first use (~90 MB, downloaded once and cached).
If torch/transformers are unavailable, is_available() returns False and callers
fall back to lexical (TF-IDF) search.
"""

from typing import List, Optional
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_tokenizer = None
_model = None
_loaded = False
_load_error: Optional[str] = None


def _load():
    """Lazily load the embedding model once. Returns (tokenizer, model) or (None, None)."""
    global _tokenizer, _model, _loaded, _load_error
    if _loaded:
        return _tokenizer, _model
    _loaded = True
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModel
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
    except Exception as e:
        _tokenizer, _model = None, None
        _load_error = str(e)
    return _tokenizer, _model


def is_available() -> bool:
    """True if the embedding model can be loaded."""
    tok, mdl = _load()
    return tok is not None and mdl is not None


def embed(texts: List[str], batch_size: int = 16) -> Optional[np.ndarray]:
    """Embed a list of texts into L2-normalized vectors (shape: len(texts) x dim).

    Returns None if the model is unavailable.
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    tok, mdl = _load()
    if tok is None or mdl is None:
        return None

    import torch
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            out = mdl(**enc)
        token_emb = out.last_hidden_state                      # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1).float()     # (B, T, 1)
        summed = (token_emb * mask).sum(dim=1)                 # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-9)               # (B, 1)
        mean = summed / counts
        mean = torch.nn.functional.normalize(mean, p=2, dim=1)
        vectors.append(mean.cpu().numpy())
    return np.vstack(vectors).astype(np.float32)


def cosine_sim(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one normalized query vector against a normalized matrix."""
    if matrix is None or matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    return matrix @ query_vec  # both already L2-normalized
