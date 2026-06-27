"""
Chat with your Paper (hybrid RAG).

Retrieval combines two signals and works fully offline:

  * Semantic search - dense MiniLM sentence embeddings (transformers), so
    paraphrased questions match relevant passages even with no shared words.
  * Lexical search  - TF-IDF + cosine, strong for exact terms, names, numbers.

When both are available they are fused with Reciprocal Rank Fusion (RRF); if the
embedding model can't load, retrieval gracefully falls back to TF-IDF only.

Only the final answer generation uses the shared AI gateway (Groq/Ollama);
the retrieved passages are always returned, so the feature stays useful offline.
Conversation history is supported for follow-up questions.
"""

import re
from typing import Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
from .gemini_helper import generate_text, is_unavailable_response
from . import embedder
from .section_splitter import split_sections

try:
    import nltk
    nltk.data.find("tokenizers/punkt")
except Exception:
    try:
        import nltk
        nltk.download("punkt")
    except Exception:
        pass


def chunk_text(text: str, target_words: int = 130, overlap_sentences: int = 1) -> List[str]:
    """Split text into sentence-aware chunks of ~target_words, with sentence overlap.

    Grouping by whole sentences (instead of arbitrary word windows) keeps each
    chunk coherent and avoids cutting an answer mid-sentence.
    """
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []

    try:
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
    if not sentences:
        return []

    chunks = []
    current: List[str] = []
    current_words = 0
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        current.append(sent)
        current_words += len(sent.split())
        if current_words >= target_words:
            chunks.append(" ".join(current))
            # Start next chunk a few sentences back for context overlap.
            overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = list(overlap)
            current_words = sum(len(s.split()) for s in current)
        i += 1
    if current:
        chunks.append(" ".join(current))
    # De-duplicate while preserving order (overlap can repeat a tail sentence).
    seen = set()
    unique = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _label_chunk_sections(chunks: List[str], text: str) -> List[Optional[str]]:
    """Best-effort: tag each chunk with the document section it came from.

    Assigns each chunk to the section whose body shares the most words with it
    (robust to chunks that include a heading word or straddle a boundary).
    Returns None for a chunk when no section overlaps at all.
    """
    secs = split_sections(text)
    if not secs:
        return [None] * len(chunks)
    sec_tokens = [(name, set(re.findall(r"[a-z0-9]+", body.lower()))) for name, body in secs.items()]
    labels = []
    for c in chunks:
        ctoks = set(re.findall(r"[a-z0-9]+", c.lower()))
        best, best_overlap = None, 0
        for name, toks in sec_tokens:
            overlap = len(ctoks & toks)
            if overlap > best_overlap:
                best, best_overlap = name, overlap
        labels.append(best)
    return labels


def prepare_document(text: str) -> Dict:
    """Index the document chunks with TF-IDF and (if available) dense embeddings."""
    chunks = chunk_text(text)
    if not chunks:
        return {"chunks": [], "error": "No readable text found in the document."}

    # Lexical index (always available)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000)
    try:
        tfidf_matrix = vectorizer.fit_transform(chunks)
    except ValueError:
        vectorizer = TfidfVectorizer(ngram_range=(1, 1))
        tfidf_matrix = vectorizer.fit_transform(chunks)

    # Semantic index (optional)
    embeddings = embedder.embed(chunks) if embedder.is_available() else None
    mode = "hybrid (semantic + lexical)" if embeddings is not None else "lexical (TF-IDF)"

    return {
        "chunks": chunks,
        "chunk_sections": _label_chunk_sections(chunks, text),
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "embeddings": embeddings,
        "mode": mode,
        "error": None,
    }


def _rank(sims: np.ndarray) -> List[int]:
    """Return indices ordered by descending similarity."""
    return sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)


def retrieve_context(question: str, doc: Dict, k: int = 4) -> List[Dict]:
    """Return the top-k most relevant chunks (with section label + score).

    Returns the top-k regardless of absolute similarity (so the assistant can
    answer readily); the per-passage `score` lets the caller judge how well the
    paper actually covers the question.
    """
    question = (question or "").strip()
    if not question or not doc or not doc.get("chunks"):
        return []

    chunks = doc["chunks"]
    n = len(chunks)
    sections = doc.get("chunk_sections") or [None] * n

    # Lexical similarities
    q_tfidf = doc["vectorizer"].transform([question])
    lex_sims = cosine_similarity(q_tfidf, doc["tfidf_matrix"])[0]

    # Semantic similarities (if available)
    sem_sims = None
    embeddings = doc.get("embeddings")
    if embeddings is not None:
        q_emb = embedder.embed([question])
        if q_emb is not None:
            sem_sims = embedder.cosine_sim(q_emb[0], embeddings)

    if sem_sims is not None:
        # Reciprocal Rank Fusion of the two rankings (robust, scale-free).
        rrf_k = 60
        scores = {i: 0.0 for i in range(n)}
        for ranking in (_rank(lex_sims), _rank(sem_sims)):
            for rank, idx in enumerate(ranking):
                scores[idx] += 1.0 / (rrf_k + rank)
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
        # Clamp the displayed relevance to [0, 1]; raw cosine can be slightly
        # negative for off-topic passages, which reads oddly in the UI.
        return [{"chunk": chunks[i], "score": max(0.0, float(sem_sims[i])), "section": sections[i]} for i in order]

    # Lexical only
    order = _rank(lex_sims)[:k]
    return [{"chunk": chunks[i], "score": float(lex_sims[i]), "section": sections[i]} for i in order]


def _format_history(history: Optional[List[Dict]], max_turns: int = 3) -> str:
    """Render recent Q/A turns for follow-up context."""
    if not history:
        return ""
    recent = history[-max_turns:]
    lines = [f"Q: {h['question']}\nA: {h['answer']}" for h in recent if h.get("answer")]
    return ("\n\n".join(lines) + "\n\n") if lines else ""


def answer_question(question: str, doc: Dict, k: int = 4,
                    history: Optional[List[Dict]] = None,
                    detailed: bool = True, use_general: bool = True) -> Dict:
    """Answer a question about the prepared document.

    detailed=True  -> thorough, structured answer (else concise).
    use_general=True -> may supplement with general knowledge, clearly labeled
                        as 'Beyond the paper:'. Otherwise answers ONLY from the
                        document. 'passages' is always returned for transparency.
    """
    question = (question or "").strip()
    if not question:
        return {"answer": "", "passages": [], "error": "Please enter a question."}
    if not doc or not doc.get("chunks"):
        return {"answer": "", "passages": [], "error": "Please load a document first."}

    passages = retrieve_context(question, doc, k=k)
    max_score = max((p.get("score", 0.0) for p in passages), default=0.0)
    grounded = max_score >= 0.15  # does the paper seem to actually cover this?

    context = "\n\n".join(
        f"[Passage {i+1}{(' · ' + p['section']) if p.get('section') else ''}]\n{p['chunk']}"
        for i, p in enumerate(passages)
    )
    history_block = _format_history(history)
    history_section = ("Recent conversation (for follow-ups):\n" + history_block) if history_block else ""

    length_instr = (
        "Give a thorough, well-structured answer with relevant detail and examples from the paper."
        if detailed else "Answer concisely (about 2-4 sentences)."
    )
    if use_general:
        grounding_instr = (
            "Use the document passages below as your PRIMARY source. You may add relevant general "
            "knowledge to give a fuller answer, but you MUST clearly attribute it: prefix statements "
            "drawn from the document with 'From the paper:' and any outside knowledge with "
            "'Beyond the paper:'. If the passages do not address the question, say so briefly, then "
            "answer from general knowledge under 'Beyond the paper:'."
        )
    else:
        grounding_instr = (
            "Answer using ONLY the document passages below. If the answer is not in them, say you "
            "could not find it in the document. Do not use outside knowledge."
        )

    prompt = (
        grounding_instr + " " + length_instr + "\n\n"
        + history_section
        + f"Document passages:\n{context}\n\n"
        + f"Question: {question}\n\nAnswer:"
    )

    result = generate_text(prompt)
    if is_unavailable_response(result):
        return {"answer": "", "passages": passages, "error": result, "grounded": grounded}
    return {"answer": result.strip(), "passages": passages, "error": None, "grounded": grounded}


def document_overview(doc: Dict, num_questions: int = 4) -> Dict:
    """Generate a short summary + a few suggested questions for a loaded document.

    Returns {"summary", "questions", "error"}. Needs an AI backend; degrades to
    an error string (offline) which the UI can simply skip.
    """
    chunks = (doc or {}).get("chunks") or []
    if not chunks:
        return {"summary": "", "questions": [], "error": "No document loaded."}

    context = "\n\n".join(chunks[:6])[:4000]
    prompt = (
        "Below is the beginning of a document. First, in 2-3 sentences, summarize what it is about. "
        f"Then list exactly {num_questions} specific questions a reader could ask that this document "
        "would answer. Respond EXACTLY in this format:\n"
        "SUMMARY: <2-3 sentence summary>\n"
        "QUESTIONS:\n- <question>\n- <question>\n\n"
        f"Document:\n{context}"
    )
    result = generate_text(prompt)
    if is_unavailable_response(result):
        return {"summary": "", "questions": [], "error": result}

    summary = ""
    m = re.search(r"SUMMARY:\s*(.+?)(?:\n\s*QUESTIONS:|\Z)", result, re.S | re.I)
    if m:
        summary = m.group(1).strip()
    questions = []
    for line in result.splitlines():
        ls = line.strip()
        if ls.lower().startswith("summary:"):
            continue
        if ls.startswith(("-", "•", "*")) or re.match(r"^\d+[\.\)]", ls):
            q = re.sub(r"^[\-•\*\d\.\)\s]+", "", ls).strip()
            if q and (q.endswith("?") or len(q.split()) >= 4):
                questions.append(q)
    if not summary:
        summary = result.strip()[:400]
    return {"summary": summary, "questions": questions[:num_questions], "error": None}
