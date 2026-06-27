# OnPaper — Complete Learning Syllabus & Architecture Guide

> **Purpose of this document:** Everything you need to learn — and understand — to fully master this project: the tech stack, the architecture, the data flow, the workflow of every feature, and a step-by-step learning path. Read it top to bottom once, then use it as a reference.

**Project in one line:** A **Streamlit** web app that generates, analyzes, and improves academic research papers using **LLMs** (Groq / Gemini / Ollama) plus classic **scikit-learn** ML for plagiarism detection, classification, and a **transformers**-powered hybrid RAG chat.

- **Language:** Python 3.13
- **Domain:** NLP · Applied ML · LLM application engineering
- **App type:** Single-user, file/pickle-based, no database, no auth

---

## Table of Contents
1. [How to use this syllabus](#1-how-to-use-this-syllabus)
2. [The mental model (read this first)](#2-the-mental-model-read-this-first)
3. [Architecture — the 3 layers](#3-architecture--the-3-layers)
4. [The runtime path (what executes when you click)](#4-the-runtime-path-what-executes-when-you-click)
5. [Feature workflows (data flow, step by step)](#5-feature-workflows-data-flow-step-by-step)
6. [The learning syllabus (9 modules)](#6-the-learning-syllabus-9-modules)
7. [File-by-file map](#7-file-by-file-map)
8. [Key design patterns to internalize](#8-key-design-patterns-to-internalize)
9. [Concept glossary](#9-concept-glossary)
10. [Suggested study plan (week by week)](#10-suggested-study-plan-week-by-week)
11. [Hands-on exercises](#11-hands-on-exercises)

---

## 1. How to use this syllabus

There are two ways to "know" this project:

1. **Understand the architecture** — how the pieces fit, what calls what, why the design is the way it is. → Sections 2–5 & 8.
2. **Know the underlying technologies** — Streamlit, scikit-learn, transformers, LLM APIs, etc. → Section 6 (the modules).

Do **both**, interleaved. The recommended order is in [Section 10](#10-suggested-study-plan-week-by-week).

A useful rule while studying: **start at `main/main.py` and follow a single feature all the way down** to the model/API and back. Then repeat for the next feature. You'll see the same patterns reused everywhere.

---

## 2. The mental model (read this first)

OnPaper is built around **one core philosophy:**

> **Every external dependency degrades gracefully and never crashes the app.**

Concretely:
- No AI key? → Generation/grammar return a helpful error string; **plagiarism, summary, readability, and type detection still work fully offline.**
- `transformers`/`torch` not installed? → The RAG chat **falls back from semantic+lexical to lexical-only (TF-IDF)** retrieval.
- Gemini SDK missing? → Gemini is silently disabled; Groq/Ollama remain.
- NLTK `punkt` model missing? → Falls back to naive `text.split(".")`.

Once you see this pattern, the whole codebase becomes predictable. Almost every module has a `try/except` that swaps a powerful path for a simpler one.

The second big idea: **one gateway for all AI.** Every feature that needs an LLM calls a single function — `generate_text(prompt)` in [src/utils/gemini_helper.py](src/utils/gemini_helper.py) — which internally routes to Groq → Ollama (→ Gemini). Features never talk to an LLM API directly.

---

## 3. Architecture — the 3 layers

```
┌─────────────────────────────────────────────────────────────┐
│  PRESENTATION LAYER          main/                           │
│  main.py  → the entire Streamlit UI (sidebar + all features) │
│  .streamlit/config.toml → server & theme                     │
└───────────────────────────┬─────────────────────────────────┘
                            │  imports & calls
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  LOGIC / SERVICES LAYER      src/utils/                      │
│  - gemini_helper.py   ⭐ AI gateway (Groq/Ollama/Gemini)     │
│  - groq_helper.py / ollama_helper.py  (backend clients)      │
│  - content_generator.py / default_paper_generator.py        │
│  - text_analyzer.py   (plagiarism: TF-IDF + cosine)         │
│  - grammar_checker.py / paraphraser.py / citation_manager.py │
│  - pdf_processor.py / readability_analyzer.py                │
│  - reference_finder.py (CrossRef API)                        │
│  - paper_chat.py + embedder.py  (hybrid RAG)                 │
│  - ai_detector.py + perplexity_detector.py (ZeroGPT)        │
│  - paper_type_detector.py / topic_type_predictor.py         │
│  - config.py   (paths, thresholds, paper types)             │
└───────────────────────────┬─────────────────────────────────┘
                            │  load .pkl artifacts
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  MODEL / DATA LAYER         src/ML/  +  processed_papers/    │
│  - enhanced_models_*.pkl  (plagiarism corpus, vectorizer,    │
│    paper-type classifier)                                    │
│  - topic_type_classifier.pkl / topic_type_vectorizer.pkl     │
│  - training data JSON (93 papers)                            │
└─────────────────────────────────────────────────────────────┘

OFFLINE-ONLY (not in the runtime path):
  src/model_devlopment_file/  → CORE API collection + model training
  scripts/  tests/  models/   → launchers, smoke tests, legacy
```

**Why this layering matters:** the UI knows nothing about TF-IDF or HTTP. The services know nothing about Streamlit. Models are just artifacts loaded by services. This separation is what lets each piece fail independently.

---

## 4. The runtime path (what executes when you click)

```
run.py
  └─ os.chdir("main") ; subprocess: streamlit run main.py
       └─ main/main.py                      (Streamlit re-runs top-to-bottom on every interaction)
            └─ from src.utils import ...     (feature modules)
                 ├─ generate_text()  → gemini_helper → groq_helper / ollama_helper / google-genai
                 ├─ check_plagiarism() → text_analyzer → loads enhanced_models_plagiarism.pkl
                 ├─ summarize()        → pdf_processor → PyPDF2 + scikit-learn
                 ├─ answer_question()  → paper_chat → embedder (MiniLM) + TF-IDF + generate_text()
                 └─ ...
```

Entry points (both do the same thing): [run.py](run.py) and [scripts/run_app.py](scripts/run_app.py). App opens at **http://localhost:8501**.

**Important Streamlit fact:** the script re-runs from the top on *every* widget interaction. That's why heavy objects (models) are loaded as **module-level globals** (e.g. `plagiarism_detector = PlagiarismDetector()` at the bottom of [text_analyzer.py](src/utils/text_analyzer.py)) — so the `.pkl` loads once at import, not on every click.

---

## 5. Feature workflows (data flow, step by step)

Study these one at a time. Each is a complete vertical slice through all 3 layers.

### 5.1 Content Generation (the flagship feature)
```
UI form (paper type, topic, length)
  → generate_default_paper() / generate_quick_paper() / generate_section_only()
  → look up the paper-type TEMPLATE (section list + word limits) in
      content_generator.py / default_paper_generator.py
  → build a prompt PER SECTION
  → generate_text(prompt)            [gemini_helper → Groq/Ollama/Gemini]
  → assemble sections + citations
  → render in Streamlit, offer .docx (python-docx) / TXT download
```
**Learn:** prompt templating, paper-type taxonomy (10 types in `config.PAPER_TYPES`), `.docx` building.

### 5.2 Plagiarism Detection (the best-engineered feature)
```
check_plagiarism(text, threshold)            [text_analyzer.py]
  1. preprocess: lowercase, strip punctuation/whitespace
  2. DOCUMENT level:
       transform(input + all 93 ref docs) in ONE batched TF-IDF call
       → cosine_similarity matrix → score = max similarity × 100
  3. SENTENCE level:
       reference sentences tokenized ONCE and cached (_ref_sentences_cache)
       all input+ref sentences vectorized in one batch
       → one cosine matrix → keep pairs above threshold
  4. return {score, is_plagiarized, similar_sentences[:5], message}
```
**Learn the optimization story:** an earlier version did an O(input × reference) loop of individual `.transform()` calls and took 10s+. The current code does **one batched transform + one matrix multiply** and runs in <1s. This is the single most important performance lesson in the repo. Read [text_analyzer.py:140-213](src/utils/text_analyzer.py#L140-L213).

### 5.3 Paper Analysis / Summary
```
upload PDF/TXT → pdf_processor.py
  → PyPDF2 extracts text
  → TF-IDF scores sentences → pick top-N → extractive summary
```
Offline, no API key needed.

### 5.4 Grammar Check
```
check_grammar_text()              [grammar_checker.py]
  → build a JSON-output prompt
  → generate_text()  [Gemini/Groq]
  → parse JSON → {corrected_text, changes[], statistics}
  → if API unavailable OR no JSON returned → REGEX FALLBACK
      (contractions, common spelling, punctuation)
  → UI shows side-by-side diff with tracked changes
```
**Learn:** asking an LLM for structured JSON, and always having a non-AI fallback.

### 5.5 Chat with your Paper — **Hybrid RAG** (most advanced)
```
prepare_document(text)            [paper_chat.py]
  → chunk_text(): sentence-aware chunks (~130 words, 1-sentence overlap)
  → LEXICAL index: TfidfVectorizer (always available)
  → SEMANTIC index: embedder.embed(chunks)  [MiniLM via transformers] (optional)

answer_question(question)
  → retrieve_context():
       lexical sims  = cosine(TF-IDF)
       semantic sims = cosine(MiniLM embeddings)
       FUSE both rankings with Reciprocal Rank Fusion (RRF, k=60)
       (if embeddings unavailable → lexical-only fallback)
  → build a grounded prompt ("answer ONLY from these passages")
  → add recent conversation history (last 3 turns)
  → generate_text()  → answer
  → ALWAYS return the retrieved passages (stays useful even if AI fails)
```
**Learn:** chunking, dense vs sparse retrieval, RRF, grounded prompting, conversation memory. Read [paper_chat.py](src/utils/paper_chat.py) and [embedder.py](src/utils/embedder.py) together.

### 5.6 Reference Finder
```
query → reference_finder.py → CrossRef REST API (no key)
  → real papers + DOIs → format in APA / IEEE / MLA
```
Contrast with `citation_manager.py`, which currently returns **mock** sample papers (a known limitation).

### 5.7 Paraphraser / Rewriter
```
text + mode (7 modes) → paraphraser.py → prompt → generate_text() → rewritten text
```

### 5.8 Readability & Quality
```
text → readability_analyzer.py
  → Flesch, Flesch-Kincaid, Gunning Fog, passive voice,
    long sentences, complex words   (pure Python, no API)
```

### 5.9 ZeroGPT — AI-vs-Human Detector (Beta)
```
text → ai_detector.py
  → perplexity_detector.py: distilGPT-2 perplexity + burstiness  [torch+transformers]
  → an HC3-trained classifier  [scikit-learn]
  → style heuristics
  → fuse all signals with NOISY-OR → AI-likelihood %
```

### 5.10 Topic → Paper Type
```
topic text → topic_type_predictor.py
  → load topic_type_classifier.pkl + topic_type_vectorizer.pkl
  → (paper_type, confidence)

paper_type_detector.py → complementary KEYWORD-based detector + per-type writing guidance
```

---

## 6. The learning syllabus (9 modules)

Each module: **what to learn** + **where it lives in this repo**.

### Module 0 — Python foundations (prerequisite)
- Modules/packages, `__init__.py`, `venv`, `pip`, pinned `requirements.txt`
- Type hints (`typing`, `Optional`, `Dict`, `List`)
- Exception handling & **graceful-degradation** patterns (the core idiom here)
- `.env` / environment variables via `python-dotenv`
- Git & `.gitignore` secrets hygiene (`.env` is git-ignored; rotate leaked keys)
- 📍 Everywhere; especially [config.py](src/utils/config.py), [gemini_helper.py](src/utils/gemini_helper.py)

### Module 1 — Streamlit (frontend / web layer)
- The re-run-on-interaction execution model
- Widgets, `st.sidebar`, tabs, file uploader, download button, session state
- Loading heavy objects once via module-level globals (not per re-run)
- `.streamlit/config.toml` (server + theme)
- Supporting libs Streamlit pulls in: `altair`, `pydeck`, `tornado`, `watchdog`, `blinker`, `click`
- 📍 [main/main.py](main/main.py), [main/.streamlit/config.toml](main/.streamlit/config.toml), [run.py](run.py)
- `streamlit==1.42.2`

### Module 2 — LLM / Generative-AI integration ⭐ (the heart)
- **Multi-backend gateway with auto-fallback** — the most important module
- Backends:
  - **Groq** (default): `requests` → OpenAI-compatible REST; `llama-3.3-70b-versatile`; rate limits 401/429
  - **Gemini**: unified `google-genai` SDK; `gemini-flash-latest`; guarded import
  - **Ollama** (local): auto-discovers server + model; `http://localhost:11434`
- Prompt engineering: section templates, JSON-output prompts, grounded prompts
- `tenacity` retries; meaningful error surfacing (return the *real* error, not "unavailable")
- 📍 [gemini_helper.py](src/utils/gemini_helper.py), `groq_helper.py`, [ollama_helper.py](src/utils/ollama_helper.py), [nlp_utils.py](src/utils/nlp_utils.py)
- `google-genai>=2.9.0`, `requests`, `httpx`, `tenacity`

### Module 3 — Classic ML (scikit-learn)
- **TF-IDF**: `TfidfVectorizer` (`max_features`, `ngram_range`, `min_df`/`max_df`, `stop_words`)
- **Cosine similarity**: `sklearn.metrics.pairwise.cosine_similarity` (basis of plagiarism + retrieval)
- Classifiers: `RandomForestClassifier`, `CalibratedClassifierCV` (probability calibration)
- `Pipeline`, model persistence with `joblib` / `pickle` (`.pkl`)
- **Batched/vectorized computation** for speed (the 10s→<1s lesson)
- Train/eval & metrics (`training_report.json`, ~58% paper-type accuracy)
- 📍 [text_analyzer.py](src/utils/text_analyzer.py), [topic_type_predictor.py](src/utils/topic_type_predictor.py), [src/ML/train_topic_type_classifier.py](src/ML/train_topic_type_classifier.py)
- `scikit-learn==1.6.1`, `numpy`, `scipy`, `pandas`, `joblib`

### Module 4 — NLP & text processing
- Tokenization: `nltk.sent_tokenize`, the `punkt` model (lazy download + fallback)
- Preprocessing: lowercase, strip punctuation, normalize whitespace
- Offline readability: Flesch / Flesch-Kincaid / Gunning Fog, passive voice, complex words
- Keyword-based classification heuristics
- 📍 [readability_analyzer.py](src/utils/readability_analyzer.py), [paper_type_detector.py](src/utils/paper_type_detector.py), [nlp_utils.py](src/utils/nlp_utils.py)
- `nltk==3.9.1`, `regex`

### Module 5 — Deep learning / Transformers (RAG & AI detection) — most advanced
- **Sentence embeddings**: `all-MiniLM-L6-v2` via `AutoTokenizer`/`AutoModel`
- Mean-pooling over tokens + L2 normalization (so dot product = cosine)
- Lazy loading, CPU inference, batching, `torch.no_grad()`
- **Hybrid RAG**: chunking, dense+sparse retrieval, **Reciprocal Rank Fusion**, grounded prompting, memory
- **Perplexity-based AI detection**: distilGPT-2, burstiness, **noisy-OR** fusion
- 📍 [embedder.py](src/utils/embedder.py), [paper_chat.py](src/utils/paper_chat.py), [ai_detector.py](src/utils/ai_detector.py), [perplexity_detector.py](src/utils/perplexity_detector.py)
- `torch>=2.6`, `transformers>=4.44,<6`

### Module 6 — Document processing & generation
- **PDF extraction**: `PyPDF2`
- **Word `.docx` generation**: `python-docx` (full-paper builder)
- **PDF generation**: `pdfkit`
- **Templating**: `Jinja2` (`MarkupSafe`)
- Images/charts: `pillow`, `matplotlib`
- 📍 [pdf_processor.py](src/utils/pdf_processor.py), [default_paper_generator.py](src/utils/default_paper_generator.py), [content_generator.py](src/utils/content_generator.py)
- `PyPDF2==3.0.1`, `python-docx`, `pdfkit`, `Jinja2`, `pillow`, `matplotlib`

### Module 7 — External APIs & data collection
- **CrossRef API** — real papers + DOIs, no key (Reference Finder)
- **CORE API** — bulk paper collection for training data (offline pipeline)
- HTTP fundamentals: `requests`/`httpx`, status codes, JSON
- Citation formatting: APA / IEEE / MLA
- 📍 [reference_finder.py](src/utils/reference_finder.py), [citation_manager.py](src/utils/citation_manager.py), [src/model_devlopment_file/](src/model_devlopment_file/)
- `requests==2.32.3`, `httpx`, `GitPython`

### Module 8 — Data, config & project engineering
- **Centralized config**: paths, thresholds, the 10 paper types ([config.py](src/utils/config.py))
- JSON datasets + `jsonschema` validation (`processed_papers/training_data/`)
- `pandas` / `pyarrow` for tabular data
- 3-layer separation: presentation / logic / models
- Smoke tests ([tests/test_imports.py](tests/test_imports.py), [tests/test_app.py](tests/test_app.py))
- CLI polish: `rich`, `tqdm`, `colorama`
- 📍 [config.py](src/utils/config.py), `tests/`, `processed_papers/`

---

## 7. File-by-file map

### Entry points
| File | Role |
|------|------|
| [run.py](run.py) | 🔑 Recommended launcher → `streamlit run main/main.py` |
| [scripts/run_app.py](scripts/run_app.py) | Alternative launcher (same behavior) |

### Presentation — `main/`
| File | Role |
|------|------|
| [main/main.py](main/main.py) | The **entire** Streamlit UI: sidebar + every feature |
| [main/.streamlit/config.toml](main/.streamlit/config.toml) | Server & theme config |
| `main/deploy_plagiarism_model.py` | Standalone helper, **not used at runtime** |

### Logic / services — `src/utils/`
| File | Role | Module |
|------|------|--------|
| [config.py](src/utils/config.py) | Paths, thresholds, paper types, citation styles | 8 |
| [gemini_helper.py](src/utils/gemini_helper.py) | ⭐ AI gateway (Groq→Ollama→Gemini) | 2 |
| `groq_helper.py` | Groq REST client | 2 |
| [ollama_helper.py](src/utils/ollama_helper.py) | Local Ollama client + auto-discovery | 2 |
| [content_generator.py](src/utils/content_generator.py) | 10 paper templates, section generation | 6 |
| [default_paper_generator.py](src/utils/default_paper_generator.py) | Rich generator → builds `.docx` | 6 |
| [text_analyzer.py](src/utils/text_analyzer.py) | `PlagiarismDetector` (TF-IDF + cosine, batched) | 3 |
| [grammar_checker.py](src/utils/grammar_checker.py) | LLM JSON grammar fix + regex fallback | 2/4 |
| [citation_manager.py](src/utils/citation_manager.py) | Citation formatting (mock data) | 7 |
| [pdf_processor.py](src/utils/pdf_processor.py) | PDF extraction + TF-IDF summary | 6/3 |
| [paper_type_detector.py](src/utils/paper_type_detector.py) | Keyword type detection + guidance | 4 |
| [topic_type_predictor.py](src/utils/topic_type_predictor.py) | Loads pickled topic→type classifier | 3 |
| [readability_analyzer.py](src/utils/readability_analyzer.py) | Offline readability metrics | 4 |
| [reference_finder.py](src/utils/reference_finder.py) | Real references via CrossRef | 7 |
| [paraphraser.py](src/utils/paraphraser.py) | Rewrite/paraphrase (7 modes) | 2 |
| [paper_chat.py](src/utils/paper_chat.py) | Hybrid RAG chat | 5 |
| [embedder.py](src/utils/embedder.py) | MiniLM sentence embeddings | 5 |
| [ai_detector.py](src/utils/ai_detector.py) | ZeroGPT signal fusion | 5 |
| [perplexity_detector.py](src/utils/perplexity_detector.py) | distilGPT-2 perplexity/burstiness | 5 |
| [nlp_utils.py](src/utils/nlp_utils.py) | Thin legacy wrappers over `generate_text()` | 2 |

### Models / data — `src/ML/` + `processed_papers/`
| Artifact | Type | Used by |
|----------|------|---------|
| `enhanced_models_plagiarism.pkl` | vectorizer + 93 ref docs | `text_analyzer.py` |
| `enhanced_models_vectorizer.pkl` | shared `TfidfVectorizer` | config |
| `enhanced_models_paper_type.pkl` | `CalibratedClassifierCV` | `paper_type_detector.py` |
| `topic_type_classifier.pkl` | `RandomForestClassifier` | `topic_type_predictor.py` |
| `topic_type_vectorizer.pkl` | `TfidfVectorizer` | `topic_type_predictor.py` |
| `processed_papers/training_data/core_training/core_dataset.json` | 93 training papers | offline training |
| [src/ML/train_topic_type_classifier.py](src/ML/train_topic_type_classifier.py) | training script | manual retrain |

### Offline only (not in runtime path)
`src/model_devlopment_file/` (CORE API collectors + enhanced trainers), `tests/`, `models/` (legacy), and all duplicate/legacy `.pkl` copies.

---

## 8. Key design patterns to internalize

These appear repeatedly. Recognizing them is "understanding the project."

1. **Single AI gateway** — all generation goes through `generate_text()`. Never call an LLM API directly from a feature. → [gemini_helper.py](src/utils/gemini_helper.py)
2. **Graceful degradation everywhere** — every optional dependency has a fallback (semantic→lexical, AI→regex, punkt→split, Gemini→disabled).
3. **Guarded imports** — heavy/optional libs (`google-genai`, `torch`, `transformers`) are imported inside `try/except` so a missing package disables one feature instead of crashing the app.
4. **Lazy + once loading** — models load lazily on first use ([embedder.py](src/utils/embedder.py) `_load()`) or once at import as a module global ([text_analyzer.py](src/utils/text_analyzer.py) `plagiarism_detector = ...`). Streamlit's re-run model makes this essential.
5. **Batch, don't loop** — vectorize everything at once and do a single matrix op. The plagiarism 10s→<1s rewrite is the canonical example.
6. **Caching invalidated on change** — `_ref_sentences_cache` is cleared whenever the reference set changes.
7. **Surface the real error** — `auto` mode reports the actual backend failure (`API_KEY_INVALID`, Groq `429`) instead of a generic message. → `is_unavailable_response()` and `last_error` tracking.
8. **Always return useful partial results** — RAG returns retrieved passages even when the LLM answer fails, so the feature stays useful offline.

---

## 9. Concept glossary

Terms you must be comfortable with:

- **TF-IDF** — Term Frequency–Inverse Document Frequency; turns text into a sparse numeric vector weighting rare-but-meaningful words.
- **Cosine similarity** — angle-based similarity between two vectors (1 = identical direction). Powers plagiarism + retrieval.
- **N-gram** — contiguous run of N tokens; `ngram_range=(1,2)` = unigrams + bigrams.
- **Embedding (dense vector)** — a fixed-length numeric representation of meaning from a neural model (MiniLM → 384 dims here).
- **Mean pooling** — averaging token embeddings (masked by attention) into one sentence vector.
- **L2 normalization** — scaling a vector to length 1 so dot product equals cosine.
- **RAG (Retrieval-Augmented Generation)** — retrieve relevant passages, then have the LLM answer using *only* them.
- **Lexical vs semantic search** — keyword match (TF-IDF) vs meaning match (embeddings). Hybrid uses both.
- **RRF (Reciprocal Rank Fusion)** — combine two ranked lists by summing `1/(k+rank)`; scale-free, robust.
- **Perplexity** — how "surprised" a language model is by text; lower can indicate AI-generated text.
- **Burstiness** — variation in sentence complexity; humans tend to be burstier than AI.
- **Noisy-OR** — probabilistic way to fuse several independent "yes" signals into one probability.
- **CalibratedClassifierCV** — wraps a classifier so its predicted probabilities are trustworthy.
- **Extractive summarization** — pick the most important existing sentences (vs generating new text).
- **Grounded prompting** — instruct the LLM to answer strictly from supplied context and admit when it can't.

---

## 10. Suggested study plan (week by week)

A realistic 4–5 week path. Adjust to your pace.

**Week 1 — Get it running + see the shape**
- Module 0 (Python foundations) + Module 1 (Streamlit).
- Run the app (`python run.py`). Click every feature.
- Read [run.py](run.py) → skim [main/main.py](main/main.py). Map each UI button to a function.

**Week 2 — The AI gateway (the heart)**
- Module 2. Read [gemini_helper.py](src/utils/gemini_helper.py) line by line.
- Trace `generate_text("hello")` through Groq → Ollama fallback.
- Follow Content Generation (5.1) and Grammar Check (5.4) end to end.

**Week 3 — Classic ML**
- Module 3 + Module 4.
- Read [text_analyzer.py](src/utils/text_analyzer.py) fully; understand the batched rewrite (5.2).
- Read [pdf_processor.py](src/utils/pdf_processor.py) (summary) and the type predictors.

**Week 4 — Documents, APIs, engineering**
- Modules 6, 7, 8.
- Read `default_paper_generator.py` (`.docx`), [reference_finder.py](src/utils/reference_finder.py) (CrossRef), [config.py](src/utils/config.py).
- Run the smoke tests in `tests/`.

**Week 5 — The advanced layer (save for last)**
- Module 5.
- Read [embedder.py](src/utils/embedder.py) then [paper_chat.py](src/utils/paper_chat.py); understand chunking + RRF (5.5).
- Read [ai_detector.py](src/utils/ai_detector.py) + [perplexity_detector.py](src/utils/perplexity_detector.py) (5.9).

**Order summary:** `0 → 1 → 2 → 3 → 4 → 6 → 7 → 8 → 5`

---

## 11. Hands-on exercises

Do these to *prove* you understand it (not just read it):

1. **Trace a request:** add `print()` statements through `generate_text()` and watch which backend answers. Turn off your key and watch the fallback.
2. **Add a paper type:** add an entry to `PAPER_TYPES` in [config.py](src/utils/config.py) and a matching template in [content_generator.py](src/utils/content_generator.py). Generate a paper of that type.
3. **Break and fix gracefully:** temporarily rename the plagiarism `.pkl`; confirm the app still starts and shows the "no model" message instead of crashing.
4. **Tune plagiarism:** change `max_features` / `ngram_range` in [text_analyzer.py](src/utils/text_analyzer.py) and observe score changes on the same input.
5. **Force lexical-only RAG:** make `embedder.is_available()` return `False`; confirm chat falls back to TF-IDF and still answers.
6. **Retrain a model:** run `python src/ML/train_topic_type_classifier.py`, then check the new `training_report.json` metrics.
7. **Write a test:** extend `tests/test_imports.py` to assert a new module imports cleanly.

---

### The single most important file
[src/utils/gemini_helper.py](src/utils/gemini_helper.py) — the multi-backend AI gateway. Understand its `auto` fallback chain (Groq → Ollama → Gemini) and you understand the project's core philosophy: **every dependency degrades gracefully and never crashes the app.**

---

*Generated as a study companion to the project README, which remains the single source of truth for current file layout and known limitations.*
