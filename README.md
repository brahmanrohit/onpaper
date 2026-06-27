# OnPaper — Research Paper Writing Assistant

OnPaper is a **Streamlit** web application that helps users **generate, analyze, and improve academic research papers**. It combines Large Language Models (Google **Gemini** or a local **Ollama** server) with classic scikit-learn ML models for plagiarism detection and paper-type classification.

This is the **single source of truth** for the project. It documents the real, current layout, what every file does, how data flows, how to run it, and the known issues.

---

## Table of Contents
1. [Features](#features)
2. [Quick Start](#quick-start)
3. [Project Structure (what every file does)](#project-structure)
4. [How It Works (data flow)](#how-it-works)
5. [Configuration](#configuration)
6. [AI Backends (Gemini & Ollama)](#ai-backends)
7. [Machine Learning Models](#machine-learning-models)
8. [Research Paper Types](#research-paper-types)
9. [Model Tuning Reference](#model-tuning-reference)
10. [Known Issues & Limitations](#known-issues--limitations)
11. [Developer Notes](#developer-notes)

---

## Features

| Feature | What it does | Powered by |
|---------|--------------|------------|
| **Content Generation** | Generate a full paper (Word `.docx`), a quick paper (TXT), a single section, or a section-wise guide. Supports 10 paper types. | Groq/Ollama + templates |
| **Paper Analysis** | Extract text from an uploaded PDF/TXT and produce a TF-IDF summary. | PyPDF2 + scikit-learn |
| **Citation Assistant** | Suggest citations and format them in APA / IEEE / MLA. | *(currently mock data)* |
| **Grammar Check** | Detect & correct grammar/spelling/style with change tracking and a side-by-side diff. | Gemini (with regex fallback) |
| **Plagiarism Detection** | Compare text against a reference corpus using TF-IDF + cosine similarity, at document and sentence level. | scikit-learn |
| **Reference Finder** | Find **real** academic papers with DOIs via the free CrossRef API; format in APA/IEEE/MLA. | CrossRef (no key) |
| **Paraphraser / Rewriter** | Rewrite text to reduce plagiarism, improve clarity, or change tone (7 modes). | Groq/Ollama |
| **Readability & Quality** | Offline writing analysis: Flesch/Flesch-Kincaid/Gunning Fog, passive voice, long sentences, complex words. | Native (no key) |
| **Chat with your Paper** | Upload a PDF and ask questions; answers grounded in the document via **hybrid RAG** (semantic MiniLM embeddings + TF-IDF, fused with RRF) with conversation memory. | transformers + scikit-learn + Groq/Ollama |
| **ZeroGPT — AI vs Human Detector** *(Beta)* | Estimate AI-likelihood by combining a perplexity model (distilGPT-2), an HC3-trained classifier, and style heuristics via noisy-OR. In the sidebar's "Upcoming Features" dropdown. | torch + transformers + scikit-learn |

---

## Quick Start

```bash
# 1. Create & activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) configure AI — see "AI Backends" below
#    Create a .env file with GROQ_API_KEY=... and/or run Ollama locally

# 4. Run the app
python run.py
#    or:  cd main && streamlit run main.py
```

The app opens at **http://localhost:8501**.

> **Note (Windows):** `gemini_helper.py` reconfigures stdout/stderr to UTF-8 on import so console status messages don't crash on cp1252 terminals. No action needed.

---

## Project Structure

Real, current layout (the older `app/` + `core/` proposal in past docs was **never implemented**).

```
onpaperfixed/
├── run.py                      # 🔑 Entry point — launches `streamlit run main/main.py`
├── requirements.txt            # Python dependencies
├── README.md                   # This file (the only documentation file)
├── .env                        # Secrets/config (GIT-IGNORED — see Configuration)
│
├── main/                       # 🎨 PRESENTATION LAYER
│   ├── main.py                 # The entire Streamlit UI: sidebar + 5 features
│   ├── deploy_plagiarism_model.py  # Standalone helper to deploy a plagiarism model (not used by the app)
│   ├── plagiarism_model.pkl    # Legacy model copy (not used at runtime)
│   ├── assets/                 # UI assets (currently empty)
│   └── .streamlit/config.toml  # Streamlit server/theme config
│
├── src/                        # 🔧 LOGIC, SERVICES & MODELS
│   ├── utils/                  # Core feature modules (imported by main.py)
│   │   ├── config.py                 # Centralized paths, thresholds, paper types
│   │   ├── gemini_helper.py          # ⭐ Central AI gateway: routes to Ollama/Gemini
│   │   ├── ollama_helper.py          # Local Ollama client (auto-discovers server & model)
│   │   ├── content_generator.py      # 10 paper-type templates; section-by-section generation
│   │   ├── default_paper_generator.py# Richer generator → builds the Word .docx (+ images)
│   │   ├── text_analyzer.py          # PlagiarismDetector (TF-IDF + cosine, batched)
│   │   ├── grammar_checker.py        # Gemini JSON grammar correction + regex fallback
│   │   ├── citation_manager.py       # Citation suggestions + APA/IEEE/MLA formatting (mock)
│   │   ├── pdf_processor.py          # PDF text extraction + TF-IDF summarization
│   │   ├── paper_type_detector.py    # Keyword-based paper-type detection + per-type guidance
│   │   ├── topic_type_predictor.py   # Loads the pickled topic→type classifier
│   │   ├── readability_analyzer.py   # Offline readability/quality metrics (Feature)
│   │   ├── reference_finder.py       # Real references via CrossRef API (Feature)
│   │   ├── paraphraser.py            # Rewrite/paraphrase via AI gateway (Feature)
│   │   ├── paper_chat.py             # Hybrid RAG: semantic+lexical retrieval + memory (Feature)
│   │   ├── embedder.py               # MiniLM sentence embeddings for semantic search
│   │   ├── ai_detector.py            # ZeroGPT: blends perplexity+ML+heuristic (Feature)
│   │   ├── perplexity_detector.py    # distilGPT-2 perplexity/burstiness scoring (Feature)
│   │   └── nlp_utils.py              # Thin legacy wrappers around generate_text()
│   │
│   ├── ML/                     # 🤖 Trained model artifacts + one training script
│   │   ├── train_topic_type_classifier.py   # Trains topic→type RandomForest
│   │   ├── training_report.json             # Metrics from the last training run
│   │   ├── enhanced_models_plagiarism.pkl   # ✅ Plagiarism refs+vectorizer (USED)
│   │   ├── enhanced_models_vectorizer.pkl   # ✅ Shared TF-IDF vectorizer (USED)
│   │   ├── enhanced_models_paper_type.pkl   # ✅ Paper-type classifier (USED)
│   │   ├── topic_type_classifier.pkl        # ✅ Topic→type classifier (USED)
│   │   ├── topic_type_vectorizer.pkl        # ✅ Topic→type vectorizer (USED)
│   │   └── (enhanced_*/plagiarism_model* )   # Training outputs / legacy copies (not used at runtime)
│   │
│   └── model_devlopment_file/  # 🛠️ OFFLINE training & data-collection scripts (not run by the app)
│       ├── core_api_collector.py / core_api_paper_collector.py / run_core_collection.py
│       ├── core_api_config.json / setup_core_api_training.py
│       ├── enhanced_ml_models.py / enhanced_ml_trainer.py / train_with_core_api.py
│       ├── enhance_plagiarism_from_api.py / process_your_papers.py
│       └── model_comparison.py / test_ml_models.py
│
├── processed_papers/           # 📚 DATA
│   ├── paper_processing_guide.py
│   ├── raw_papers/sample_healthcare_paper.txt
│   └── training_data/
│       ├── training_dataset.json
│       ├── plagiarism_test_cases.json
│       └── core_training/core_dataset.json   # 93 papers used to train the models
│
├── scripts/run_app.py          # 🚀 Alternative launcher (same as run.py)
├── models/                     # Legacy model dir (used only by deploy_plagiarism_model.py)
└── tests/                      # ✅ test_imports.py, test_app.py (smoke tests)
```

### The runtime path (what actually executes when you use the app)
```
run.py → main/main.py → src/utils/*  → (gemini_helper → ollama_helper / Gemini API)
                                      → (text_analyzer, content_generator, … load pkls from src/ML/)
```
Everything under `src/model_devlopment_file/`, `scripts/`, `tests/`, `main/deploy_plagiarism_model.py`, and the legacy `*.pkl` copies are **not** part of that runtime path.

---

## How It Works

### Content generation
`main.py` → `generate_default_paper()` / `generate_quick_paper()` / `generate_section_only()`
→ looks up the paper-type **template** (structure + word limits) in `content_generator.py` /
`default_paper_generator.py` → builds a prompt per section → calls **`generate_text()`**
(`gemini_helper.py`) → assembles sections + citations → renders in Streamlit and offers
a `.docx`/TXT download.

### Plagiarism detection
`main.py` → `check_plagiarism(text, threshold)` (`text_analyzer.py`):
1. Preprocess (lowercase, strip punctuation/whitespace).
2. **Document-level:** one batched TF-IDF transform of the input + all reference docs → `cosine_similarity` matrix → `plagiarism_score = max similarity × 100`.
3. **Sentence-level:** reference sentences are tokenized **once and cached**; all input and reference sentences are vectorized in a single batched transform and compared via one cosine-similarity matrix; pairs above the threshold are returned.

> This batched design keeps a full-paper check well under a second. (An earlier version did a per-sentence-pair transform loop that took 10s+.)

### Grammar check
`check_grammar_text()` → builds a JSON-output prompt → Gemini → parse JSON → structured
`{corrected_text, changes[], statistics}`. If the API is unavailable or returns no JSON,
a **regex fallback** applies common contraction/spelling/punctuation fixes.

### Topic → paper type
`TopicTypePredictor` loads `topic_type_classifier.pkl` + `topic_type_vectorizer.pkl` and
returns `(paper_type, confidence)`. `paper_type_detector.py` provides a complementary
**keyword-based** detector + per-type writing guidance.

---

## Configuration

All settings live in [src/utils/config.py](src/utils/config.py): model paths, default
plagiarism threshold (`0.7`), summary length, max file size, the 10 paper types, citation
styles, and allowed upload types.

### `.env` (git-ignored)
```bash
# AI backend: auto (Groq→Ollama), groq, or ollama
AI_BACKEND=auto

# Groq (primary) — free key at https://console.groq.com/keys
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Ollama (fallback; local & unlimited; auto-detected if omitted)
OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3.2
```

> **Security:** `.env` is git-ignored and must never be committed. If a key was ever
> committed, **rotate it** — removing the file does not purge it from git history.

---

## AI Backends

The app uses **open-source models only** (no Gemini / no proprietary cloud). `generate_text(prompt, backend="auto")`
in [gemini_helper.py](src/utils/gemini_helper.py) is the single entry point used by every
generation feature. *(The module keeps its legacy `gemini_helper` filename for import
compatibility, but Gemini has been removed.)*

- **`auto`** *(default)*: try **Groq** first (fast cloud), then fall back to a local **Ollama** server.
- **`groq`**: cloud, OpenAI-compatible, **fast + generous free tier**, runs open-source Llama models.
  Get a free key at https://console.groq.com/keys (starts with `gsk_`); set `GROQ_API_KEY` and
  optionally `GROQ_MODEL` (default `llama-3.3-70b-versatile`). Implemented in
  [groq_helper.py](src/utils/groq_helper.py) using `requests` — no extra dependency.
- **`ollama`**: fully **local, open-source, unlimited & offline**. Install from https://ollama.com,
  run `ollama pull llama3.2`; `ollama_helper.py` auto-discovers the server URL and picks a
  preferred model (`llama3.2`, `llama3.1`, `mistral`, …) or the first installed model.

If a backend fails in `auto` mode, the **real error** is surfaced (e.g. a Groq `401`/`429`)
instead of a generic "not available" message.

---

## Machine Learning Models

| Model file (`src/ML/`) | Type | Used by | Purpose |
|------------------------|------|---------|---------|
| `enhanced_models_plagiarism.pkl` | dict: vectorizer + 93 reference docs | `text_analyzer.py` | Plagiarism reference corpus |
| `enhanced_models_vectorizer.pkl` | TfidfVectorizer | config | Shared TF-IDF vectorizer |
| `enhanced_models_paper_type.pkl` | CalibratedClassifierCV | `paper_type_detector.py` | Paper-type classification |
| `topic_type_classifier.pkl` | RandomForestClassifier | `topic_type_predictor.py` | Topic → paper type |
| `topic_type_vectorizer.pkl` | TfidfVectorizer | `topic_type_predictor.py` | Topic features |

**Training data:** 93 papers (`processed_papers/training_data/core_training/core_dataset.json`),
spanning computer_science (58), medicine (25), social_sciences (7), business (3).
Last run (`training_report.json`): paper-type accuracy ≈ 58%.

Retrain the topic→type model:
```bash
python src/ML/train_topic_type_classifier.py
```
The broader model pipeline lives in `src/model_devlopment_file/` (CORE API collection +
enhanced trainers) and is **offline only**.

---

## Research Paper Types

10 supported types, each with a customized section structure and word limits
(see `content_generator.py`):

`empirical`, `theoretical`, `review`, `comparative`, `case_study`, `analytical`,
`methodological`, `position`, `technical`, `interdisciplinary`.

### Perfect-paper section guideline
| Section | Standard | Simple |
|---------|----------|--------|
| Abstract | 150–250 | 150–200 |
| Introduction | 500–800 | 400–600 |
| Literature Review | 800–1200 | — |
| Methodology | 400–600 | 300–500 |
| Results | 400–600 | 300–500 |
| Discussion | 600–800 | 400–600 |
| Conclusion | 300–500 | 200–400 |
| **Total** | ~3,500–5,000 | ~2,000–3,000 |

---

## Model Tuning Reference

**Plagiarism (TF-IDF):** `max_features=5000`, `ngram_range=(1,2)`, `min_df`/`max_df` tuned per
corpus. Score bands: `<30` low, `30–60` moderate, `60–threshold` high, `>threshold` plagiarism risk.

**Paper-type / topic classifier (RandomForest):** `n_estimators=100`, `random_state=42`,
TF-IDF features (3000–8000), `stop_words='english'`.

**Maintenance:** retrain when the reference corpus grows; expand the dataset to improve the
(currently modest) paper-type accuracy; consider semantic embeddings for plagiarism instead
of pure TF-IDF.

---

## Known Issues & Limitations

- **Citations are mock data** — `citation_manager.suggest_citations()` returns random sample
  papers, not real lookups. Wire it to a real source (CrossRef / Semantic Scholar) for production.
- **Plagiarism corpus is generic** — the score reflects similarity to the 93 bundled academic
  papers, not the web. There is no UI to add your own reference documents yet.
- **No database / no auth / single-user** — state is files + pickles; models load at startup.
- **`paper_type_detector` uses keyword matching at runtime** — the trained classifier pkl is
  loaded but `detect_paper_type()` currently returns keyword-based results.
- **AI features require credentials** — without a valid `GROQ_API_KEY` or a running Ollama,
  generation/grammar return an actionable error; plagiarism, summary, and type detection work offline.

---

## Developer Notes

- **Entry points:** `run.py` (recommended) or `scripts/run_app.py`. Both `cd` into `main/`
  and run `streamlit run main.py`.
- **Smoke tests:** `python tests/test_imports.py` and `python tests/test_app.py`.
- **Adding a paper type:** add it to `PAPER_TYPES` in `config.py` and to the template dicts in
  `content_generator.py` / `default_paper_generator.py`.
- **`__pycache__/` and `.env`** are git-ignored; keep them out of commits.
