# Graph Report - onpaper  (2026-11-10)

## Corpus Check
- 76 files · ~88,680 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 15 file(s) not represented in the graph (top: .pkl 9, (none) 4, .toml 2)

## Summary
- 1063 nodes · 2101 edges · 55 communities (36 shown, 19 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 98 edges (avg confidence: 0.86)
- Token cost: about 193,000 tokens for the documentation pass (code parsing used none)

## Community Hubs (Navigation)
- Syllabus & Design Principles
- Enhanced Paper-Type Models
- Config & Paper-Type Detection
- AI Detection & RAG Concepts
- Security Limits & Audit
- AI Prompt Generator
- Streamlit UI Core
- CI Workflow
- AI Peer Reviewer Scoring
- Paper Health Check Engine
- CORE Paper Collector
- Model Training Dependencies
- Paper Processing Guide
- Grammar Checker
- Plagiarism Detector
- AI Gateway & Backends
- Paper Content Generator
- Ollama Client
- PDF & NLTK Utilities
- Enhanced ML Trainer
- Section Generation & Structure
- CORE API Trainer
- CORE Data Collection Scripts
- End-to-End UI Tests
- Cohesion & Embeddings
- CrossRef Citations
- Default Paper & Word Export
- AI Text Generation Calls
- Shared Draft & Uploads
- Chat with Paper (RAG)
- Groq Client & HTTP
- CORE Training Setup
- Claims & Section Splitting
- Retrieval & Ranking
- Paper Types & Templates
- App Entry Points
- ML Model Tests
- Reference Search & Verify
- Default Paper Generator
- Model Comparison
- Paper Processing Scripts
- Abstract & Word Targets
- Sample Draft Content
- Readability Analyzer
- Trainer Pipeline Run
- Product Features (README)
- Startup Tests
- Tone Auditor
- Rewrite Apply Flow
- Sentence Difficulty
- Revision Plan

## God Nodes (most connected - your core abstractions)
1. `generate_text()` - 44 edges
2. `EnhancedMLTrainer` - 24 edges
3. `run_health_check()` - 22 edges
4. `PlagiarismDetector` - 20 edges
5. `EnhancedPaperTypeDetector` - 18 edges
6. `PaperProcessor` - 17 edges
7. `CoreAPIPaperCollector` - 17 edges
8. `is_unavailable_response()` - 17 edges
9. `ResearchContentGenerator` - 15 edges
10. `OnPaper` - 15 edges

## Surprising Connections (you probably didn't know these)
- `Paper Health Check` --references--> `generate_text()`  [INFERRED]
  README.md → src/utils/gemini_helper.py
- `Machine Learning Applications in Healthcare: A Comprehensive Review` --semantically_similar_to--> `Sample draft: late-evening screen time and undergraduate sleep`  [INFERRED] [semantically similar]
  processed_papers/raw_papers/sample_healthcare_paper.txt → main/sample_draft.txt
- `Hands-on exercises` --references--> `generate_text()`  [EXTRACTED]
  onpaper_syllabus.md → src/utils/gemini_helper.py
- `Content Generation` --references--> `generate_text()`  [EXTRACTED]
  README.md → src/utils/gemini_helper.py
- `Paraphraser / Rewriter` --references--> `generate_text()`  [EXTRACTED]
  README.md → src/utils/gemini_helper.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **AI gateway auto-fallback chain (generate_text routes to Groq, then Ollama)** — src_utils_gemini_helper_generate_text, src_utils_gemini_helper, src_utils_groq_helper, src_utils_ollama_helper, readme_groq, readme_ollama, readme_ai_backends, onpaper_syllabus_single_ai_gateway [EXTRACTED 1.00]
- **Hybrid RAG retrieval pipeline (TF-IDF + MiniLM rankings fused with RRF, grounded prompt, lexical-only fallback)** — src_utils_paper_chat, src_utils_embedder, onpaper_syllabus_tf_idf, onpaper_syllabus_all_minilm_l6_v2, onpaper_syllabus_rrf_reciprocal_rank_fusion, onpaper_syllabus_grounded_prompting, onpaper_syllabus_lexical_only_fallback, src_utils_gemini_helper_generate_text [EXTRACTED 1.00]
- **Fallbacks that implement graceful degradation** — onpaper_syllabus_graceful_degradation, onpaper_syllabus_guarded_imports, onpaper_syllabus_lexical_only_fallback, onpaper_syllabus_nltk_punkt_fallback, readme_regex_fallback, onpaper_syllabus_always_return_useful_partial_results, requirements_optional_torch_transformers, readme_ai_backends [INFERRED 0.95]

## Communities (55 total, 19 thin omitted)

### Community 0 - "Syllabus & Design Principles"
Cohesion: 0.05
Nodes (50): OnPaper Complete Learning Syllabus & Architecture Guide, CORE API, Hands-on exercises, Logic / services layer (src/utils/), Model / data layer (src/ML/ + processed_papers/), Module 0 - Python foundations, Module 1 - Streamlit, Module 2 - LLM / Generative-AI integration (+42 more)

### Community 1 - "Enhanced Paper-Type Models"
Cohesion: 0.06
Nodes (4): EnhancedPaperTypeDetector, EnhancedPlagiarismDetector, EnhancedTextPreprocessor, test_enhanced_models()

### Community 2 - "Config & Paper-Type Detection"
Cohesion: 0.06
Nodes (14): topic_type_vectorizer.pkl, generate_comprehensive_paper(), check_grammar(), generate_content(), auto_detect_paper_type(), get_paper_type_detector(), get_type_guidance(), __getattr__() (+6 more)

### Community 3 - "AI Detection & RAG Concepts"
Cohesion: 0.08
Nodes (15): all-MiniLM-L6-v2, distilGPT-2, HC3-trained classifier, Module 5 - Deep learning / Transformers, Chat with your Paper, ZeroGPT - AI vs Human Detector (Beta), _clamp(), detect_ai_text() (+7 more)

### Community 4 - "Security Limits & Audit"
Cohesion: 0.06
Nodes (9): CI workflow (.github/workflows/ci.yml), pip-audit, safe_filename(), test_clip_text_cuts_long_input_and_reports_it(), test_groq_error_shown_to_users_has_no_org_id(), test_pdf_extraction_stops_at_page_and_size_caps(), test_real_pdf_text_is_extracted(), test_safe_filename_removes_path_and_header_characters() (+1 more)

### Community 5 - "AI Prompt Generator"
Cohesion: 0.11
Nodes (18): _build_clarity_rules(), _build_hygiene_rules(), _conditional_clauses(), generate_prompt(), _hard_terms(), _hygiene_issues(), _kw_hit(), _mode_instruction() (+10 more)

### Community 6 - "Streamlit UI Core"
Cohesion: 0.08
Nodes (16): ai_available(), cached_analyze_text(), _cached_backend_status(), cached_check_plagiarism(), cached_health_check(), cached_search_references(), cached_verify_references(), _find_source_for() (+8 more)

### Community 7 - "CI Workflow"
Cohesion: 0.07
Nodes (33): if, runs-on, steps, URL, jobs, audit, keep-awake, test (+25 more)

### Community 8 - "AI Peer Reviewer Scoring"
Cohesion: 0.10
Nodes (16): _baseline_line(), _best_array(), _best_score_object(), _build_image_prompt(), _build_text_prompt(), _clamp_score(), _clean_str(), generate_checklist() (+8 more)

### Community 9 - "Paper Health Check Engine"
Cohesion: 0.12
Nodes (15): analyze_consistency(), annotate_html(), _clamp(), grade(), _ok(), report_markdown(), run_health_check(), score_dimensions() (+7 more)

### Community 11 - "Model Training Dependencies"
Cohesion: 0.11
Nodes (5): download_hc3(), train(), fetch_papers_from_api(), main(), save_enhanced_plagiarism_model()

### Community 13 - "Grammar Checker"
Cohesion: 0.10
Nodes (6): Gemini, Grammar Check, check_grammar_text(), get_grammar_checker(), __getattr__(), GrammarChecker

### Community 14 - "Plagiarism Detector"
Cohesion: 0.10
Nodes (3): build_custom_detector(), PlagiarismDetector, _seed_sample_data()

### Community 15 - "AI Gateway & Backends"
Cohesion: 0.13
Nodes (10): generate_text_with_ollama(), get_available_backends(), get_backend_status(), load_env_file(), is_groq_available(), generate_text_with_ollama(), get_ollama_helper(), is_ollama_available() (+2 more)

### Community 18 - "PDF & NLTK Utilities"
Cohesion: 0.14
Nodes (6): ensure_nltk_data(), add_reference_document(), get_plagiarism_detector(), get_plagiarism_statistics(), __getattr__(), save_plagiarism_model()

### Community 20 - "Section Generation & Structure"
Cohesion: 0.13
Nodes (11): edit_section(), generate_quick_paper(), generate_section_guide(), generate_section_only(), get_available_paper_types(), get_content_generator(), _canon(), check_structure() (+3 more)

### Community 22 - "CORE Data Collection Scripts"
Cohesion: 0.20
Nodes (4): CoreAPICollector, main(), main(), run_paper_collection()

### Community 23 - "End-to-End UI Tests"
Cohesion: 0.16
Nodes (8): app(), fakes(), test_clear_draft(), test_every_page_renders(), test_find_a_source_opens_reference_finder_with_results(), test_rewrite_then_apply_updates_draft_and_score_history(), test_sample_draft_runs_health_check_and_follows_you(), test_send_to_paraphraser_from_readability()

### Community 24 - "Cohesion & Embeddings"
Cohesion: 0.16
Nodes (6): analyze_cohesion(), _split_paragraphs(), _starts_with_connector(), is_available(), _load(), summarize_text()

### Community 25 - "CrossRef Citations"
Cohesion: 0.18
Nodes (9): Module 7 - External APIs & data collection, CrossRef API, format_citation(), _authors_apa(), _authors_ieee(), _authors_mla(), format_reference(), _initials() (+1 more)

### Community 26 - "Default Paper & Word Export"
Cohesion: 0.16
Nodes (5): create_word_document(), generate_default_paper(), get_default_paper_generator(), __getattr__(), text_to_docx()

### Community 27 - "AI Text Generation Calls"
Cohesion: 0.16
Nodes (6): generate_text(), is_unavailable_response(), answer_question(), document_overview(), get_modes(), paraphrase_text()

### Community 28 - "Shared Draft & Uploads"
Cohesion: 0.16
Nodes (8): analysis_input(), _draft_box(), _keep_draft(), read_uploaded_file(), clipped(), show_counter(), clip_text(), extract_text_from_pdf()

### Community 29 - "Chat with Paper (RAG)"
Cohesion: 0.17
Nodes (4): chunk_text(), _format_history(), _label_chunk_sections(), prepare_document()

### Community 30 - "Groq Client & HTTP"
Cohesion: 0.18
Nodes (6): generate_text_with_groq(), _get_api_key(), get_groq_model(), _retry_wait(), redact(), test_redact_hides_org_ids_and_api_keys()

### Community 31 - "CORE Training Setup"
Cohesion: 0.20
Nodes (8): check_requirements(), create_config_file(), create_usage_guide(), main(), print_banner(), run_ml_training(), setup_core_api(), update_streamlit_app()

### Community 32 - "Claims & Section Splitting"
Cohesion: 0.21
Nodes (6): analyze_claims(), _count_markers(), _has_citation(), detected_section_names(), _normalize_heading(), split_sections()

### Community 33 - "Retrieval & Ranking"
Cohesion: 0.18
Nodes (6): _ask(), cosine_sim(), embed(), _rank(), retrieve_context(), dedupe_and_rank()

### Community 34 - "Paper Types & Templates"
Cohesion: 0.19
Nodes (6): Content Generation, Research Paper Types, __getattr__(), _high_word_bound(), section_guidance(), strip_section_heading()

### Community 36 - "ML Model Tests"
Cohesion: 0.23
Nodes (5): test_models(), main(), test_model_performance(), test_paper_type_detection_accuracy(), test_plagiarism_detection_accuracy()

### Community 38 - "Reference Search & Verify"
Cohesion: 0.21
Nodes (6): suggest_citations(), _parse_crossref(), search_references(), _content_tokens(), _split_entries(), verify_references()

### Community 40 - "Model Comparison"
Cohesion: 0.24
Nodes (4): calculate_metrics(), create_test_dataset(), main(), print_comparison()

### Community 42 - "Abstract & Word Targets"
Cohesion: 0.28
Nodes (4): section_word_badge(), score_abstract(), analyze_text(), compare_to_target()

### Community 43 - "Sample Draft Content"
Cohesion: 0.33
Nodes (8): Sample draft: late-evening screen time and undergraduate sleep, 10 pm device curfew recommendation, Key finding (each extra hour of late-evening screen time = 22 fewer minutes of sleep), Late-evening screen time (after 10 pm), Sleep duration (actigraph-measured), Study design (412 undergraduates, 14 days, logging app + wrist actigraph, mixed-effects model), Walker (2017), Perfect-paper section guideline

### Community 44 - "Readability Analyzer"
Cohesion: 0.22
Nodes (3): _find_passive_sentences(), grade_band_status(), _interpret_flesch()

### Community 46 - "Product Features (README)"
Cohesion: 0.48
Nodes (7): One draft, every tool, OnPaper, Paper Health Check, Paraphraser / Rewriter, Readability & Quality, Reference Finder, Sidebar stages (Check, Write, Cite, Read)

## Ambiguous Edges - Review These
- `citation_manager.py` → `CrossRef API`  [AMBIGUOUS]
  README.md · relation: references
- `AI Backends (auto mode: Groq then Ollama)` → `Gemini`  [AMBIGUOUS]
  README.md · relation: conceptually_related_to
- `Gemini` → `Single AI gateway`  [AMBIGUOUS]
  onpaper_syllabus.md · relation: references
- `OnPaper Complete Learning Syllabus & Architecture Guide` → `requirements.txt`  [AMBIGUOUS]
  onpaper_syllabus.md · relation: references

## Knowledge Gaps
- **1 isolated node(s):** `numpy (>=2.2)`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 446 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **19 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `generate_text()` connect `AI Text Generation Calls` to `Syllabus & Design Principles`, `Paper Types & Templates`, `Config & Paper-Type Detection`, `AI Prompt Generator`, `Default Paper Generator`, `AI Peer Reviewer Scoring`, `Paper Processing Scripts`, `Grammar Checker`, `Product Features (README)`, `AI Gateway & Backends`, `Paper Content Generator`, `PDF & NLTK Utilities`, `Default Paper & Word Export`, `Chat with Paper (RAG)`, `Groq Client & HTTP`?**
  _High betweenness centrality (0.104) - this node is a cross-community bridge._
- **Are the 9 inferred relationships involving `run_health_check()` (e.g. with `score_abstract()` and `analyze_claims()`) actually correct?**
  _`run_health_check()` has 9 INFERRED edges - model-reasoned connections that need verification._
- **What connects `numpy (>=2.2)` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Syllabus & Design Principles` be split into smaller, more focused modules?**
  _Cohesion score 0.05086071987480438 - nodes in this community are weakly interconnected._
- **Why does `CoreAPIPaperCollector` connect `CORE Paper Collector` to `Groq Client & HTTP`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `citation_manager.py` and `CrossRef API`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `AI Backends (auto mode: Groq then Ollama)` and `Gemini`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._