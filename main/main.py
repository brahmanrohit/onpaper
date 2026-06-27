import streamlit as st
import os
import sys
import hashlib
import html as _html

# Add the project root to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.pdf_processor import extract_text_from_pdf, summarize_text
from src.utils.text_analyzer import check_plagiarism, build_custom_detector
from src.utils.citation_manager import suggest_citations, format_citation
from src.utils.content_generator import (
    generate_section_only, generate_quick_paper, get_available_paper_types,
    edit_section, generate_section_guide,
)
from src.utils.default_paper_generator import generate_default_paper, create_word_document, text_to_docx

DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
from src.utils.topic_type_predictor import TopicTypePredictor
from src.utils.grammar_checker import check_grammar_text
from src.utils.readability_analyzer import analyze_text, compare_to_target, analyze_sentences, AUDIENCE_PRESETS, grade_band_status
from src.utils.reference_finder import search_references, format_reference as format_real_reference, dedupe_and_rank
from src.utils.reference_verifier import verify_references
from src.utils.paraphraser import paraphrase_text, get_modes
from src.utils.paper_chat import prepare_document, answer_question, document_overview, retrieve_context
from src.utils.ai_detector import detect_ai_text
from src.utils.consistency_checker import analyze_consistency
from src.utils.redundancy_finder import find_repetition
from src.utils.keyword_extractor import extract_keywords
from src.utils.prompt_generator import generate_prompt, refine_prompt, PROMPT_FRAMEWORKS, TARGET_MODELS
from src.utils.score import score_paper, generate_checklist, DEFAULT_RUBRIC, build_revision_plan, revise_paper
from src.utils.gemini_helper import get_backend_status
from src.utils.cohesion_analyzer import analyze_cohesion
from src.utils.claim_checker import analyze_claims
from src.utils.structure_checker import check_structure, get_paper_types
from src.utils.abstract_scorecard import score_abstract
from src.utils.tone_auditor import analyze_tone
import matplotlib.pyplot as plt
import io

# Ensure NLTK data is downloaded
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page config (MUST be the first Streamlit command).
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)

# Secrets note: on Streamlit Community Cloud, top-level entries in the app's
# Secrets are exported to os.environ automatically, so the os.getenv-based AI
# gateway picks up GROQ_API_KEY / AI_BACKEND with no extra code. Locally those
# values come from .env (python-dotenv). No st.secrets access here on purpose,
# so a missing local secrets.toml never raises a "No secrets found" error.

# Load the topic->type classifier once and reuse it across reruns.
@st.cache_resource
def get_topic_predictor():
    return TopicTypePredictor()


def suggest_paper_type(topic: str, key: str):
    """Render a button that predicts the best paper type for a topic."""
    if topic and st.button("🔮 Suggest paper type", key=key):
        try:
            pred, confidence = get_topic_predictor().predict(topic)
            conf_txt = f" ({confidence * 100:.0f}% confidence)" if confidence else ""
            st.info(f"Suggested paper type: **{pred}**{conf_txt}")
        except Exception as e:
            st.warning(f"Could not suggest a paper type: {e}")


# ---- Tier 1 UX helpers -----------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def _cached_backend_status():
    """Backend availability, cached briefly so the sidebar doesn't re-probe each rerun."""
    return get_backend_status()


def render_backend_status():
    """Show which AI backend is live so failures are self-explanatory."""
    try:
        status = _cached_backend_status()
    except Exception:
        return
    groq = status.get("groq", {})
    ollama = status.get("ollama", {})
    st.sidebar.markdown("---")
    if groq.get("available") or ollama.get("available"):
        if groq.get("available"):
            st.sidebar.success(f"🟢 AI: Groq — {groq.get('model', 'ready')}")
        if ollama.get("available"):
            models = ollama.get("models") or []
            st.sidebar.success(f"🟢 AI: Ollama — {(models[0] if models else 'local')}")
    else:
        st.sidebar.error("🔴 No AI backend connected")
        st.sidebar.caption(
            "Generation, grammar & paraphrase are disabled. Set `GROQ_API_KEY` in .env "
            "(console.groq.com/keys) or run Ollama. Plagiarism, readability & references "
            "still work without an AI backend."
        )


def show_counter(text: str):
    """Live word/character/read-time caption for an input box."""
    n_words = len((text or "").split())
    n_chars = len(text or "")
    if n_words:
        st.caption(f"📊 {n_words} words · {n_chars} characters · ~{max(1, round(n_words / 200))} min read")
    else:
        st.caption("📊 0 words · 0 characters")


def section_word_badge(word_count: int, word_limit: str):
    """Show a per-section word count vs its target range with an under/ok/over badge."""
    try:
        low, high = [int(x) for x in str(word_limit).split("-")]
    except (ValueError, AttributeError):
        return
    res = compare_to_target(word_count, low, high)
    icon = {"ok": "🟢", "under": "🟡", "over": "🟠"}.get(res["status"], "⚪")
    st.caption(f"{icon} {word_count} words · target {low}–{high} ({res['message']})")


# Shared "remember the topic across the four Content-Generation tabs" state.
_TOPIC_KEYS = ("def_topic", "quick_topic", "sec_topic", "guide_topic")


def _sync_topic(src_key: str):
    """Copy the topic typed in one tab into all the other tabs' inputs."""
    value = st.session_state.get(src_key, "")
    for k in _TOPIC_KEYS:
        st.session_state[k] = value


def read_uploaded_file(uploaded_file):
    """Read an uploaded PDF/TXT robustly. Returns (text, error).

    Fixes two whole classes of bugs: (1) extract_text_from_pdf returns an
    'Error…' STRING on failure (scanned/corrupt/empty PDF) that callers used to
    treat as real content; (2) non-UTF-8 .txt files used to crash the page with
    UnicodeDecodeError. On any failure we return (None, friendly_message).
    """
    if uploaded_file is None:
        return None, "No file provided."
    try:
        # Streamlit reuses the SAME file object across reruns; without seeking to
        # the start, a second read returns empty (the pointer is left at EOF).
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
        name = (getattr(uploaded_file, "name", "") or "").lower()
        if getattr(uploaded_file, "type", "") == "application/pdf" or name.endswith(".pdf"):
            text = extract_text_from_pdf(uploaded_file)
            if not text or str(text).strip().lower().startswith("error"):
                return None, ("Could not extract text from this PDF. If it is a scanned/image-only "
                              "PDF it has no selectable text — try a text-based PDF or paste the text.")
            return text, None
        # TXT (or anything else): decode robustly across common encodings.
        raw = uploaded_file.read()
        if isinstance(raw, str):
            return (raw, None) if raw.strip() else (None, "The file appears to be empty.")
        for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                text = raw.decode(enc)
                return (text, None) if text.strip() else (None, "The file appears to be empty.")
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace"), None
    except Exception as e:
        return None, f"Could not read the file: {e}"


def analysis_input(key_prefix: str, label: str = "Or paste text here:"):
    """Shared 'upload PDF/TXT or paste text' input for the analysis tools.
    Returns the text (str) or None. Shows a friendly warning on upload errors."""
    col1, col2 = st.columns(2)
    with col1:
        up = st.file_uploader("Upload a PDF or TXT file:", type=["pdf", "txt"], key=f"{key_prefix}_file")
    with col2:
        txt = st.text_area(label, height=160, key=f"{key_prefix}_text")
    show_counter(txt)
    if up is not None:
        text, err = read_uploaded_file(up)
        if err:
            st.warning(err)
            return None
        return text
    return txt if (txt and txt.strip()) else None


def render_heatstrip(sentences):
    """Render a per-sentence readability heat-strip (green=easy … red=hard)."""
    bg = {"easy": "#d7f5dd", "medium": "#fff3cd", "hard": "#f8d7da"}
    spans = [
        f'<span style="background:{bg.get(s["level"], "#eee")};padding:1px 3px;border-radius:3px" '
        f'title="{s["words"]} words · {s["syllables_per_word"]} syll/word">{_html.escape(s["sentence"])}</span>'
        for s in sentences
    ]
    st.markdown('<div style="line-height:2.2">' + " ".join(spans) + "</div>", unsafe_allow_html=True)
    st.caption("🟩 easy · 🟨 medium · 🟥 hard — by sentence length & word complexity (hover for details)")


def _send_to_paraphraser(sentence: str):
    """Seed the Paraphraser with a sentence and switch to that feature."""
    st.session_state["para_text"] = sentence
    st.session_state["para_mode"] = "Improve clarity"
    st.session_state["_goto_para"] = True
    st.rerun()


# --- Cached wrappers for the slow / network-bound calls (Tier 4) -------------
@st.cache_data(ttl=3600, show_spinner=False)
def cached_suggest_citations(query, rows):
    return suggest_citations(query, rows=rows)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_search_references(query, rows):
    return search_references(query, rows=rows)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_verify_references(text):
    return verify_references(text)


@st.cache_data(show_spinner=False)
def cached_analyze_text(text, long_threshold):
    return analyze_text(text, long_sentence_words=long_threshold)


@st.cache_data(show_spinner=False)
def cached_check_plagiarism(text, threshold):
    return check_plagiarism(text, threshold)


# Main title
st.title("📚 Research Paper Assistant")

# A "Send to Paraphraser" handoff from another feature: switch the menu BEFORE
# the selectbox is instantiated (you can't set a widget's value after it renders).
if st.session_state.pop("_goto_para", False):
    st.session_state["feature_menu"] = "Paraphraser"
    st.session_state["upcoming_feature"] = "— None —"

# Sidebar navigation
menu = ["Content Generation", "Paper Analysis", "Citation Assistant", "Reference Finder", "Verify References", "Grammar Check", "Paraphraser", "Plagiarism Detection", "Readability & Quality", "Chat with your Paper"]
choice = st.sidebar.selectbox("Select a Feature", menu, key="feature_menu")

# Upcoming / experimental features (beta). Selecting one overrides the main menu.
st.sidebar.markdown("---")
st.sidebar.caption("🧪 Upcoming Features (Beta)")
upcoming_menu = ["— None —", "ZeroGPT — Writing Analysis Suite", "AI Prompt Generator", "AI Peer Reviewer"]
upcoming = st.sidebar.selectbox("Try an experimental feature:", upcoming_menu, key="upcoming_feature")
if upcoming != "— None —":
    choice = upcoming

# The ZeroGPT suite bundles the AI detector + the offline analysis tools behind a
# tab-style selector. It remaps `choice` to the chosen tool so the existing
# feature blocks below render it (no duplication of their logic).
if choice == "ZeroGPT — Writing Analysis Suite":
    st.markdown("### 🧪 ZeroGPT — Writing Analysis Suite")
    st.caption("AI detector plus offline writing-analysis tools — pick one:")
    _SUITE_TOOLS = {
        "🤖 AI Detector": "ZeroGPT — AI vs Human Detector",
        "🔀 Flow & Cohesion": "Flow & Cohesion Map",
        "📑 Citation Density": "Citation Density & Uncited Claims",
        "✅ Section Completeness": "Section Completeness Check",
        "📋 Abstract Scorecard": "Abstract Scorecard",
        "🎚️ Tone Auditor": "Tone Auditor (Hedging/Overclaiming)",
        "🔤 Terminology": "Terminology & Acronym Check",
        "♻️ Repetition": "Repetition / Self-Overlap Finder",
        "🏷️ Keywords": "Keyword & Contribution Extractor",
    }
    _picked = st.radio("Tool", list(_SUITE_TOOLS.keys()), horizontal=True,
                       label_visibility="collapsed", key="suite_tool")
    choice = _SUITE_TOOLS[_picked]

# Sidebar: live AI backend status (Tier 1).
render_backend_status()

# Feature logic (placeholders)
if choice == "Content Generation":
    st.header("📝 Content Generation")
    # Remember the topic across all four tabs (Tier 1).
    for _k in _TOPIC_KEYS:
        st.session_state.setdefault(_k, "")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Default Research Paper Generator",
        "Quick Paper Generator",
        "Section Generator/Editor",
        "Paper Guide"
    ])

    with tab1:
        st.subheader("Default Research Paper Generator (Best & Realistic)")
        topic = st.text_input("Enter your research topic:", key="def_topic",
                              on_change=_sync_topic, args=("def_topic",))
        suggest_paper_type(topic, "def_suggest")
        paper_types = [t['key'] for t in get_available_paper_types()]
        paper_type = st.selectbox("Select paper type:", paper_types, key="def_type")
        suggestions = st.text_area("(Optional) Suggestions for style, focus, or content:", key="def_sugg")
        # Convert suggestions to dict if not empty
        suggestions_dict = {"all": suggestions} if suggestions else {}

        # Detect if user wants a chart/image
        wants_image = False
        image_keywords = ["chart", "bar", "graph", "image", "figure", "plot"]
        if suggestions:
            for kw in image_keywords:
                if kw in suggestions.lower():
                    wants_image = True
                    break

        # Generate a placeholder chart if needed
        images = []
        chart_bytes = None
        if wants_image and topic:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(["A", "B", "C"], [3, 7, 5], color="#4F8DFD")
            ax.set_title(f"Sample Chart for: {topic}")
            ax.set_ylabel("Value")
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png")
            buf.seek(0)
            chart_bytes = buf.read()
            # Section is assigned after generation (below) to a section that
            # actually exists for the chosen paper type.
            images.append({
                "section": None,
                "image_bytes": chart_bytes,
                "caption": f"Sample chart related to '{topic}'"
            })
            plt.close(fig)

        if st.button("Generate Default Paper", key="def_gen"):
            if topic:
                with st.spinner("Generating your paper — this can take a minute…"):
                    paper = generate_default_paper(topic, paper_type, suggestions_dict, images)
                # Attach the chart to a results-like section that EXISTS for this
                # paper type (most types have no "Results" section); else the last one.
                chart_section = None
                if wants_image and chart_bytes and paper.get("sections"):
                    sec_keys = list(paper["sections"].keys())
                    priority = ["Results", "Case Analysis", "Analysis", "Comparative Analysis",
                                "Technical Analysis", "Cross-Disciplinary Analysis", "Validation", "Discussion"]
                    chart_section = next((s for s in priority if s in sec_keys), sec_keys[-1] if sec_keys else None)
                    for img in paper.get("images", []):
                        img["section"] = chart_section
                st.success("Default research paper generated!")
                for section, data in paper['sections'].items():
                    st.markdown(f"### {section}")
                    st.write(data['content'])
                    section_word_badge(data['word_count'], data['word_limit'])
                    # Show image preview under the section it was attached to.
                    if chart_section and section == chart_section and chart_bytes:
                        st.image(chart_bytes, caption=f"Sample chart for '{topic}'", use_container_width=True)
                st.markdown("---")
                st.markdown("#### References")
                for i, citation in enumerate(paper['citations'], 1):
                    st.write(f"{i}. {citation['citation']}")
                # Download button
                docx_bytes = create_word_document(paper)
                st.download_button(
                    label="Download as Word Document",
                    data=docx_bytes,
                    file_name=f"{paper['title'].replace(' ', '_')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            else:
                st.warning("Please enter a topic.")

    with tab2:
        st.subheader("Quick Paper Generator")
        st.caption("Generates a full paper. The slider sets the approximate length of each section.")
        topic = st.text_input("Enter your research topic:", key="quick_topic",
                              on_change=_sync_topic, args=("quick_topic",))
        suggest_paper_type(topic, "quick_suggest")
        paper_types = [t['key'] for t in get_available_paper_types()]
        paper_type = st.selectbox("Select paper type:", paper_types, key="quick_type")
        word_count = st.slider("Approx. words per section:", min_value=150, max_value=1200, value=400, step=50, key="quick_wordcount")
        if word_count >= 800:
            st.caption("⏳ Longer sections take more time and may be throttled on the free Groq tier.")
        if st.button("Generate In-Depth Quick Paper", key="quick_gen"):
            if topic:
                with st.spinner("Generating your paper — this can take a minute…"):
                    paper = generate_quick_paper(topic, paper_type, target_words=word_count)
                st.success("In-depth quick research paper generated!")
                txt_content = f"{paper['sections'].get('Title', '')}\n\n"
                for section, data in paper['sections'].items():
                    if section != 'Title':
                        content = data['content']
                        st.markdown(f"### {section}")
                        st.write(content)
                        section_word_badge(data['word_count'], data['word_limit'])
                        txt_content += f"\n## {section}\n{content}"
                st.markdown("---")
                st.markdown("#### References")
                for i, citation in enumerate(paper['citations'], 1):
                    st.write(f"{i}. {citation['citation']}")
                txt_content += "\n\nReferences\n" + '\n'.join([f"{i+1}. {c['citation']}" for i, c in enumerate(paper['citations'])])
                st.download_button(
                    label="Download Paper as TXT",
                    data=txt_content,
                    file_name="in_depth_quick_paper.txt",
                    mime="text/plain"
                )
            else:
                st.warning("Please enter a topic.")

    with tab3:
        st.subheader("Section Generator/Editor")
        topic = st.text_input("Enter your research topic:", key="sec_topic",
                              on_change=_sync_topic, args=("sec_topic",))
        paper_types = [t['key'] for t in get_available_paper_types()]
        paper_type = st.selectbox("Select paper type:", paper_types, key="sec_type")
        section_list = [
            "Title", "Abstract", "Introduction", "Literature Review", "Methodology", "Results", "Discussion", "Conclusion"
        ]
        section = st.selectbox("Select section:", section_list, key="sec_section")
        sec_mode = st.radio("Mode:", ["Generate new", "Edit / improve existing"],
                            horizontal=True, key="sec_mode")

        if sec_mode == "Generate new":
            sec_instructions = st.text_area(
                "(Optional) focus or instructions for this section:", key="sec_instructions")
            if st.button("Generate Section", key="sec_gen"):
                if topic and section:
                    with st.spinner(f"Generating the {section} section…"):
                        content = generate_section_only(topic, section, paper_type,
                                                        instructions=sec_instructions)
                    st.success(f"Section '{section}' generated!")
                    st.markdown(f"### {section}")
                    st.write(content)
                    show_counter(content)
                    sdc1, sdc2 = st.columns(2)
                    sdc1.download_button("Download as TXT", data=content,
                                         file_name=f"{section.lower().replace(' ', '_')}.txt",
                                         mime="text/plain", key="sec_dl")
                    sdc2.download_button("Download as Word (.docx)",
                                         data=text_to_docx(section, content),
                                         file_name=f"{section.lower().replace(' ', '_')}.docx",
                                         mime=DOCX_MIME, key="sec_docx")
                else:
                    st.warning("Please enter a topic and select a section.")
        else:  # Edit / improve existing
            existing = st.text_area("Paste the section text to improve:", height=220, key="sec_existing")
            show_counter(existing)
            edit_instr = st.text_area(
                "(Optional) how should it be improved? e.g. 'more formal', 'expand', 'fix grammar', 'tighten'",
                key="sec_edit_instr")
            if st.button("Improve Section", key="sec_edit_btn"):
                if existing and existing.strip():
                    with st.spinner("Improving the section…"):
                        improved = edit_section(existing, section, paper_type, edit_instr)
                    st.success("Section improved!")
                    st.markdown(f"### Improved {section}")
                    st.write(improved)
                    show_counter(improved)
                    with st.expander("Compare original vs improved"):
                        oc1, oc2 = st.columns(2)
                        with oc1:
                            st.markdown("**Original**")
                            st.text_area("Original", value=existing, height=250, key="sec_orig_cmp")
                        with oc2:
                            st.markdown("**Improved**")
                            st.text_area("Improved", value=improved, height=250, key="sec_new_cmp")
                    edc1, edc2 = st.columns(2)
                    edc1.download_button("Download as TXT", data=improved,
                                         file_name=f"{section.lower().replace(' ', '_')}_improved.txt",
                                         mime="text/plain", key="sec_edit_dl")
                    edc2.download_button("Download as Word (.docx)",
                                         data=text_to_docx(f"Improved {section}", improved),
                                         file_name=f"{section.lower().replace(' ', '_')}_improved.docx",
                                         mime=DOCX_MIME, key="sec_edit_docx")
                else:
                    st.warning("Please paste some text to improve.")

    with tab4:
        st.subheader("Paper Guide (Section-wise, Detailed)")
        st.caption("How-to guidance for each section — purpose, what to include, tips, and common "
                   "mistakes. (This is advice on writing the paper, not the paper content itself.)")
        topic = st.text_input("Enter your research topic:", key="guide_topic",
                              on_change=_sync_topic, args=("guide_topic",))
        paper_types = [t['key'] for t in get_available_paper_types()]
        paper_type = st.selectbox("Select paper type:", paper_types, key="guide_type")
        section_list = [
            "Title", "Abstract", "Introduction", "Literature Review", "Methodology", "Results", "Discussion", "Conclusion"
        ]
        selected_sections = st.multiselect("Select sections to include:", section_list, default=section_list)
        word_count = st.slider("Word count per section:", min_value=100, max_value=1000, value=400, step=50)
        suggestions = st.text_area("(Optional) Suggestions or focus for the guide:", key="guide_sugg")
        if st.button("Generate Paper Guide", key="guide_gen"):
            if topic and selected_sections:
                guide = ""
                total = len(selected_sections)
                progress = st.progress(0.0)
                status = st.empty()
                for i, section in enumerate(selected_sections):
                    status.info(f"Writing guide for **{section}** ({i + 1}/{total})…")
                    content = generate_section_guide(topic, section, paper_type,
                                                     target_words=word_count, instructions=suggestions)
                    st.markdown(f"### {section} Guide")
                    st.write(content)
                    guide += f"\n\n## {section}\n{content}"
                    progress.progress((i + 1) / total)
                status.success(f"Generated {total} section(s).")
                st.download_button(
                    label="Download Full Guide as TXT",
                    data=guide,
                    file_name=f"{topic.replace(' ', '_')}_paper_guide.txt",
                    mime="text/plain"
                )
            else:
                st.warning("Please enter a topic and select at least one section.")
elif choice == "Paper Analysis":
    st.header("🔍 Paper Analysis")
    tab1, tab2 = st.tabs(["Analyze Uploaded File", "Analyze Pasted Text"])
    with tab1:
        uploaded_file = st.file_uploader("Upload a PDF or TXT file for analysis:", type=["pdf", "txt"], key="analysis_file")
        if uploaded_file is not None:
            text, err = read_uploaded_file(uploaded_file)
            if err:
                st.warning(err)
            else:
                st.text_area("Extracted Text:", value=text, height=200)
                if st.button("Analyze & Suggest Content", key="analyze_file_btn"):
                    summary = summarize_text(text)
                    st.subheader("Suggested Content:")
                    st.write(summary)
                    # Download suggested content
                    st.download_button(
                        label="Download Suggested Content as TXT",
                        data=summary,
                        file_name="suggested_content.txt",
                        mime="text/plain",
                        key="analysis_file_dl"
                    )
        else:
            st.info("Please upload a file to analyze.")
    with tab2:
        input_text = st.text_area("Paste text for analysis:", height=200, key="analysis_text")
        if st.button("Analyze & Suggest Content", key="analyze_text_btn"):
            if input_text:
                summary = summarize_text(input_text)
                st.subheader("Suggested Content:")
                st.write(summary)
                st.download_button(
                    label="Download Suggested Content as TXT",
                    data=summary,
                    file_name="suggested_content.txt",
                    mime="text/plain",
                    key="analysis_text_dl"
                )
            else:
                st.warning("Please paste some text.")
elif choice == "Citation Assistant":
    st.header("📚 Citation Assistant")
    st.caption("Finds real papers (with DOIs) via the free CrossRef API and formats them in APA/IEEE/MLA.")
    query = st.text_input("Enter your research query or topic for citations:")
    style = st.selectbox("Select citation style:", ["APA", "IEEE", "MLA"])
    num_citations = st.slider("Number of citations:", min_value=1, max_value=8, value=3)
    if st.button("Generate Citations"):
        if query:
            with st.spinner("Searching CrossRef for real references..."):
                citations = cached_suggest_citations(query, num_citations)
            if citations:
                st.subheader("Suggested Citations:")
                for i, paper in enumerate(citations[:num_citations], 1):
                    citation_text = format_citation(paper, style=style)
                    st.write(f"{i}. {citation_text}")
                # Download citations
                citations_txt = '\n'.join([format_citation(p, style=style) for p in citations[:num_citations]])
                st.download_button(
                    label="Download Citations as TXT",
                    data=citations_txt,
                    file_name="citations.txt",
                    mime="text/plain"
                )
            else:
                st.warning("No references found for that query. Try different or broader keywords.")
        else:
            st.warning("Please enter a query or topic.")
elif choice == "Grammar Check":
    st.header("✍️ Grammar Check")
    
    # Add info about AI backend
    if not (os.getenv('GROQ_API_KEY') or os.getenv('OLLAMA_BASE_URL')):
        st.info("💡 **Tip**: Add a free Groq API key (GROQ_API_KEY) to the .env file for AI grammar checking, or run a local Ollama server. Get a free Groq key at https://console.groq.com/keys")
    
    text = st.text_area("Enter text to check for grammar:", height=200, key="grammar_text")
    show_counter(text)

    col1, col2 = st.columns([1, 1])
    with col1:
        check_button = st.button("Check Grammar", key="grammar_check_btn")
    with col2:
        show_changes = st.button("Show Changes", key="show_changes_btn")
    
    if check_button:
        if text:
            with st.spinner("Checking grammar..."):
                result = check_grammar_text(text)
            
            st.success(result['message'])
            
            # Display statistics
            stats = result['statistics']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Errors", stats['total_errors'])
            with col2:
                st.metric("Grammar Errors", stats['grammar_errors'])
            with col3:
                st.metric("Spelling Errors", stats['spelling_errors'])
            with col4:
                st.metric("Style Errors", stats['style_errors'])
            
            # Display corrected text
            st.subheader("Corrected Text:")
            st.text_area("Corrected Text:", value=result['corrected_text'], height=200, key="corrected_grammar_text")
            
            # Download buttons
            gdc1, gdc2 = st.columns(2)
            gdc1.download_button(
                label="Download as TXT",
                data=result['corrected_text'],
                file_name="corrected_text.txt",
                mime="text/plain"
            )
            gdc2.download_button(
                label="Download as Word (.docx)",
                data=text_to_docx("Corrected Text", result['corrected_text']),
                file_name="corrected_text.docx",
                mime=DOCX_MIME,
                key="grammar_docx"
            )
            
            # Store changes in session state for the "Show Changes" button
            st.session_state.grammar_changes = result['changes']
            st.session_state.grammar_original = text
            st.session_state.grammar_corrected = result['corrected_text']
            
        else:
            st.warning("Please enter text to check.")
    
    # Show changes button functionality
    if show_changes and hasattr(st.session_state, 'grammar_changes') and st.session_state.grammar_changes:
        st.subheader("📝 Detailed Changes Made:")
        
        changes = st.session_state.grammar_changes
        original_text = st.session_state.grammar_original
        corrected_text = st.session_state.grammar_corrected
        
        if changes:
            for i, change in enumerate(changes, 1):
                with st.expander(f"Change {i}: {change['type']} - {change['message']}", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original:**")
                        st.code(change['original'], language=None)
                    with col2:
                        st.markdown("**Corrected:**")
                        st.code(change['corrected'], language=None)
                    
                    st.markdown("**Context:**")
                    st.text(change['context'])
                    
                    # Color coding based on severity
                    severity_color = {
                        'error': '🔴',
                        'warning': '🟡', 
                        'info': '🔵'
                    }.get(change['severity'], '⚪')
                    
                    st.markdown(f"**Severity:** {severity_color} {change['severity'].title()}")
            
            # Show side-by-side comparison
            st.subheader("📊 Side-by-Side Comparison:")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Text:**")
                st.text_area("Original", value=original_text, height=300, key="original_side")
            with col2:
                st.markdown("**Corrected Text:**")
                st.text_area("Corrected", value=corrected_text, height=300, key="corrected_side")
        else:
            st.info("No changes were made to the text.")
    elif show_changes:
        st.warning("Please run grammar check first to see changes.")
elif choice == "Plagiarism Detection":
    st.header("🔍 Plagiarism Detection")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload a file for plagiarism check:", type=["pdf", "txt"], key="plag_file")
    with col2:
        input_text = st.text_area("Or paste text here:", height=150, key="plag_text")
        show_counter(input_text)
    threshold = st.slider("Plagiarism threshold (%):", min_value=10, max_value=100, value=70, step=1)

    # Tier 3: compare against YOUR OWN reference documents, not just the bundled corpus.
    with st.expander("➕ Add your own reference documents (compare against your sources or prior drafts)"):
        ref_files = st.file_uploader("Upload reference PDFs/TXTs to compare against:",
                                     type=["pdf", "txt"], accept_multiple_files=True, key="plag_refs")
        only_mine = st.checkbox("Compare ONLY against my documents (ignore the bundled 93-paper corpus)",
                                value=False, key="plag_only_mine")

    text = None
    if uploaded_file is not None:
        text, _upload_err = read_uploaded_file(uploaded_file)
        if _upload_err:
            st.warning(_upload_err)
    elif input_text:
        text = input_text

    if st.button("Check Plagiarism"):
        if not text:
            st.warning("Please upload a file or enter text to check.")
        else:
            # Gather any user-supplied reference documents.
            custom_docs = []
            for f in (ref_files or []):
                t, e = read_uploaded_file(f)
                if e:
                    st.warning(f"{f.name}: {e}")
                elif t:
                    custom_docs.append((f.name, t))

            result, corpus_note = None, "OnPaper's bundled corpus of ~93 academic papers — **not the web**"
            if custom_docs:
                sig = (tuple(sorted((n, hashlib.md5(t.encode("utf-8")).hexdigest())
                                     for n, t in custom_docs)), only_mine)
                if st.session_state.get("plag_det_sig") != sig:
                    with st.spinner("Indexing your reference documents…"):
                        st.session_state["plag_det"] = build_custom_detector(custom_docs, include_bundled=not only_mine)
                        st.session_state["plag_det_sig"] = sig
                result = st.session_state["plag_det"].check_plagiarism(text, threshold=threshold / 100)
                corpus_note = (f"your {len(custom_docs)} uploaded document(s)"
                               + ("" if only_mine else " plus the 93 bundled papers"))
            elif only_mine:
                st.warning("You chose 'only my documents' but uploaded none. Add reference documents, or uncheck that option.")
            else:
                result = cached_check_plagiarism(text, threshold / 100)

            if result:
                st.subheader("Plagiarism Report:")
                st.write(f"Plagiarism Score: {result['plagiarism_score']}%")
                st.write(result['message'])
                if result['similar_sentences']:
                    st.write("Similar Sentences:")
                    for sent in result['similar_sentences']:
                        st.write(f"- {sent['input_sentence']} (Similarity: {sent['similarity']:.2f})")

                # Why this score? (Tier 1 explainer)
                with st.expander("ℹ️ Why this score? — how to read it"):
                    score = result['plagiarism_score']
                    if score > threshold:
                        band = f"above your {threshold}% threshold — flagged as a risk"
                    elif score < 30:
                        band = "low (under 30%) — the text appears original"
                    elif score < 60:
                        band = "moderate (30–60%) — some overlap; review for paraphrasing"
                    else:
                        band = "high overlap, but below your threshold"
                    st.markdown(
                        f"- **Score: {score}%** → {band}.\n"
                        f"- **What it measures:** similarity to {corpus_note}. A high score means your text "
                        f"resembles those reference documents; it is a signal, not proof of plagiarism.\n"
                        f"- **Threshold {threshold}%:** the headline score is the single closest document-level "
                        f"match. The sentences listed above come from a separate sentence-by-sentence pass — "
                        f"each is shown when its own similarity exceeds the threshold."
                    )
                    if result['similar_sentences']:
                        top = result['similar_sentences'][0]
                        st.markdown(
                            f"- **Closest match:** {top['similarity'] * 100:.0f}% similar to a sentence in "
                            f"reference `{top.get('reference_doc', '?')}`."
                        )

                report = f"Plagiarism Score: {result['plagiarism_score']}%\n{result['message']}\n"
                if result['similar_sentences']:
                    report += "Similar Sentences:\n" + '\n'.join([f"- {s['input_sentence']} (Similarity: {s['similarity']:.2f})" for s in result['similar_sentences']])
                st.download_button(
                    label="Download Report as TXT",
                    data=report,
                    file_name="plagiarism_report.txt",
                    mime="text/plain"
                )
elif choice == "Paraphraser":
    st.header("🔄 Paraphraser / Rewriter")
    st.caption("Rewrite text to reduce plagiarism, improve clarity, or change tone. Requires an AI backend (Groq/Ollama).")

    para_text = st.text_area("Enter text to rewrite:", height=200, key="para_text")
    show_counter(para_text)
    mode = st.selectbox("Rewrite mode:", get_modes(), key="para_mode")

    if st.button("Rewrite", key="para_btn"):
        if para_text and para_text.strip():
            with st.spinner("Rewriting..."):
                out = paraphrase_text(para_text, mode=mode)
            if out["error"]:
                st.warning(out["error"])
            else:
                st.success("Done!")
                st.subheader("Rewritten Text")
                st.write(out["result"])

                with st.expander("Compare original vs rewritten"):
                    oc1, oc2 = st.columns(2)
                    with oc1:
                        st.markdown("**Original**")
                        st.text_area("Original", value=para_text, height=250, key="para_orig")
                    with oc2:
                        st.markdown("**Rewritten**")
                        st.text_area("Rewritten", value=out["result"], height=250, key="para_new")

                dlc1, dlc2 = st.columns(2)
                dlc1.download_button(
                    label="Download as TXT",
                    data=out["result"],
                    file_name="rewritten_text.txt",
                    mime="text/plain"
                )
                dlc2.download_button(
                    label="Download as Word (.docx)",
                    data=text_to_docx("Rewritten Text", out["result"]),
                    file_name="rewritten_text.docx",
                    mime=DOCX_MIME,
                    key="para_docx"
                )
        else:
            st.warning("Please enter some text to rewrite.")
elif choice == "Reference Finder":
    st.header("🔗 Reference Finder")
    st.caption("Finds real academic papers with DOIs via CrossRef — no AI key required. (Requires internet.)")

    query = st.text_input("Search for references on a topic or title:", key="ref_query")
    colr1, colr2 = st.columns([1, 1])
    with colr1:
        style = st.selectbox("Citation style:", ["APA", "IEEE", "MLA"], key="ref_style")
    with colr2:
        num = st.slider("Number of references:", min_value=1, max_value=20, value=8, key="ref_num")

    if st.button("Find References", key="ref_btn"):
        if query and query.strip():
            with st.spinner("Searching CrossRef..."):
                result = cached_search_references(query, num)
            if result["error"]:
                st.warning(result["error"])
            else:
                raw = result["references"]
                with st.spinner("Ranking results (first run loads the embedding model)…"):
                    refs, reranked = dedupe_and_rank(raw, query)
                removed = len(raw) - len(refs)
                parts = []
                if removed:
                    parts.append(f"merged {removed} duplicate(s)")
                if reranked:
                    parts.append("ranked by relevance")
                suffix = f" ({'; '.join(parts)})" if parts else ""
                st.success(f"Found {len(refs)} reference(s).{suffix}")
                formatted = []
                for i, ref in enumerate(refs, 1):
                    citation = format_real_reference(ref, style=style)
                    formatted.append(citation)
                    st.markdown(f"**{i}.** {citation}")
                    if ref.get("doi"):
                        st.caption(f"DOI: {ref['doi']}  ·  Type: {ref.get('type','n/a')}")

                bib = "\n\n".join(f"{i}. {c}" for i, c in enumerate(formatted, 1))
                st.download_button(
                    label="Download Bibliography as TXT",
                    data=bib,
                    file_name=f"references_{style.lower()}.txt",
                    mime="text/plain"
                )
        else:
            st.warning("Please enter a search query.")
elif choice == "Readability & Quality":
    st.header("📈 Readability & Writing Quality")
    st.caption("Offline analysis — works without any AI key. Paste text or upload a file.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload a PDF or TXT file:", type=["pdf", "txt"], key="read_file")
    with col2:
        input_text = st.text_area("Or paste text here:", height=150, key="read_text")
        show_counter(input_text)

    cset1, cset2 = st.columns(2)
    with cset1:
        long_threshold = st.slider("Flag sentences longer than (words):", min_value=15, max_value=50, value=25, step=1)
    with cset2:
        audience = st.selectbox("Target audience (grade-level check):", list(AUDIENCE_PRESETS.keys()), key="read_audience")

    text = None
    if uploaded_file is not None:
        text, _upload_err = read_uploaded_file(uploaded_file)
        if _upload_err:
            st.warning(_upload_err)
    elif input_text:
        text = input_text

    if st.button("Analyze", key="read_btn"):
        if text and text.strip():
            result = cached_analyze_text(text, long_threshold)
            if result.get("error"):
                st.warning(result["error"])
            else:
                # Readability scores
                st.subheader("Readability")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Flesch Reading Ease", result["flesch_reading_ease"])
                    st.caption(result["flesch_reading_ease_label"])
                with c2:
                    st.metric("Flesch-Kincaid Grade", result["flesch_kincaid_grade"])
                    _lo, _hi = AUDIENCE_PRESETS[audience]
                    _gs = grade_band_status(result["flesch_kincaid_grade"], _lo, _hi)
                    st.caption(("🟢 " if _gs == "on target" else "🟠 ") + _gs)
                with c3:
                    st.metric("Gunning Fog Index", result["gunning_fog"])
                    st.caption("Years of education to read")

                # Counts
                st.subheader("Statistics")
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    st.metric("Words", result["word_count"])
                with s2:
                    st.metric("Sentences", result["sentence_count"])
                with s3:
                    st.metric("Avg words/sentence", result["avg_words_per_sentence"])
                with s4:
                    st.metric("Reading time", f"{result['reading_time_min']} min")

                # Quality flags
                st.subheader("Writing Quality")
                q1, q2, q3 = st.columns(3)
                with q1:
                    st.metric("Passive sentences", f"{result['passive_count']} ({result['passive_percent']}%)")
                with q2:
                    st.metric(f"Long sentences (>{result['long_sentence_threshold']}w)", result["long_sentence_count"])
                with q3:
                    st.metric("Complex words", f"{result['hard_word_count']} ({result['hard_word_percent']}%)")

                if result["long_sentences"]:
                    with st.expander(f"⚠️ {result['long_sentence_count']} long sentence(s) — consider splitting"):
                        for idx, item in enumerate(result["long_sentences"]):
                            st.markdown(f"**({item['words']} words)** {item['sentence']}")
                            if st.button("✏️ Send to Paraphraser", key=f"para_long_{idx}"):
                                _send_to_paraphraser(item["sentence"])

                if result["passive_sentences"]:
                    with st.expander(f"🔎 {result['passive_count']} possible passive-voice sentence(s)"):
                        for idx, sent in enumerate(result["passive_sentences"]):
                            st.markdown(f"- {sent}")
                            if st.button("✏️ Send to Paraphraser", key=f"para_pass_{idx}"):
                                _send_to_paraphraser(sent)

                # Readability heat-strip (per-sentence difficulty).
                with st.expander("🌡️ Readability heat-strip (where the hard sentences are)"):
                    render_heatstrip(analyze_sentences(text))

                # Downloadable report
                report = (
                    "Readability & Writing Quality Report\n"
                    "====================================\n"
                    f"Flesch Reading Ease : {result['flesch_reading_ease']} ({result['flesch_reading_ease_label']})\n"
                    f"Flesch-Kincaid Grade: {result['flesch_kincaid_grade']}\n"
                    f"Gunning Fog Index   : {result['gunning_fog']}\n\n"
                    f"Words               : {result['word_count']}\n"
                    f"Sentences           : {result['sentence_count']}\n"
                    f"Avg words/sentence  : {result['avg_words_per_sentence']}\n"
                    f"Reading time        : {result['reading_time_min']} min\n\n"
                    f"Passive sentences   : {result['passive_count']} ({result['passive_percent']}%)\n"
                    f"Long sentences      : {result['long_sentence_count']}\n"
                    f"Complex words       : {result['hard_word_count']} ({result['hard_word_percent']}%)\n"
                )
                st.download_button(
                    label="Download Report as TXT",
                    data=report,
                    file_name="readability_report.txt",
                    mime="text/plain"
                )
        else:
            st.warning("Please paste text or upload a file to analyze.")
elif choice == "Chat with your Paper":
    st.header("💬 Chat with your Paper")
    st.caption("Upload a paper, then ask questions answered from its content. Hybrid semantic + keyword retrieval works offline; the written answer needs an AI backend.")

    col_u, col_k = st.columns([3, 1])
    with col_u:
        uploaded_file = st.file_uploader("Upload a PDF or TXT paper:", type=["pdf", "txt"], key="chat_file")
    with col_k:
        top_k = st.slider("Passages to retrieve", min_value=3, max_value=12, value=6, key="chat_k")

    if uploaded_file is not None and st.button("Load Document", key="chat_load"):
        doc_text, _chat_err = read_uploaded_file(uploaded_file)
        if _chat_err:
            st.warning(_chat_err)
            st.session_state.pop("chat_doc", None)
        else:
            with st.spinner("Indexing document (first run loads the embedding model)..."):
                doc = prepare_document(doc_text)
            if doc["error"]:
                st.warning(doc["error"])
                st.session_state.pop("chat_doc", None)
            else:
                st.session_state["chat_doc"] = doc
                st.session_state["chat_doc_name"] = uploaded_file.name
                st.session_state["chat_history"] = []  # reset conversation for new doc
                with st.spinner("Summarizing the paper…"):
                    st.session_state["chat_overview"] = document_overview(doc)
                st.success(f"Loaded '{uploaded_file.name}' — indexed {len(doc['chunks'])} passage(s).")

    if "chat_doc" in st.session_state:
        doc = st.session_state["chat_doc"]
        st.info(f"📄 {st.session_state.get('chat_doc_name', 'document')}  ·  🔎 Retrieval: {doc.get('mode', 'lexical')}")

        c1, c2, c3 = st.columns([1.1, 1.6, 0.9])
        with c1:
            length_mode = st.radio("Answer length", ["Detailed", "Brief"], horizontal=True, key="chat_len")
        with c2:
            use_general = st.checkbox("Use general knowledge too (labeled separately)", value=True, key="chat_general")
        with c3:
            if st.button("🗑️ Clear chat", key="chat_clear"):
                st.session_state["chat_history"] = []
        find_only = st.checkbox("🔎 Find passages only (no AI answer — instant, works offline)",
                                value=False, key="chat_find")
        detailed = (length_mode == "Detailed")

        history = st.session_state.setdefault("chat_history", [])

        def _ask(q):
            q = (q or "").strip()
            if not q:
                return
            if find_only:
                passages = retrieve_context(q, doc, k=top_k)
                history.append({"question": q, "answer": None, "error": None,
                                "passages": passages, "find_only": True})
            else:
                with st.spinner("Thinking…"):
                    res = answer_question(q, doc, k=top_k, history=history,
                                          detailed=detailed, use_general=use_general)
                history.append({
                    "question": q, "answer": res["answer"],
                    "error": res["error"], "passages": res["passages"],
                })
            st.rerun()

        # Auto-summary + suggested questions (from document_overview at load time).
        ov = st.session_state.get("chat_overview") or {}
        if ov.get("summary"):
            st.markdown(f"**📄 Summary:** {ov['summary']}")
        if ov.get("questions"):
            st.caption("💡 Suggested questions — click to ask:")
            qcols = st.columns(2)
            for i, sq in enumerate(ov["questions"]):
                if qcols[i % 2].button(sq, key=f"chat_sugg_{i}"):
                    _ask(sq)

        # Render the running conversation.
        for turn in history:
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                if turn.get("find_only"):
                    if turn.get("passages"):
                        st.caption("🔎 Most relevant passages (no AI answer):")
                        for i, p in enumerate(turn["passages"], 1):
                            sec = f" · {p['section']}" if p.get("section") else ""
                            st.markdown(f"**Passage {i}**{sec} · relevance {p['score']:.2f}")
                            st.write(p["chunk"])
                    else:
                        st.warning("No matching passages found for that query.")
                else:
                    if turn.get("answer"):
                        st.write(turn["answer"])
                    elif turn.get("error"):
                        st.warning(turn["error"])
                    if turn.get("passages"):
                        with st.expander(f"📑 Source passages ({len(turn['passages'])})"):
                            for i, p in enumerate(turn["passages"], 1):
                                sec = f" · {p['section']}" if p.get("section") else ""
                                st.markdown(f"**Passage {i}**{sec} · relevance {p['score']:.2f}")
                                st.write(p["chunk"])

        question = st.chat_input("Ask a question about the paper...")
        if question and question.strip():
            _ask(question)
    else:
        st.info("Upload a paper and click **Load Document** to begin.")
elif choice == "ZeroGPT — AI vs Human Detector":
    st.header("🤖 ZeroGPT — AI vs Human Detector")
    st.caption("🧪 Experimental & offline. Combines a trained classifier (HC3) with style heuristics.")
    st.warning(
        "⚠️ **AI detection is unreliable.** Treat this as a weak signal, not proof. "
        "Short text, edited AI text, and AI from models other than ChatGPT can evade it, "
        "and confident human writing can be falsely flagged. Never use it to accuse anyone."
    )

    text = st.text_area("Paste a paragraph to analyze:", height=220, key="zgpt_text")
    show_counter(text)

    if st.button("Detect", key="zgpt_btn"):
        if text and text.strip():
            with st.spinner("Analyzing (first run loads the language model)..."):
                result = detect_ai_text(text, use_perplexity=True)
            if result.get("error"):
                st.warning(result["error"])
            else:
                score = result["ai_score"]
                verdict = result["verdict"]
                # Verdict banner
                if score >= 65:
                    st.error(f"🤖 {verdict} — {score}% AI")
                elif score >= 35:
                    st.warning(f"❓ {verdict} — {score}% AI")
                else:
                    st.success(f"🧑 {verdict} — {score}% AI")

                st.progress(min(int(score), 100))
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("AI-generated", f"{result['ai_score']}%")
                with c2:
                    ppl = result.get("perplexity_score")
                    st.metric("Perplexity signal", f"{ppl}%" if ppl is not None else "n/a")
                with c3:
                    ml = result.get("ml_score")
                    st.metric("Trained model", f"{ml}%" if ml is not None else "n/a")
                with c4:
                    st.metric("Style heuristic", f"{result['heuristic_score']}%")
                st.caption(f"Method: {result.get('method', 'heuristic only')}")

                # Signal breakdown
                sig = result["signals"]
                pinfo = result.get("perplexity_info")
                with st.expander("Why? — signal breakdown"):
                    if pinfo:
                        st.markdown(
                            f"- **Perplexity (predictability):** {pinfo['perplexity']} "
                            f"(lower = more AI-like) · **burstiness:** {pinfo['burstiness']}"
                        )
                    st.markdown(
                        f"- **Trained model (HC3) P(AI):** {result.get('ml_score', 'n/a')}%\n"
                        f"- **Sentence uniformity (burstiness):** CV={sig['burstiness_cv']} "
                        f"→ AI-signal {sig['uniformity_score']} (lower variation = more AI-like)\n"
                        f"- **Vocabulary diversity (TTR):** {sig['vocabulary_diversity_ttr']} "
                        f"→ AI-signal {sig['diversity_score']}\n"
                        f"- **AI-giveaway phrases:** {sig['ai_phrase_hits']} hit(s) "
                        f"→ AI-signal {sig['phrase_score']}"
                    )
                    if result["flagged_phrases"]:
                        st.markdown("**Flagged phrases:** " + ", ".join(f"`{p}`" for p in result["flagged_phrases"]))

                st.caption(f"Analyzed {result['word_count']} words across {result['sentence_count']} sentences.")
        else:
            st.warning("Please paste some text to analyze.")
elif choice == "Flow & Cohesion Map":
    st.header("🔀 Flow & Cohesion Map")
    st.caption("🧪 Offline. Flags abrupt topic jumps between consecutive paragraphs so you can add "
               "transitions. Reuses the embedding model (TF-IDF fallback).")
    text = analysis_input("coh", "Or paste your paper (keep blank lines between paragraphs):")
    if st.button("Analyze flow", key="coh_btn"):
        if text:
            with st.spinner("Mapping paragraph-to-paragraph flow…"):
                r = analyze_cohesion(text)
            if r.get("error"):
                st.warning(r["error"])
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Paragraphs", r["paragraphs"])
                c2.metric("Avg ¶→¶ similarity", r["avg_similarity"])
                c3.metric("Weak transitions", r["weak_count"])
                st.caption(f"Retrieval: {r['mode']} · weak = below {r['threshold']:.2f} and no connector word")
                for p in r["pairs"]:
                    icon = "🔴" if p["weak"] else ("🟢" if not p["has_connector"] else "🔗")
                    st.markdown(f"{icon} **¶{p['from']} → ¶{p['to']}** · similarity {p['similarity']:.2f}"
                                + (" — weak jump, consider a transition" if p["weak"] else ""))
                    if p["weak"]:
                        st.caption(f"¶{p['to']} starts: “{p['preview']}”")
        else:
            st.warning("Please paste or upload some text with at least 2 paragraphs.")
elif choice == "Citation Density & Uncited Claims":
    st.header("📑 Citation Density & Uncited Claims")
    st.caption("🧪 Offline. Counts in-text citations and flags evidence-bearing sentences that cite nothing "
               "(the #1 reviewer complaint).")
    text = analysis_input("claim")
    if st.button("Analyze citations", key="claim_btn"):
        if text:
            r = analyze_claims(text)
            if r.get("error"):
                st.warning(r["error"])
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Citation markers", r["citation_markers"])
                c2.metric("Citations / 1000 words", r["citations_per_1000_words"])
                c3.metric("Uncited claims", r["uncited_count"])
                if r["per_section"]:
                    st.subheader("Per-section citation density")
                    for s in r["per_section"]:
                        st.markdown(f"- **{s['section']}** — {s['markers']} citation(s) in {s['words']} words "
                                    f"({s['per_1000']}/1000)")
                if r["uncited_claims"]:
                    with st.expander(f"⚠️ {r['uncited_count']} claim sentence(s) with no nearby citation"):
                        for s in r["uncited_claims"]:
                            st.markdown(f"- {s}")
                else:
                    st.success("No obvious uncited claims found.")
        else:
            st.warning("Please paste or upload some text.")
elif choice == "Section Completeness Check":
    st.header("✅ Section Completeness Check")
    st.caption("🧪 Offline. Checks your draft's sections (detected from headings) against the expected "
               "structure for a paper type, with per-section word budgets.")
    _ptypes = get_paper_types()
    _labels = [n for _, n in _ptypes]
    _keys = [k for k, _ in _ptypes]
    _sel = st.selectbox("Paper type:", _labels, key="struct_type")
    paper_type = _keys[_labels.index(_sel)]
    text = analysis_input("struct", "Or paste your paper (include section headings, one per line):")
    if st.button("Check structure", key="struct_btn"):
        if text:
            r = check_structure(text, paper_type)
            if r.get("error"):
                st.warning(r["error"])
            else:
                st.metric("Completeness", f"{r['completeness_pct']}%")
                st.write(f"**Present:** {', '.join(r['present']) or '—'}")
                if r["missing"]:
                    st.warning("Missing: " + ", ".join(r["missing"]))
                if not r["order_ok"]:
                    st.info("Section order differs from the recommended template order.")
                if r["extra"]:
                    st.caption("Unrecognized / extra headings: " + ", ".join(r["extra"]))
                st.subheader("Word counts vs target")
                for row in r["rows"]:
                    icon = {"ok": "🟢", "under": "🟡", "over": "🟠"}.get(row["status"], "⚪")
                    st.markdown(f"{icon} **{row['section']}** — {row['words']} words (target {row['target']})")
        else:
            st.warning("Please paste or upload a draft that includes section headings.")
elif choice == "Abstract Scorecard":
    st.header("📋 Abstract Scorecard")
    st.caption("🧪 Offline. Scores an abstract on the 5 expected moves (background, gap/aim, methods, "
               "results, conclusion) plus length and readability.")
    abs_text = st.text_area("Paste your abstract (or a full paper with an 'Abstract' heading):",
                            height=180, key="abs_text")
    show_counter(abs_text)
    if st.button("Score abstract", key="abs_btn"):
        if abs_text and abs_text.strip():
            r = score_abstract(abs_text)
            if r.get("error"):
                st.warning(r["error"])
            else:
                st.metric("Score", f"{r['score']}/100")
                st.caption(f"{r['word_count']} words · length: {r['length_status']} ({r['length_message']})"
                           + (f" · Flesch {r['flesch_reading_ease']}" if r['flesch_reading_ease'] is not None else ""))
                st.subheader("Expected moves")
                for move, ok in r["moves"].items():
                    st.markdown(f"{'✅' if ok else '❌'} {move}")
                if r["tips"]:
                    st.subheader("Suggestions")
                    for t in r["tips"]:
                        st.markdown(f"- {t}")
        else:
            st.warning("Please paste your abstract.")
elif choice == "Tone Auditor (Hedging/Overclaiming)":
    st.header("🎚️ Academic Tone Auditor")
    st.caption("🧪 Offline. Counts hedging, overclaiming, filler/weasel words and first-person usage, "
               "with example sentences — to help calibrate claim strength.")
    text = analysis_input("tone")
    if st.button("Audit tone", key="tone_btn"):
        if text:
            r = analyze_tone(text)
            if r.get("error"):
                st.warning(r["error"])
            else:
                cats = r["categories"]
                cols = st.columns(len(cats))
                for col, (cat, info) in zip(cols, cats.items()):
                    col.metric(cat, info["count"], f"{info['per_1000_words']}/1k words")
                for cat, info in cats.items():
                    if info["terms"]:
                        st.caption(f"**{cat}:** " + ", ".join(info["terms"]))
                if r["examples"]["overclaim"]:
                    with st.expander("Possible overclaims (strong/absolute wording)"):
                        for s in r["examples"]["overclaim"]:
                            st.markdown(f"- {s}")
                if r["examples"]["over_hedged"]:
                    with st.expander("Over-hedged sentences (2+ hedges)"):
                        for s in r["examples"]["over_hedged"]:
                            st.markdown(f"- {s}")
        else:
            st.warning("Please paste or upload some text.")
elif choice == "Verify References":
    st.header("🔎 Verify References")
    st.caption("Paste your reference list — each entry is checked against CrossRef to catch fabricated / "
               "AI-hallucinated citations. Requires internet; checks up to 20 references.")
    ref_text = st.text_area("Paste your references (one per line):", height=240, key="verify_text")
    if st.button("Verify references", key="verify_btn"):
        if ref_text and ref_text.strip():
            with st.spinner("Checking each reference against CrossRef…"):
                r = cached_verify_references(ref_text)
            if r.get("error"):
                st.warning(r["error"])
            else:
                c = r["counts"]
                m1, m2, m3 = st.columns(3)
                m1.metric("✅ Verified", c["Verified"])
                m2.metric("⚠️ Possible", c["Possible match"])
                m3.metric("❌ Not found", c["Not found"])
                if r["counts"]["Not found"]:
                    st.warning(f"{r['counts']['Not found']} reference(s) could not be matched — verify "
                               "these manually; they may be fabricated.")
                if r["truncated"]:
                    st.caption(f"Checked the first {r['checked']} of {r['total_found']} references.")
                for item in r["results"]:
                    icon = {"Verified": "✅", "Possible match": "⚠️", "Not found": "❌"}[item["verdict"]]
                    with st.expander(f"{icon} {item['verdict']} ({item['confidence']}%) — {item['entry'][:80]}"):
                        st.markdown(f"**Your entry:** {item['entry']}")
                        if item["matched_title"]:
                            st.markdown(f"**Closest real paper:** {item['matched_title']}")
                            if item["url"]:
                                st.markdown(f"**Link:** {item['url']}")
                        else:
                            st.markdown("_No matching paper found in CrossRef — verify this citation "
                                        "manually; it may be fabricated._")
                        if item["doi_mismatch"]:
                            st.warning(f"⚠️ DOI mismatch: your entry's DOI (`{item['entry_doi']}`) ≠ the "
                                       f"matched paper's DOI (`{item['matched_doi']}`).")
        else:
            st.warning("Please paste at least one reference.")
elif choice == "Terminology & Acronym Check":
    st.header("🔤 Terminology & Acronym Consistency")
    st.caption("🧪 Offline. Flags acronyms used before they're defined or never expanded, and inconsistent "
               "term spellings (e.g. 'dataset' vs 'data-set').")
    text = analysis_input("term")
    if st.button("Check consistency", key="term_btn"):
        if text:
            r = analyze_consistency(text)
            if r.get("error"):
                st.warning(r["error"])
            else:
                st.subheader("Acronyms")
                if r["acronym_issues"]:
                    for a in r["acronym_issues"]:
                        st.markdown(f"- **{a['acronym']}** ({a['uses']} use(s)) — {a['issue']}")
                else:
                    st.success(f"No acronym issues found ({r['acronyms_found']} acronym(s) detected).")
                st.subheader("Inconsistent term spellings")
                if r["term_variants"]:
                    for v in r["term_variants"]:
                        forms = ", ".join(f"{k} ×{n}" for k, n in v["forms"].items())
                        st.markdown(f"- {forms}")
                else:
                    st.success("No inconsistent term spellings detected.")
        else:
            st.warning("Please paste or upload some text.")
elif choice == "Repetition / Self-Overlap Finder":
    st.header("♻️ Repetition / Self-Overlap Finder")
    st.caption("🧪 Offline. Finds near-duplicate sentences within your paper (recycled phrasing, an idea "
               "restated). Compares the document to itself — not an external corpus.")
    text = analysis_input("rep")
    rep_threshold = st.slider("Similarity threshold:", min_value=0.5, max_value=0.95, value=0.8, step=0.05)
    if st.button("Find repetition", key="rep_btn"):
        if text:
            r = find_repetition(text, threshold=rep_threshold)
            if r.get("error"):
                st.warning(r["error"])
            else:
                st.metric("Repeated sentence pairs", r["pair_count"])
                if r["pair_count"] > len(r["pairs"]):
                    st.caption(f"Showing the top {len(r['pairs'])} of {r['pair_count']} pairs.")
                if r["pairs"]:
                    for p in r["pairs"]:
                        with st.expander(f"Sentences {p['i']} ↔ {p['j']} — {int(p['similarity']*100)}% similar"):
                            st.markdown(f"**[{p['i']}]** {p['a']}")
                            st.markdown(f"**[{p['j']}]** {p['b']}")
                else:
                    st.success("No significant repetition found at this threshold.")
        else:
            st.warning("Please paste or upload some text.")
elif choice == "Keyword & Contribution Extractor":
    st.header("🏷️ Keyword & Contribution Extractor")
    st.caption("🧪 Offline. Suggests candidate keywords (TF-IDF) and surfaces likely contribution / novelty "
               "statements from your draft.")
    text = analysis_input("kw")
    if st.button("Extract", key="kw_btn"):
        if text:
            r = extract_keywords(text)
            if r.get("error"):
                st.warning(r["error"])
            else:
                st.subheader("Candidate keywords")
                if r["keywords"]:
                    st.markdown("  ".join(f"`{k}`" for k in r["keywords"]))
                else:
                    st.info("No keywords extracted.")
                st.subheader("Likely contribution statements")
                if r["contributions"]:
                    for s in r["contributions"]:
                        st.markdown(f"- {s}")
                else:
                    st.info("No explicit contribution statements detected (look for 'we propose', 'our contribution', etc.).")
        else:
            st.warning("Please paste or upload some text.")
elif choice == "AI Prompt Generator":
    st.header("✨ AI Prompt Generator")
    st.caption("🧪 Turn a rough idea (written casually OR formally) into a clean, structured prompt for "
               "an LLM. Output follows the prompt-quality rule-set: role to context to task to output "
               "format, with zero em dashes, smart quotes, or AI-tell filler. Pick a framework, "
               "generate, then optionally refine. Needs an AI backend.")

    idea = st.text_area("Describe your idea / what you want the AI to do:", height=140, key="pg_idea")
    show_counter(idea)
    pgc1, pgc2 = st.columns(2)
    with pgc1:
        framework = st.selectbox("Prompt framework:", list(PROMPT_FRAMEWORKS.keys()), key="pg_framework")
    with pgc2:
        target_model = st.selectbox("Target model:", TARGET_MODELS, key="pg_model")
    task_type = st.text_input("(Optional) task type or domain — e.g. 'email', 'code', 'research summary':",
                              key="pg_tasktype")
    pgc3, pgc4 = st.columns([1.4, 1])
    with pgc3:
        mode_label = st.selectbox(
            "Prompt style:",
            ["Outcome-oriented (let the AI choose its method)",
             "Procedure-oriented (spell out the steps)"],
            key="pg_mode",
        )
    with pgc4:
        self_check = st.checkbox("Add a self-check checklist for the target model", key="pg_selfcheck")
    mode = "procedure" if mode_label.startswith("Procedure") else "outcome"

    if st.button("Generate prompt", key="pg_gen"):
        if idea and idea.strip():
            with st.spinner("Generating your prompt…"):
                res = generate_prompt(idea, framework, target_model, task_type, mode, self_check)
            if res.get("error"):
                st.warning(res["error"])
                st.session_state.pop("pg_refine", None)  # don't leave a stale 'Refined' banner
            else:
                st.session_state["pg_result"] = res["prompt"]
                st.session_state["pg_ctx"] = {"idea": idea, "framework": framework}
                st.session_state.pop("pg_refine", None)  # clear stale critique
        else:
            st.warning("Please describe your idea first.")

    if st.session_state.get("pg_result"):
        prompt_out = st.session_state["pg_result"]
        st.subheader("Your prompt")
        st.code(prompt_out, language=None)  # st.code shows a one-click copy button
        d1, d2, d3 = st.columns([1, 1, 1.4])
        d1.download_button("Download .txt", data=prompt_out, file_name="prompt.txt",
                           mime="text/plain", key="pg_txt")
        d2.download_button("Download .docx", data=text_to_docx("Generated Prompt", prompt_out),
                           file_name="prompt.docx", mime=DOCX_MIME, key="pg_docx")
        if d3.button("✨ Refine (critique & improve)", key="pg_refine_btn"):
            ctx = st.session_state.get("pg_ctx", {})
            with st.spinner("Critiquing and refining…"):
                ref = refine_prompt(prompt_out, ctx.get("idea", ""), ctx.get("framework", ""))
            if ref.get("error"):
                st.warning(ref["error"])
            else:
                st.session_state["pg_result"] = ref["improved"]
                st.session_state["pg_refine"] = ref
                st.rerun()

        if st.session_state.get("pg_refine"):
            rf = st.session_state["pg_refine"]
            st.success(f"Refined · quality score of the previous draft: {rf['score']}/100")
            if rf.get("warning"):
                st.warning(rf["warning"])
            if rf["issues"]:
                with st.expander("What the refine pass flagged & fixed"):
                    for it in rf["issues"]:
                        st.markdown(f"- {it}")

elif choice == "AI Peer Reviewer":
    st.header("🧑‍⚖️ AI Peer Reviewer")
    st.caption("🧪 Score a paper or draft like a strict journal reviewer. Each criterion is rated 0-100 "
               "(50 = a solid, publishable standard) with a severity-tagged weakness, the exact sentence "
               "it judged, and a fix. Then get a prioritized revision plan and a one-click revised draft "
               "you can re-score. Needs an AI backend.")

    pr_paper = st.text_area("Paste the paper / draft to review:", height=240, key="pr_paper")
    show_counter(pr_paper)

    with st.expander("Optional: paste a reference paper (score against it instead of the general standard)"):
        pr_ref = st.text_area("Reference paper text:", height=140, key="pr_ref")

    rub_choice = st.radio("Rubric:", ["Auto-generate from the paper", "Standard academic rubric"],
                          horizontal=True, key="pr_rubric_mode")

    if st.button("Review paper", key="pr_go"):
        paper = (pr_paper or "").strip()
        if not paper:
            st.warning("Please paste the paper text first.")
        else:
            fb = False
            if rub_choice.startswith("Auto"):
                with st.spinner("Designing a rubric for this paper…"):
                    gen = generate_checklist(paper)
                if gen.get("error"):
                    st.warning(gen["error"])
                checklist = gen.get("checklist")
                fb = bool(gen.get("fallback"))
            else:
                checklist = DEFAULT_RUBRIC
            with st.spinner("Scoring against each criterion…"):
                res = score_paper(paper, checklist=checklist, reference_text=st.session_state.get("pr_ref", ""))
            if res.get("error"):
                st.warning(res["error"])
                st.session_state.pop("pr_result", None)
            else:
                res["fallback_rubric"] = fb
                res["checklist"] = checklist          # keep for re-scoring the revised draft
                res["paper"] = paper
                st.session_state["pr_result"] = res
                st.session_state.pop("pr_revised", None)

    if st.session_state.get("pr_result"):
        res = st.session_state["pr_result"]
        total = res["total_score"]
        mode_label = ("vs your reference paper (50 = as good as it)" if res.get("mode") == "reference"
                      else "vs a publishable standard (50 = the bar)")
        delta = None
        if res.get("prev_total") is not None:
            delta = f"{total - res['prev_total']:+.1f} vs previous draft"
        mc1, mc2 = st.columns([1, 2])
        mc1.metric("Weighted total", f"{total}/100", delta=delta)
        mc2.caption(mode_label)
        st.progress(min(1.0, max(0.0, total / 100.0)))
        if res.get("warning"):
            st.warning(res["warning"])
        if res.get("fallback_rubric"):
            st.info("Used the built-in academic rubric (rubric auto-generation was unavailable).")

        _SEV_EMOJI = {"Critical": "🔴", "Major": "🟠", "Minor": "🟡", "None": "🟢", "": "⚪"}
        st.subheader("Per-criterion review")
        for it in res["items"]:
            tag = "🖼️" if it["type"] == "image" else "📝"
            head = it["content"][:80] + ("…" if len(it["content"]) > 80 else "")
            score_label = "not scored" if it.get("unavailable") else f"{it['score']}/100"
            sev = it.get("severity", "")
            badge = _SEV_EMOJI.get(sev, "⚪")
            sev_txt = f" · {sev}" if sev else ""
            with st.expander(f"{badge} {tag} {score_label}{sev_txt} · w {it['weight']:.2f} — {head}"):
                st.markdown(f"**Criterion:** {it['content']}")
                st.markdown(f"**Score:** {score_label}   ·   **Severity:** {sev or 'n/a'}   ·   "
                            f"**Weight:** {it['weight']:.2f}")
                if it.get("evidence"):
                    loc = f"  _(section: {it['location']})_" if it.get("location") else ""
                    st.markdown(f"**Evidence (from your paper):** “{it['evidence']}”{loc}")
                st.markdown(f"**Reviewer reasoning:** {it['reasoning']}")
                if it.get("fix"):
                    st.markdown(f"**Suggested fix:** {it['fix']}")

        # Prioritized revision plan + revise -> re-score loop.
        plan = build_revision_plan(res)
        if plan:
            st.subheader("📋 Revision plan (highest-impact fixes first)")
            for i, p in enumerate(plan, 1):
                sev_lbl = f"{_SEV_EMOJI.get(p.get('severity',''), '⚪')} **{p['severity']}** · " if p.get("severity") else ""
                st.markdown(f"{i}. {sev_lbl}{p['content']} ({p['score']}/100)")
                if p.get("fix"):
                    st.markdown(f"&nbsp;&nbsp;&nbsp;↳ _Fix:_ {p['fix']}")

            if st.button("✍️ Generate a revised draft addressing these", key="pr_revise"):
                with st.spinner("Drafting a revision…"):
                    rev = revise_paper(res.get("paper", ""), plan)
                if rev.get("error"):
                    st.warning(rev["error"])
                else:
                    st.session_state["pr_revised"] = rev["revised"]

        if st.session_state.get("pr_revised"):
            st.markdown("**Revised draft** — review carefully; `[ADD: …]` marks where you must supply specifics:")
            st.text_area("Revised draft", value=st.session_state["pr_revised"], height=240, key="pr_revised_view")
            rv1, rv2, rv3 = st.columns(3)
            rv1.download_button("Download revised (.txt)", data=st.session_state["pr_revised"],
                                file_name="revised_paper.txt", mime="text/plain", key="pr_rev_txt")
            rv2.download_button("Download revised (.docx)",
                                data=text_to_docx("Revised Paper", st.session_state["pr_revised"]),
                                file_name="revised_paper.docx", mime=DOCX_MIME, key="pr_rev_docx")
            if rv3.button("🔁 Re-score the revised draft", key="pr_rescore"):
                with st.spinner("Re-scoring the revised draft…"):
                    new = score_paper(st.session_state["pr_revised"], checklist=res.get("checklist"),
                                      reference_text=st.session_state.get("pr_ref", ""))
                if new.get("error"):
                    st.warning(new["error"])
                else:
                    new["fallback_rubric"] = res.get("fallback_rubric")
                    new["checklist"] = res.get("checklist")
                    new["paper"] = st.session_state["pr_revised"]
                    new["prev_total"] = res["total_score"]
                    st.session_state["pr_result"] = new
                    st.session_state.pop("pr_revised", None)
                    st.rerun()

        # Downloadable full review report (with evidence, severity, and fixes).
        lines = [f"AI Peer Review — weighted total {total}/100 ({mode_label})", ""]
        for it in res["items"]:
            sl = "not scored" if it.get("unavailable") else f"{it['score']}/100"
            sev = f", {it['severity']}" if it.get("severity") else ""
            lines.append(f"[{sl}, weight {it['weight']:.2f}{sev}] {it['content']}")
            if it.get("evidence"):
                lines.append(f"    Evidence: \"{it['evidence']}\"" + (f" (section: {it['location']})" if it.get("location") else ""))
            lines.append(f"    Reasoning: {it['reasoning']}")
            if it.get("fix"):
                lines.append(f"    Fix: {it['fix']}")
            lines.append("")
        report_txt = "\n".join(lines)
        rc1, rc2 = st.columns(2)
        rc1.download_button("Download review (.txt)", data=report_txt, file_name="peer_review.txt",
                            mime="text/plain", key="pr_txt")
        rc2.download_button("Download review (.docx)", data=text_to_docx("AI Peer Review", report_txt),
                            file_name="peer_review.docx", mime=DOCX_MIME, key="pr_docx")