import streamlit as st
import os
import sys

# Add the project root to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.pdf_processor import extract_text_from_pdf, summarize_text
from src.utils.text_analyzer import PlagiarismDetector, check_plagiarism, add_reference_document
from src.utils.citation_manager import suggest_citations, format_citation
from src.utils.content_generator import generate_comprehensive_paper, generate_section_only, generate_quick_paper, get_available_paper_types
from src.utils.default_paper_generator import generate_default_paper, create_word_document
from src.utils.topic_type_predictor import TopicTypePredictor
from src.utils.grammar_checker import check_grammar_text
import matplotlib.pyplot as plt
import io
import zipfile
import pickle

# Ensure NLTK data is downloaded
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set page config
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)

# Main title
st.title("📚 Research Paper Assistant")

# Sidebar navigation
menu = ["Content Generation", "Paper Analysis", "Citation Assistant", "Grammar Check", "Plagiarism Detection"]
choice = st.sidebar.selectbox("Select a Feature", menu)

# Feature logic (placeholders)
if choice == "Content Generation":
    st.header("📝 Content Generation")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Default Research Paper Generator",
        "Quick Paper Generator",
        "Section Generator/Editor",
        "Paper Guide"
    ])

    with tab1:
        st.subheader("Default Research Paper Generator (Best & Realistic)")
        topic = st.text_input("Enter your research topic:", key="def_topic")
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
            images.append({
                "section": "Results",
                "image_bytes": chart_bytes,
                "caption": f"Sample chart related to '{topic}'"
            })
            plt.close(fig)

        if st.button("Generate Default Paper", key="def_gen"):
            if topic:
                paper = generate_default_paper(topic, paper_type, suggestions_dict, images)
                st.success("Default research paper generated!")
                for section, data in paper['sections'].items():
                    st.markdown(f"### {section}")
                    st.write(data['content'])
                    # Show image preview in Results section
                    if wants_image and section == "Results" and chart_bytes:
                        st.image(chart_bytes, caption=f"Sample chart for '{topic}'", use_column_width=True)
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
        topic = st.text_input("Enter your research topic:", key="quick_topic")
        paper_types = [t['key'] for t in get_available_paper_types()]
        paper_type = st.selectbox("Select paper type:", paper_types, key="quick_type")
        word_count = st.slider("Word count per section (bulk/in-depth):", min_value=200, max_value=2000, value=500, step=50, key="quick_wordcount")
        if st.button("Generate In-Depth Quick Paper", key="quick_gen"):
            if topic:
                paper = generate_quick_paper(topic, paper_type)
                st.success("In-depth quick research paper generated!")
                txt_content = f"{paper['sections'].get('Title', '')}\n\n"
                for section, data in paper['sections'].items():
                    if section != 'Title':
                        # If section is too short, pad with extra prompt
                        content = data['content']
                        if len(content.split()) < word_count:
                            # Simulate bulk by repeating or expanding (in real use, call a generator with a higher word count prompt)
                            content += '\n' + f"(Expand this section to at least {word_count} words. Add more detail, examples, and depth.)"
                        st.markdown(f"### {section}")
                        st.write(content)
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
        topic = st.text_input("Enter your research topic:", key="sec_topic")
        paper_types = [t['key'] for t in get_available_paper_types()]
        paper_type = st.selectbox("Select paper type:", paper_types, key="sec_type")
        section_list = [
            "Title", "Abstract", "Introduction", "Literature Review", "Methodology", "Results", "Discussion", "Conclusion"
        ]
        section = st.selectbox("Select section to generate/edit:", section_list, key="sec_section")
        if st.button("Generate Section", key="sec_gen"):
            if topic and section:
                content = generate_section_only(topic, section, paper_type)
                st.success(f"Section '{section}' generated!")
                st.markdown(f"### {section}")
                st.write(content)
            else:
                st.warning("Please enter a topic and select a section.")

    with tab4:
        st.subheader("Paper Guide (Section-wise, Detailed)")
        topic = st.text_input("Enter your research topic:", key="guide_topic")
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
                for section in selected_sections:
                    st.markdown(f"### {section} Guide")
                    prompt = f"Write a detailed, information-rich guide for the '{section}' section of a {paper_type} research paper on '{topic}'. Focus on structure, tips, and what to include. Target length: {word_count} words. {suggestions}"
                    content = generate_section_only(topic, section, paper_type)
                    if len(content.split()) < word_count:
                        content += '\n' + f"(Expand this section to at least {word_count} words. Include tips, structure, and common mistakes to avoid.)"
                    st.write(content)
                    guide += f"\n\n## {section}\n{content}"
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
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
            else:
                text = uploaded_file.read().decode("utf-8")
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
                    mime="text/plain"
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
                    mime="text/plain"
                )
            else:
                st.warning("Please paste some text.")
elif choice == "Citation Assistant":
    st.header("📚 Citation Assistant")
    query = st.text_input("Enter your research query or topic for citations:")
    style = st.selectbox("Select citation style:", ["APA", "IEEE", "MLA"])
    num_citations = st.slider("Number of citations:", min_value=1, max_value=8, value=3)
    if st.button("Generate Citations"):
        if query:
            citations = suggest_citations(query)
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
            st.warning("Please enter a query or topic.")
elif choice == "Grammar Check":
    st.header("✍️ Grammar Check")
    
    # Add info about API key
    if not os.getenv('GOOGLE_API_KEY'):
        st.info("💡 **Tip**: Add your Google API key to the .env file for enhanced grammar checking using Gemini AI. Get free API key from https://makersuite.google.com/app/apikey")
    
    text = st.text_area("Enter text to check for grammar:", height=200, key="grammar_text")
    
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
            
            # Download button
            st.download_button(
                label="Download Corrected Text as TXT",
                data=result['corrected_text'],
                file_name="corrected_text.txt",
                mime="text/plain"
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
    threshold = st.slider("Plagiarism threshold (%):", min_value=10, max_value=100, value=70, step=1)
    text = None
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8")
    elif input_text:
        text = input_text
    if st.button("Check Plagiarism"):
        if text:
            result = check_plagiarism(text, threshold=threshold/100)
            st.subheader("Plagiarism Report:")
            st.write(f"Plagiarism Score: {result['plagiarism_score']}%")
            st.write(result['message'])
            if result['similar_sentences']:
                st.write("Similar Sentences:")
                for sent in result['similar_sentences']:
                    st.write(f"- {sent['input_sentence']} (Similarity: {sent['similarity']:.2f})")
            # Download report
            report = f"Plagiarism Score: {result['plagiarism_score']}%\n{result['message']}\n"
            if result['similar_sentences']:
                report += "Similar Sentences:\n" + '\n'.join([f"- {s['input_sentence']} (Similarity: {s['similarity']:.2f})" for s in result['similar_sentences']])
            st.download_button(
                label="Download Report as TXT",
                data=report,
                file_name="plagiarism_report.txt",
                mime="text/plain"
            )
        else:
            st.warning("Please upload a file or enter text to check.")