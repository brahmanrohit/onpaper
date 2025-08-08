import docx
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from typing import Dict, List, Any
from .gemini_helper import generate_text
import io

class DefaultPaperGenerator:
    """Generate default research papers in Word format with balanced structure."""
    
    def __init__(self):
        self.section_word_limits = {
            "Abstract": "150-200",
            "Introduction": "200-250", 
            "Literature Review": "200-250",
            "Methodology": "150-200",
            "Results": "200-250",
            "Discussion": "200-250",
            "Conclusion": "150-200"
        }
    
    def generate_default_paper(self, topic: str, paper_type: str = "empirical", suggestions: dict = None, images: list = None) -> Dict[str, Any]:
        """Generate a balanced, type-specific research paper with advanced realism, academic polish, user suggestions, and images."""
        # Section structures for each type
        type_structures = {
            "empirical": ["Abstract", "Introduction", "Literature Review", "Methodology", "Results", "Discussion", "Conclusion"],
            "theoretical": ["Abstract", "Introduction", "Theoretical Framework", "Literature Review", "Analysis", "Discussion", "Conclusion"],
            "review": ["Abstract", "Introduction", "Methodology", "Literature Review", "Analysis", "Discussion", "Conclusion"],
            "comparative": ["Abstract", "Introduction", "Literature Review", "Methodology", "Comparative Analysis", "Discussion", "Conclusion"],
            "case_study": ["Abstract", "Introduction", "Case Background", "Methodology", "Case Analysis", "Discussion", "Conclusion"],
            "analytical": ["Abstract", "Introduction", "Literature Review", "Analytical Framework", "Analysis", "Discussion", "Conclusion"],
            "methodological": ["Abstract", "Introduction", "Literature Review", "Methodology Development", "Validation", "Discussion", "Conclusion"],
            "position": ["Abstract", "Introduction", "Background", "Position Statement", "Supporting Arguments", "Discussion", "Conclusion"],
            "technical": ["Abstract", "Introduction", "Technical Background", "Methods", "Results", "Technical Analysis", "Conclusion"],
            "interdisciplinary": ["Abstract", "Introduction", "Theoretical Integration", "Methodology", "Cross-Disciplinary Analysis", "Discussion", "Conclusion"]
        }
        # Section word limits (default fallback)
        default_word_limits = {
            "Abstract": "150-200", "Introduction": "200-250", "Literature Review": "200-250", "Methodology": "150-200", "Results": "200-250", "Discussion": "200-250", "Conclusion": "150-200",
            "Theoretical Framework": "200-250", "Analysis": "200-250", "Case Background": "200-250", "Case Analysis": "200-250", "Comparative Analysis": "200-250", "Analytical Framework": "200-250", "Methodology Development": "200-250", "Validation": "200-250", "Background": "200-250", "Position Statement": "200-250", "Supporting Arguments": "200-250", "Technical Background": "200-250", "Methods": "200-250", "Technical Analysis": "200-250", "Theoretical Integration": "200-250", "Cross-Disciplinary Analysis": "200-250"
        }
        # Generate shared variables for consistency
        hypothesis = generate_text(f"Formulate a clear, testable hypothesis for a {paper_type} research paper on: {topic}. Format as a single, formal sentence.").strip()
        sample_size = generate_text(f"Suggest a realistic sample size for a {paper_type} research paper on: {topic}. Format as a number or short phrase.").strip()
        key_stat = generate_text(f"Suggest a key statistical result or finding for a {paper_type} research paper on: {topic}, using the sample size '{sample_size}'. Format as a short, formal sentence.").strip()
        advanced_method = generate_text(f"Name one advanced method or tool (e.g., bibliometric analysis, network analysis, sentiment analysis) relevant for a {paper_type} research paper on: {topic}.").strip()
        extra_angle = generate_text(f"Suggest an extra dimension or angle (e.g., ESG aspects, policy implications, long-term trends) to broaden the scope of a {paper_type} research paper on: {topic}.").strip()
        # Use type-specific structure
        structure = type_structures.get(paper_type, type_structures["empirical"])
        # Generate paper content
        paper = {
            "topic": topic,
            "paper_type": paper_type,
            "title": self._generate_title(topic),
            "sections": {},
            "citations": self._generate_default_citations(topic, advanced_method, paper_type),
            "word_count": 0,
            "hypothesis": hypothesis,
            "sample_size": sample_size,
            "key_stat": key_stat,
            "advanced_method": advanced_method,
            "extra_angle": extra_angle,
            "images": images or []
        }
        for section_name in structure:
            word_limit = default_word_limits.get(section_name, "200-250")
            suggestion_text = ""
            if suggestions:
                # If suggestions is a dict, check for section-specific suggestions
                if isinstance(suggestions, dict) and section_name in suggestions:
                    suggestion_text = suggestions[section_name]
                # If suggestions is a string, use for all sections
                elif isinstance(suggestions, str):
                    suggestion_text = suggestions
            content = self._generate_section(
                section_name, topic, word_limit, hypothesis, sample_size, key_stat, advanced_method, extra_angle, paper_type, suggestion_text
            )
            paper["sections"][section_name] = {
                "content": content,
                "word_count": len(content.split()),
                "word_limit": word_limit
            }
            paper["word_count"] += len(content.split())
        return paper
    
    def _generate_title(self, topic: str) -> str:
        """Generate a clear, professional title."""
        title_prompt = f"""
        Create a clear, professional research paper title for: {topic}
        
        Requirements:
        - Short and descriptive (under 15 words)
        - Include key keywords
        - Professional academic style
        - Avoid jargon when possible
        
        Format: Main topic - Subtitle or focus area
        """
        
        try:
            return generate_text(title_prompt).strip()
        except Exception as e:
            return f"Research on {topic}"
    
    def _generate_section(self, section_name: str, topic: str, word_limit: str,
                         hypothesis: str, sample_size: str, key_stat: str, advanced_method: str, extra_angle: str, paper_type: str = "empirical", suggestion_text: str = "") -> str:
        """Generate a specific section with advanced realism, academic polish, and user suggestions."""
        # Type-specific prompt logic
        prompt = ""
        if section_name == "Abstract":
            prompt = f"""
            Write a formal, clear abstract for a {paper_type} research paper on: {topic}.
            Hypothesis: {hypothesis}
            Sample Size: {sample_size}
            Key Stat: {key_stat}
            Advanced Method: {advanced_method}
            Extra Angle: {extra_angle}
            {suggestion_text}
            Structure ({word_limit} words):
            1. Purpose and scope
            2. State the hypothesis explicitly
            3. Methods (mention sample size and advanced method)
            4. Results (mention key stat and reference a visual, e.g., 'Figure 1')
            5. Conclusion and extra angle
            Use a formal, academic tone. No placeholders or incomplete lines.
            """
        elif section_name == "Introduction":
            prompt = f"""
            Write a formal introduction for a {paper_type} research paper on: {topic}.
            Hypothesis: {hypothesis}
            Extra Angle: {extra_angle}
            {suggestion_text}
            Structure ({word_limit} words):
            1. Topic importance and context
            2. State the research question and hypothesis
            3. Brief overview of approach (mention advanced method)
            4. Introduce extra angle (e.g., ESG, policy, long-term trends)
            Use a formal, academic tone. No placeholders or incomplete lines.
            """
        elif section_name == "Literature Review":
            prompt = f"""
            Write a formal literature review for a {paper_type} research paper on: {topic}.
            {suggestion_text}
            Structure ({word_limit} words):
            1. Recent and relevant studies (cite at least one 2023–2024 source)
            2. Key findings and gaps
            3. How this research fits in
            Use a formal, academic tone. No placeholders or incomplete lines.
            """
        elif section_name in ["Methodology", "Methods", "Methodology Development"]:
            prompt = f"""
            Write a formal {section_name} section for a {paper_type} research paper on: {topic}.
            Sample Size: {sample_size}
            Advanced Method: {advanced_method}
            {suggestion_text}
            Structure ({word_limit} words):
            1. Data collection and analysis (mention sample size)
            2. Describe the advanced method/tool used
            3. Ensure consistency with Abstract and Results
            Use a formal, academic tone. No placeholders or incomplete lines.
            """
        elif section_name in ["Results", "Validation", "Analysis", "Comparative Analysis", "Case Analysis", "Technical Analysis", "Cross-Disciplinary Analysis"]:
            prompt = f"""
            Write a formal {section_name} section for a {paper_type} research paper on: {topic}.
            Sample Size: {sample_size}
            Key Stat: {key_stat}
            Advanced Method: {advanced_method}
            {suggestion_text}
            Structure ({word_limit} words):
            1. Present main findings (use sample size and key stat)
            2. Reference at least one visual/table (e.g., 'Figure 1 shows...')
            3. Highlight findings from advanced method
            Use a formal, academic tone. No placeholders or incomplete lines.
            """
        elif section_name in ["Discussion"]:
            prompt = f"""
            Write a formal discussion section for a {paper_type} research paper on: {topic}.
            Key Stat: {key_stat}
            Extra Angle: {extra_angle}
            {suggestion_text}
            Structure ({word_limit} words):
            1. Interpret results (reference key stat)
            2. Compare to previous research
            3. Discuss implications, including extra angle (e.g., ESG, policy, long-term trends)
            4. Mention at least one limitation
            Use a formal, academic tone. No placeholders or incomplete lines.
            """
        elif section_name in ["Conclusion"]:
            prompt = f"""
            Write a formal conclusion for a {paper_type} research paper on: {topic}.
            Hypothesis: {hypothesis}
            Extra Angle: {extra_angle}
            {suggestion_text}
            Structure ({word_limit} words):
            1. Summarize main findings and hypothesis
            2. Key contributions
            3. Future work and extra angle
            Use a formal, academic tone. No placeholders or incomplete lines.
            """
        else:
            prompt = f"Write a {section_name} section for a {paper_type} research paper on: {topic} ({word_limit} words). {suggestion_text} Use a formal, academic tone."
        try:
            return generate_text(prompt).strip()
        except Exception as e:
            return f"This is the {section_name} section for a {paper_type} research paper on {topic}. Content will be generated here."
    
    def _generate_default_citations(self, topic: str, advanced_method: str, paper_type: str = "empirical") -> List[Dict]:
        """Generate 6-8 relevant academic citations, including 2–3 real, recent (2023–2024) references with DOIs."""
        citation_prompt = f"""
        List 6-8 relevant academic sources for a {paper_type} research paper on: {topic}.
        Requirements:
        - At least 2–3 real, recent (2023–2024) journal articles with DOIs
        - 1–2 classic/important papers
        - 1–2 books or reports if relevant
        - At least one source related to {advanced_method}
        Format each as:
        Author(s). (Year). Title. Journal/Publisher. DOI:xxxxx
        Keep it simple, relevant, and formal.
        """
        try:
            citations_text = generate_text(citation_prompt)
            citations = []
            lines = citations_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 20 and not line.startswith('Include:'):
                    citations.append({
                        "citation": line,
                        "type": "journal" if "Journal" in line else "book" if "Book" in line else "other"
                    })
            return citations[:8]
        except Exception as e:
            return [
                {"citation": f"Smith, J. (2024). Advances in {paper_type.title()} Research Methods. Journal of Research. DOI:10.1234/example1", "type": "journal"},
                {"citation": f"Johnson, A. (2023). Understanding Academic Writing in {paper_type.title()}. Academic Press. DOI:10.1234/example2", "type": "book"},
                {"citation": f"Brown, M. (2024). Current Trends in {paper_type.title()} Research. Research Quarterly. DOI:10.1234/example3", "type": "journal"}
            ]
    
    def create_word_document(self, paper: Dict[str, Any]) -> bytes:
        """Create a Word document (.docx) from the paper, inserting images into the correct sections."""
        try:
            doc = docx.Document()
            
            # Set margins
            sections = doc.sections
            for section in sections:
                section.top_margin = Inches(1)
                section.bottom_margin = Inches(1)
                section.left_margin = Inches(1)
                section.right_margin = Inches(1)
            
            # Helper: get images for a section
            def get_section_images(section):
                return [img for img in paper.get('images', []) if img.get('section') == section]
            
            # Title
            title = doc.add_heading(paper['title'], 0)
            title.alignment = WD_ALIGN_PARAGRAPH.CENTER
            doc.add_paragraph()
            
            # Section order (fallback)
            section_order = [
                "Abstract", "Introduction", "Literature Review", "Methodology", "Results", "Discussion", "Conclusion"
            ]
            
            # Add all sections (with images)
            for section in section_order:
                if section in paper['sections']:
                    doc.add_heading(section, level=1)
                    doc.add_paragraph(paper['sections'][section]['content'])
                    # Insert images for this section
                    for img in get_section_images(section):
                        try:
                            doc.add_picture(io.BytesIO(img['image_bytes']), width=Inches(4.5))
                            if img.get('caption'):
                                last_paragraph = doc.paragraphs[-1]
                                last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                                doc.add_paragraph(img['caption']).alignment = WD_ALIGN_PARAGRAPH.CENTER
                        except Exception as e:
                            doc.add_paragraph("[Image could not be loaded]")
                    doc.add_paragraph()
            
            # References
            doc.add_heading('References', level=1)
            for i, citation in enumerate(paper['citations'], 1):
                doc.add_paragraph(f"{i}. {citation['citation']}")
            
            # Save to bytes
            doc_bytes = io.BytesIO()
            doc.save(doc_bytes)
            doc_bytes.seek(0)
            
            return doc_bytes.getvalue()
            
        except Exception as e:
            # Return a simple text document if Word creation fails
            simple_doc = f"""
            {paper['title']}
            
            Abstract
            {paper['sections']['Abstract']['content']}
            
            Introduction
            {paper['sections']['Introduction']['content']}
            
            Literature Review
            {paper['sections']['Literature Review']['content']}
            
            Methodology
            {paper['sections']['Methodology']['content']}
            
            Results
            {paper['sections']['Results']['content']}
            
            Discussion
            {paper['sections']['Discussion']['content']}
            
            Conclusion
            {paper['sections']['Conclusion']['content']}
            
            References
            {chr(10).join([f"{i+1}. {citation['citation']}" for i, citation in enumerate(paper['citations'])])}
            """
            return simple_doc.encode('utf-8')

# Global instance
default_paper_generator = DefaultPaperGenerator()

def generate_default_paper(topic: str, paper_type: str = "empirical", suggestions: dict = None, images: list = None) -> Dict[str, Any]:
    """Generate a default research paper."""
    return default_paper_generator.generate_default_paper(topic, paper_type, suggestions, images)

def create_word_document(paper: Dict[str, Any]) -> bytes:
    """Create Word document from paper."""
    return default_paper_generator.create_word_document(paper) 