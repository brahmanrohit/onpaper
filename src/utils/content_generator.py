import json
import random
from typing import Dict, List, Any
from .gemini_helper import generate_text

class ResearchContentGenerator:
    """Balanced research paper content generator with type-specific customization."""
    
    def __init__(self):
        self.research_templates = self._load_research_templates()
        self.citation_sources = self._load_citation_sources()
    
    def _load_research_templates(self) -> Dict:
        """Load research paper templates for different types with customized structures."""
        return {
            "empirical": {
                "name": "Empirical Research Paper (Data-based)",
                "structure": [
                    "Title", "Abstract", "Introduction", "Literature Review", 
                    "Methodology", "Results", "Discussion", "Conclusion", "References"
                ],
                "word_limits": {
                    "Abstract": "150-250",
                    "Introduction": "500-800",
                    "Literature Review": "600-1000",
                    "Methodology": "600-800",
                    "Results": "800-1200",
                    "Discussion": "800-1200",
                    "Conclusion": "300-500"
                },
                "focus": "Data analysis, statistics, sample size, empirical findings"
            },
            "theoretical": {
                "name": "Theoretical Research Paper",
                "structure": [
                    "Title", "Abstract", "Introduction", "Theoretical Framework", 
                    "Literature Review", "Analysis", "Discussion", "Conclusion", "References"
                ],
                "word_limits": {
                    "Abstract": "150-250",
                    "Introduction": "500-800",
                    "Theoretical Framework": "800-1200",
                    "Literature Review": "600-1000",
                    "Analysis": "800-1200",
                    "Discussion": "600-800",
                    "Conclusion": "300-500"
                },
                "focus": "Theoretical concepts, frameworks, conceptual analysis"
            },
            "review": {
                "name": "Review Paper (Literature or Systematic Review)",
                "structure": [
                    "Title", "Abstract", "Introduction", "Methodology", 
                    "Literature Review", "Analysis", "Discussion", "Conclusion", "References"
                ],
                "word_limits": {
                    "Abstract": "150-250",
                    "Introduction": "400-600",
                    "Methodology": "400-600",
                    "Literature Review": "1200-1800",
                    "Analysis": "800-1200",
                    "Discussion": "600-800",
                    "Conclusion": "300-500"
                },
                "focus": "Past studies, research gaps, future directions"
            },
            "comparative": {
                "name": "Comparative Research Paper",
                "structure": [
                    "Title", "Abstract", "Introduction", "Literature Review", 
                    "Methodology", "Comparative Analysis", "Discussion", "Conclusion", "References"
                ],
                "word_limits": {
                    "Abstract": "150-250",
                    "Introduction": "500-800",
                    "Literature Review": "600-1000",
                    "Methodology": "400-600",
                    "Comparative Analysis": "1000-1400",
                    "Discussion": "600-800",
                    "Conclusion": "300-500"
                },
                "focus": "Comparison between subjects, contrast analysis, similarities/differences"
            },
            "case_study": {
                "name": "Case Study",
                "structure": [
                    "Title", "Abstract", "Introduction", "Case Background", 
                    "Methodology", "Case Analysis", "Discussion", "Conclusion", "References"
                ],
                "word_limits": {
                    "Abstract": "150-250",
                    "Introduction": "400-600",
                    "Case Background": "600-800",
                    "Methodology": "400-600",
                    "Case Analysis": "800-1200",
                    "Discussion": "600-800",
                    "Conclusion": "300-500"
                },
                "focus": "Deep dive on specific subject/event, context, unique factors"
            },
            "analytical": {
                "name": "Analytical Research Paper",
                "structure": [
                    "Title", "Abstract", "Introduction", "Literature Review", 
                    "Analytical Framework", "Analysis", "Discussion", "Conclusion", "References"
                ],
                "word_limits": {
                    "Abstract": "150-250",
                    "Introduction": "500-800",
                    "Literature Review": "600-1000",
                    "Analytical Framework": "600-800",
                    "Analysis": "800-1200",
                    "Discussion": "600-800",
                    "Conclusion": "300-500"
                },
                "focus": "Critical analysis, logical reasoning, systematic examination"
            },
            "methodological": {
                "name": "Methodological Research Paper",
                "structure": [
                    "Title", "Abstract", "Introduction", "Literature Review", 
                    "Methodology Development", "Validation", "Discussion", "Conclusion", "References"
                ],
                "word_limits": {
                    "Abstract": "150-250",
                    "Introduction": "400-600",
                    "Literature Review": "600-1000",
                    "Methodology Development": "800-1200",
                    "Validation": "600-800",
                    "Discussion": "600-800",
                    "Conclusion": "300-500"
                },
                "focus": "Method development, validation, procedural innovation"
            },
            "position": {
                "name": "Position Paper / Opinion Paper",
                "structure": [
                    "Title", "Abstract", "Introduction", "Background", 
                    "Position Statement", "Supporting Arguments", "Discussion", "Conclusion", "References"
                ],
                "word_limits": {
                    "Abstract": "150-250",
                    "Introduction": "400-600",
                    "Background": "600-800",
                    "Position Statement": "400-600",
                    "Supporting Arguments": "800-1200",
                    "Discussion": "600-800",
                    "Conclusion": "300-500"
                },
                "focus": "Clear position, supporting arguments, persuasive reasoning"
            },
            "technical": {
                "name": "Technical Report",
                "structure": [
                    "Title", "Abstract", "Introduction", "Technical Background", 
                    "Methods", "Results", "Technical Analysis", "Conclusion", "References"
                ],
                "word_limits": {
                    "Abstract": "150-250",
                    "Introduction": "400-600",
                    "Technical Background": "600-800",
                    "Methods": "400-600",
                    "Results": "600-800",
                    "Technical Analysis": "800-1200",
                    "Conclusion": "300-500"
                },
                "focus": "Technical details, practical implementation, technical specifications"
            },
            "interdisciplinary": {
                "name": "Interdisciplinary Research Paper",
                "structure": [
                    "Title", "Abstract", "Introduction", "Theoretical Integration", 
                    "Methodology", "Cross-Disciplinary Analysis", "Discussion", "Conclusion", "References"
                ],
                "word_limits": {
                    "Abstract": "150-250",
                    "Introduction": "500-800",
                    "Theoretical Integration": "800-1200",
                    "Methodology": "600-800",
                    "Cross-Disciplinary Analysis": "1000-1400",
                    "Discussion": "600-800",
                    "Conclusion": "300-500"
                },
                "focus": "Multiple disciplines, integration, cross-disciplinary insights"
            }
        }
    
    def _load_citation_sources(self) -> Dict:
        """Load essential citation sources."""
        return {
            "academic_databases": [
                "Google Scholar", "PubMed", "IEEE Xplore", "ScienceDirect",
                "JSTOR", "Web of Science"
            ],
            "citation_formats": {
                "APA": "American Psychological Association",
                "IEEE": "Institute of Electrical and Electronics Engineers",
                "MLA": "Modern Language Association"
            }
        }
    
    def generate_research_paper(self, topic: str, paper_type: str = "empirical", 
                              include_citations: bool = True, include_data: bool = False) -> Dict:
        """Generate a type-specific research paper following the perfect rules, with realistic upgrades."""
        
        # Validate paper type
        if paper_type not in self.research_templates:
            paper_type = "empirical"
        
        template = self.research_templates[paper_type]
        
        # Generate unique research question/hypothesis and consistent data points
        research_question = generate_text(f"Generate a unique, specific research question or hypothesis for a {template['name']} on: {topic}. Format as a single clear sentence.").strip()
        sample_size = generate_text(f"Suggest a realistic sample size or data point for a {template['name']} on: {topic}. Format as a number or short phrase.").strip()
        key_stat = generate_text(f"Suggest a key statistical result or finding for a {template['name']} on: {topic}, using the sample size '{sample_size}'. Format as a short sentence.").strip()
        
        # Generate type-specific content
        paper = {
            "topic": topic,
            "paper_type": paper_type,
            "paper_name": template["name"],
            "sections": {},
            "citations": [],
            "word_count": 0,
            "focus": template["focus"],
            "research_question": research_question,
            "sample_size": sample_size,
            "key_stat": key_stat
        }
        
        # Generate each section with type-specific prompts
        for section in template["structure"]:
            if section != "Title":  # Title is handled separately
                paper["sections"][section] = self._generate_section(
                    section, topic, template, include_citations, include_data,
                    research_question, sample_size, key_stat
                )
                paper["word_count"] += paper["sections"][section]["word_count"]
        
        # Add title
        paper["sections"]["Title"] = self._generate_title(topic, paper_type)
        
        # Add citations if requested
        if include_citations:
            paper["citations"] = self._generate_simple_citations(topic, paper_type)
        
        return paper
    
    def _generate_title(self, topic: str, paper_type: str) -> str:
        """Generate a type-specific title."""
        template = self.research_templates[paper_type]
        
        title_prompt = f"""
        Create a clear, concise research paper title for a {template['name']} on: {topic}
        
        Requirements:
        - Short and clear (under 15 words)
        - Include key keywords
        - Reflect the {paper_type} type focus: {template['focus']}
        - Professional but not overly academic
        - Avoid jargon when possible
        
        Format: Main topic - Subtitle or focus area
        """
        
        return generate_text(title_prompt).strip()
    
    def _generate_section(self, section_name: str, topic: str, template: Dict, 
                         include_citations: bool, include_data: bool,
                         research_question: str = None, sample_size: str = None, key_stat: str = None) -> Dict:
        """Generate a type-specific section following the perfect research paper rules, with realistic upgrades."""
        
        word_limit = template["word_limits"].get(section_name, "400-600")
        paper_type = template["name"]
        focus = template["focus"]
        
        # Section-specific prompt upgrades
        if section_name == "Abstract":
            prompt = f"""
            Write a clear abstract for a {paper_type} on: {topic}
            Focus: {focus}
            Research Question: {research_question}
            Sample Size: {sample_size}
            Key Stat: {key_stat}
            Follow this structure (150-250 words):
            1. Purpose: What was studied?
            2. Research Question/Hypothesis: {research_question}
            3. Methods: How was it done? (mention sample size: {sample_size})
            4. Results: What was found? (mention key stat: {key_stat})
            5. Conclusion: What does it mean?
            No placeholders. No incomplete lines. No filler.
            """
        elif section_name == "Introduction":
            prompt = f"""
            Write an introduction for a {paper_type} on: {topic}
            Focus: {focus}
            Research Question: {research_question}
            Include (150-250 words):
            1. What is the topic?
            2. What question are you answering? (state: {research_question})
            3. Why is this important?
            4. Briefly mention sample size and key stat for context.
            No placeholders. No incomplete lines. No filler.
            """
        elif section_name == "Results":
            prompt = f"""
            Write a results section for a {paper_type} on: {topic}
            Focus: {focus}
            Sample Size: {sample_size}
            Key Stat: {key_stat}
            Include (150-250 words):
            1. Present the main findings clearly (use sample size: {sample_size}, key stat: {key_stat})
            2. Mention at least one visual (e.g., 'Figure 1 shows a scatterplot of ...')
            3. Highlight unique angles or findings.
            No placeholders. No incomplete lines. No filler.
            """
        elif section_name == "Discussion":
            prompt = f"""
            Write a discussion section for a {paper_type} on: {topic}
            Focus: {focus}
            Sample Size: {sample_size}
            Key Stat: {key_stat}
            Include (150-250 words):
            1. What do the results mean? (reference key stat: {key_stat})
            2. How do they compare to previous research?
            3. What are the implications?
            4. Mention 1-2 honest limitations (e.g., small sample, cross-sectional design).
            No placeholders. No incomplete lines. No filler.
            """
        elif section_name == "Conclusion":
            prompt = f"""
            Write a conclusion for a {paper_type} on: {topic}
            Focus: {focus}
            Research Question: {research_question}
            Include (80-150 words):
            1. Main takeaway
            2. Key contributions
            3. Future work suggestions
            No placeholders. No incomplete lines. No filler.
            """
        else:
            # Default for other sections
            prompt = f"""
            Write a {section_name} section for a {paper_type} on: {topic}
            Focus: {focus}
            Research Question: {research_question}
            Sample Size: {sample_size}
            Key Stat: {key_stat}
            Include (80-250 words):
            No placeholders. No incomplete lines. No filler.
            """
        
        content = generate_text(prompt)
        # Enforce word count limits (80-250 words)
        words = content.split()
        if len(words) > 250:
            content = ' '.join(words[:250])
        elif len(words) < 80:
            # If too short, ask for expansion
            content += '\n' + generate_text(f"Expand this section to at least 80 words, keeping it relevant and non-repetitive: {content}")
            content = ' '.join(content.split()[:250])
        
        return {
            "content": content.strip(),
            "word_count": len(content.split()),
            "word_limit": word_limit
        }
    
    def _generate_simple_citations(self, topic: str, paper_type: str) -> List[Dict]:
        """Generate type-specific citations (5-8 sources, at least 2-3 recent)."""
        
        template = self.research_templates[paper_type]
        
        citation_prompt = f"""
        List 6-8 relevant academic sources for a {template['name']} on: {topic}
        Focus: {template['focus']}
        Requirements:
        - At least 2-3 recent journal articles (last 3 years)
        - 1-2 classic/important papers
        - 1-2 books or reports if relevant
        Format each as:
        Author(s). (Year). Title. Journal/Publisher.
        Keep it simple and relevant to the {paper_type} type.
        """
        
        citations_text = generate_text(citation_prompt)
        # Simple parsing
        citations = []
        lines = citations_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 20 and not line.startswith('Include:'):
                citations.append({
                    "citation": line,
                    "type": "journal" if "Journal" in line else "book" if "Book" in line else "other"
                })
        return citations[:8]  # Limit to 8 citations
    
    def generate_section_only(self, topic: str, section: str, paper_type: str = "empirical") -> str:
        """Generate a specific section for a paper type."""
        template = self.research_templates.get(paper_type, self.research_templates["empirical"])
        
        if section == "Title":
            return self._generate_title(topic, paper_type)
        else:
            section_data = self._generate_section(section, topic, template, True, False)
            return section_data["content"]
    
    def generate_quick_paper(self, topic: str, paper_type: str = "empirical") -> Dict:
        """Generate a simple, quick research paper of specific type."""
        return self.generate_research_paper(topic, paper_type, include_citations=True, include_data=False)
    
    def get_available_types(self) -> List[Dict]:
        """Get list of available paper types with descriptions."""
        types = []
        for key, template in self.research_templates.items():
            types.append({
                "key": key,
                "name": template["name"],
                "focus": template["focus"],
                "sections": len(template["structure"])
            })
        return types

# Global instance
content_generator = ResearchContentGenerator()

def generate_comprehensive_paper(topic: str, research_type: str = "empirical", 
                               include_data: bool = False, include_citations: bool = True) -> Dict:
    """Main function to generate type-specific research paper."""
    return content_generator.generate_research_paper(topic, research_type, include_citations, include_data)

def generate_section_only(topic: str, section: str, research_type: str = "empirical") -> str:
    """Generate a specific section for a paper type."""
    return content_generator.generate_section_only(topic, section, research_type)

def generate_quick_paper(topic: str, research_type: str = "empirical") -> Dict:
    """Generate a simple, quick research paper of specific type."""
    return content_generator.generate_quick_paper(topic, research_type)

def get_available_paper_types() -> List[Dict]:
    """Get list of available paper types."""
    return content_generator.get_available_types() 