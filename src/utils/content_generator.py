import re
from typing import Dict, List
from .gemini_helper import generate_text, is_unavailable_response
from .reference_finder import search_references, format_reference

# Rules appended to EVERY section prompt to stop the model from inventing
# data. This is the core fix for fabricated statistics / figures / citations.
GROUND_RULES = (
    "STRICT RULES (must follow):\n"
    "- Use only accurate, well-established, real information about the topic.\n"
    "- Do NOT invent statistics, percentages, sample sizes, dates, study results, "
    "figures, or tables. If a precise figure is unknown, describe it qualitatively.\n"
    "- Never write 'Figure 1', 'Table 1', '85% of sources', or reference visuals/data "
    "that do not exist.\n"
    "- Do NOT fabricate citations, author names, journals, or DOIs. Refer to prior work "
    "only in general terms (e.g., 'prior studies', 'published records').\n"
    "- If the topic is a real person, place, event, or concept, use genuine facts.\n"
    "- Formal academic tone, complete sentences, no repetition, no placeholders, no filler."
)

# What each section should actually contain — covers every section name used
# across all 10 paper-type templates, so each type renders its proper structure.
SECTION_GUIDANCE = {
    "Abstract": "A self-contained summary: the topic and purpose, the central question or thesis, the approach taken, the main points/findings, and the key takeaway. No citations.",
    "Introduction": "Introduce the topic and why it matters, give the needed background/context, state the central question or thesis, and outline how the paper proceeds.",
    "Literature Review": "Summarize what existing published work establishes about the topic — major themes, well-known findings, and open questions or gaps. Refer to prior work in general terms only.",
    "Theoretical Framework": "Explain the key theories, concepts, or models relevant to the topic and how they frame the analysis.",
    "Theoretical Integration": "Integrate concepts and theories from the relevant disciplines and explain how, combined, they illuminate the topic.",
    "Methodology": "Describe the approach used to investigate or synthesize the topic — the kinds of sources, evidence, or reasoning the paper draws on and how it is organized. If this is not an original experiment, describe the review/analytical approach honestly; do NOT invent a sample size or experimental setup.",
    "Methods": "Describe the approach and procedure used. If no original experiment was run, describe how the topic is examined from existing evidence; do NOT invent a sample size or apparatus.",
    "Methodology Development": "Explain the proposed method or procedure: its rationale, design, and steps. Keep it grounded; do not invent results.",
    "Background": "Provide the factual context the reader needs: history, setting, key facts, and relevant prior developments about the topic.",
    "Case Background": "Give the factual background of the specific case: who/what/when/where, the context, and the relevant circumstances.",
    "Technical Background": "Explain the technical context and prerequisites needed to understand the topic: relevant systems, concepts, and prior approaches.",
    "Results": "Present the main findings or key established facts about the topic in an organized way. Use real, verifiable information; if exact figures are unknown, describe them qualitatively. Do not invent numbers or reference non-existent figures/tables.",
    "Analysis": "Critically examine the topic: interpret the key facts and evidence, identify patterns, strengths, weaknesses, and implications, using clear logical reasoning.",
    "Analytical Framework": "Lay out the criteria, lens, or framework used to analyze the topic and justify the choice.",
    "Comparative Analysis": "Compare and contrast the relevant subjects/options across clearly defined dimensions, noting similarities, differences, and what they imply.",
    "Case Analysis": "Analyze the specific case in depth: what happened and why, the contributing factors, and the lessons that can be drawn.",
    "Technical Analysis": "Analyze the technical aspects: how it works, design choices, trade-offs, and practical considerations.",
    "Cross-Disciplinary Analysis": "Analyze the topic through multiple disciplinary lenses and synthesize the cross-disciplinary insights.",
    "Validation": "Explain how the approach or findings are assessed for soundness — reasoning, consistency checks, or comparison with established knowledge. Do not invent benchmark numbers.",
    "Position Statement": "State the paper's clear position or argument on the issue in a focused, unambiguous way.",
    "Supporting Arguments": "Present well-reasoned arguments and real evidence supporting the position, and briefly address the main counterarguments.",
    "Discussion": "Interpret what the key points mean, connect them to the broader context and prior work, discuss implications, and note honest limitations.",
    "Conclusion": "Summarize the main points and the answer to the central question, state the key contribution/takeaway, and suggest directions for further work. Introduce no new facts.",
}

_DEFAULT_GUIDANCE = "Write a focused, factual treatment of this section appropriate to its name and the paper type."


def section_guidance(section_name: str) -> str:
    """Return the writing guidance for a section name (with a sensible default)."""
    return SECTION_GUIDANCE.get(section_name, _DEFAULT_GUIDANCE)


def _high_word_bound(word_limit: str, default: int = 250) -> int:
    """Parse the upper bound from a '150-250' style limit, clamped to a sane range."""
    try:
        high = int(str(word_limit).split("-")[-1])
    except (ValueError, AttributeError):
        high = default
    return max(120, min(high, 350))


def trim_to_sentence(text: str, max_words: int) -> str:
    """Trim text to at most max_words, cutting only at a sentence boundary.

    Replaces the old hard ' '.join(words[:250]) slice that chopped mid-sentence.
    """
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out, count = [], 0
    for s in sentences:
        n = len(s.split())
        if out and count + n > max_words:
            break
        out.append(s)
        count += n
    result = " ".join(out).strip() if out else " ".join(words[:max_words]).strip()
    # Hard cap: sentence-aware trimming leaves the first sentence whole, and
    # bullet/checklist output has no '.' boundaries — so enforce the cap at a
    # word boundary as a backstop (never breaks mid-word).
    if len(result.split()) > max_words:
        result = " ".join(result.split()[:max_words])
    return result


def strip_section_heading(text: str, section_name: str) -> str:
    """Remove a leading echoed heading line that just repeats the section name.

    Models often start a section with its own title (e.g. "Introduction\\n\\n...");
    the UI and .docx already render the heading, so this avoids a duplicate.
    Only strips when the first line EXACTLY matches the section name (allowing
    leading '#' and a trailing ':'), so real content is never removed.
    """
    if not text or not section_name:
        return (text or "").strip()
    lines = text.lstrip().split("\n")
    if lines:
        first = lines[0].strip().lstrip("#").strip().rstrip(":").strip()
        if first.lower() == section_name.strip().lower():
            rest = lines[1:]
            while rest and not rest[0].strip():
                rest.pop(0)
            return "\n".join(rest).strip()
    return text.strip()


def _crossref_citations(topic: str, rows: int = 8) -> List[Dict]:
    """Fetch REAL references for a topic from CrossRef. Never fabricates.

    Returns a list of {"citation", "type", "doi"}. If CrossRef is unreachable
    or returns nothing, returns a single honest note instead of fake sources.
    """
    try:
        result = search_references(topic, rows=rows)
    except Exception as e:
        result = {"references": [], "error": str(e)}
    refs = result.get("references", []) if isinstance(result, dict) else []
    citations = []
    for ref in refs:
        formatted = format_reference(ref, "APA")
        if formatted and len(formatted) > 10:
            citations.append({
                "citation": formatted,
                "type": (ref.get("type") or "journal"),
                "doi": ref.get("doi", ""),
            })
    if not citations:
        citations.append({
            "citation": ("(No verified references found via CrossRef. Connect to the internet, "
                         "or use the Reference Finder tab to find real sources for this topic.)"),
            "type": "note",
            "doi": "",
        })
    return citations


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
                              include_citations: bool = True, include_data: bool = False,
                              target_words: int = None) -> Dict:
        """Generate a type-specific research paper following the perfect rules, with realistic upgrades."""
        
        # Validate paper type
        if paper_type not in self.research_templates:
            paper_type = "empirical"
        
        template = self.research_templates[paper_type]

        # A single, grounded thesis/central question — NOT a fabricated hypothesis
        # with invented variables. (Replaces the old sample_size/key_stat scaffolding.)
        thesis = generate_text(
            f'In one clear, factual sentence, state the central question or thesis that a '
            f'{template["name"]} on "{topic}" would address. Be specific. Do not invent statistics.'
        ).strip()

        # Generate type-specific content
        paper = {
            "topic": topic,
            "paper_type": paper_type,
            "paper_name": template["name"],
            "sections": {},
            "citations": [],
            "word_count": 0,
            "focus": template["focus"],
            "thesis": thesis,
        }

        # Generate each section with type-specific, fabrication-free prompts
        for section in template["structure"]:
            if section not in ("Title", "References"):  # handled separately
                paper["sections"][section] = self._generate_section(
                    section, topic, template, thesis, target_words=target_words)
                paper["word_count"] += paper["sections"][section]["word_count"]

        # Add title
        paper["sections"]["Title"] = self._generate_title(topic, paper_type)

        # Real citations from CrossRef (never fabricated)
        if include_citations:
            paper["citations"] = self._generate_simple_citations(topic, paper_type)

        return paper
    
    def _generate_title(self, topic: str, paper_type: str, instructions: str = "") -> str:
        """Generate a type-specific title."""
        template = self.research_templates[paper_type]
        instr = (f"        - Take this user focus into account: {instructions}\n"
                 if instructions and instructions.strip() else "")

        title_prompt = f"""
        Create a clear, concise research paper title for a {template['name']} on: {topic}

        Requirements:
        - Short and clear (under 15 words)
        - Include key keywords
        - Reflect the {paper_type} type focus: {template['focus']}
        - Professional but not overly academic
        - Avoid jargon when possible
{instr}
        Format: Main topic - Subtitle or focus area
        """

        return generate_text(title_prompt).strip()
    
    def _generate_section(self, section_name: str, topic: str, template: Dict,
                          thesis: str = None, target_words: int = None, instructions: str = "") -> Dict:
        """Generate a section with type-appropriate, fabrication-free prompting."""
        word_limit = template["word_limits"].get(section_name, "400-600")
        paper_type = template["name"]
        focus = template["focus"]
        thesis_line = f'Central question / thesis: {thesis}\n' if thesis else ""
        instr_line = (f'Additional focus / instructions: {instructions}\n'
                      if instructions and instructions.strip() else "")

        # An explicit per-section word count (e.g. the Quick generator slider)
        # overrides the template default; the Abstract stays short by convention.
        if target_words:
            max_words = max(120, min(int(target_words), 1500))
            if section_name == "Abstract":
                max_words = min(max_words, 300)
            word_limit = f"{int(max_words * 0.8)}-{max_words}"
            length_str = f"about {max_words} words"
        else:
            max_words = _high_word_bound(word_limit)
            # Reflect the actual achievable cap in the advertised range so the
            # per-section word badge isn't always "under" for large templates.
            word_limit = f"{int(max_words * 0.8)}-{max_words}"
            length_str = f"about {max_words} words"

        prompt = (
            f'Write the "{section_name}" section of a {paper_type} on the topic: "{topic}".\n'
            f'Paper focus: {focus}\n'
            f'{thesis_line}{instr_line}'
            f'\nWhat this section should contain:\n{section_guidance(section_name)}\n'
            f'\nTarget length: {length_str}.\n\n'
            f'{GROUND_RULES}'
        )

        content = generate_text(prompt)

        # Expand only if genuinely too thin (kept factual, non-repetitive).
        # Use is_unavailable_response so a backend-down message (which does NOT
        # start with "Error") doesn't trigger a wasted retry or get concatenated.
        if len(content.split()) < 80 and not is_unavailable_response(content):
            extra = generate_text(
                f'Expand the following {section_name} on "{topic}" to be more complete and '
                f'specific, staying factual and non-repetitive. {GROUND_RULES}\n\n{content}'
            )
            if not is_unavailable_response(extra):
                content = content + "\n" + extra

        content = strip_section_heading(content, section_name)
        content = trim_to_sentence(content, max_words)
        return {
            "content": content.strip(),
            "word_count": len(content.split()),
            "word_limit": word_limit,
        }

    def _generate_simple_citations(self, topic: str, paper_type: str) -> List[Dict]:
        """Return REAL references from CrossRef (never fabricated)."""
        return _crossref_citations(topic)

    def generate_section_only(self, topic: str, section: str, paper_type: str = "empirical",
                              instructions: str = "") -> str:
        """Generate a specific section for a paper type."""
        template = self.research_templates.get(paper_type, self.research_templates["empirical"])

        if section == "Title":
            return self._generate_title(topic, paper_type, instructions)
        else:
            section_data = self._generate_section(section, topic, template, instructions=instructions)
            return section_data["content"]

    def edit_section_text(self, text: str, section: str, paper_type: str = "empirical",
                          instructions: str = "") -> str:
        """Revise/improve an existing section's text (the 'Editor' behavior)."""
        if not text or not text.strip():
            return "Please paste some text to edit."
        template = self.research_templates.get(paper_type, self.research_templates["empirical"])
        instr = (f'Specific instructions from the user: {instructions}\n'
                 if instructions and instructions.strip() else "")
        prompt = (
            f'You are editing the "{section}" section of a {template["name"]}.\n'
            f'Improve grammar, clarity, flow, structure, and academic tone, and strengthen the '
            f'writing, while preserving the original meaning and any real facts. {instr}'
            f'Return ONLY the revised section text, with no preamble or commentary.\n\n'
            f'{GROUND_RULES}\n\nSECTION TO EDIT:\n{text}'
        )
        return generate_text(prompt).strip()

    def generate_section_guide(self, topic: str, section: str, paper_type: str = "empirical",
                               target_words: int = 400, instructions: str = "") -> str:
        """Write how-to GUIDANCE for a section (not the section content itself)."""
        template = self.research_templates.get(paper_type, self.research_templates["empirical"])
        instr = (f'Also address this focus from the user: {instructions}\n'
                 if instructions and instructions.strip() else "")
        prompt = (
            f'Write a practical how-to GUIDE (advice, NOT the section text itself) for writing the '
            f'"{section}" section of a {template["name"]} on the topic: "{topic}".\n'
            f'Cover: (1) the purpose of this section, (2) what to include and in what order, '
            f'(3) concrete tips specific to this topic, and (4) common mistakes to avoid. '
            f'Use a short bulleted checklist where helpful.\n{instr}'
            f'Target length: about {target_words} words.\n\n{GROUND_RULES}'
        )
        content = generate_text(prompt)
        cap = max(200, min(int(target_words) + 200, 1300))
        return trim_to_sentence(content, cap).strip()

    def generate_quick_paper(self, topic: str, paper_type: str = "empirical",
                             target_words: int = None) -> Dict:
        """Generate a simple, quick research paper of specific type."""
        return self.generate_research_paper(
            topic, paper_type, include_citations=True, include_data=False, target_words=target_words)
    
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

# Lazy global instance — cheap to build (just templates) but kept lazy for
# consistency and zero import-time work. Reused across reruns as a module global.
_content_generator = None


def get_content_generator() -> "ResearchContentGenerator":
    """Return the shared content generator, constructing it on first use."""
    global _content_generator
    if _content_generator is None:
        _content_generator = ResearchContentGenerator()
    return _content_generator


def __getattr__(name):
    """Backward-compat: ``content_generator.content_generator`` still resolves (lazily)."""
    if name == "content_generator":
        return get_content_generator()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def generate_comprehensive_paper(topic: str, research_type: str = "empirical",
                               include_data: bool = False, include_citations: bool = True) -> Dict:
    """Main function to generate type-specific research paper."""
    return get_content_generator().generate_research_paper(topic, research_type, include_citations, include_data)

def generate_section_only(topic: str, section: str, research_type: str = "empirical",
                          instructions: str = "") -> str:
    """Generate a specific section for a paper type."""
    return get_content_generator().generate_section_only(topic, section, research_type, instructions)

def edit_section(text: str, section: str, research_type: str = "empirical",
                 instructions: str = "") -> str:
    """Revise/improve an existing section's text."""
    return get_content_generator().edit_section_text(text, section, research_type, instructions)

def generate_section_guide(topic: str, section: str, research_type: str = "empirical",
                           target_words: int = 400, instructions: str = "") -> str:
    """Write how-to guidance for a section."""
    return get_content_generator().generate_section_guide(topic, section, research_type, target_words, instructions)

def generate_quick_paper(topic: str, research_type: str = "empirical",
                         target_words: int = None) -> Dict:
    """Generate a simple, quick research paper of specific type."""
    return get_content_generator().generate_quick_paper(topic, research_type, target_words)

def get_available_paper_types() -> List[Dict]:
    """Get list of available paper types."""
    return get_content_generator().get_available_types()