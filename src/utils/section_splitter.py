"""
Shared section splitter — turns a pasted/extracted paper into named sections by
detecting heading lines. Used by Citation Density, Section Completeness, and the
Abstract Scorecard. Fully offline (regex only).

Heading detection is heuristic: a short line (<= 8 words) that, after stripping
leading numbering / markdown / punctuation, matches a known section synonym.
Returns an insertion-ordered dict {Canonical Section Name: body text}.
"""

import re
from collections import OrderedDict

# Map many real-world heading spellings to a canonical name. Covers the section
# names used across all 10 paper-type templates plus common variants.
SECTION_SYNONYMS = {
    "abstract": "Abstract", "summary": "Abstract",
    "introduction": "Introduction", "intro": "Introduction",
    "background": "Background", "case background": "Case Background",
    "technical background": "Technical Background",
    "literature review": "Literature Review", "related work": "Literature Review",
    "literature": "Literature Review", "review of literature": "Literature Review",
    "theoretical framework": "Theoretical Framework",
    "theoretical integration": "Theoretical Integration",
    "analytical framework": "Analytical Framework",
    "methodology": "Methodology", "methods": "Methodology", "method": "Methodology",
    "materials and methods": "Methodology", "methodology development": "Methodology Development",
    "results": "Results", "findings": "Results", "results and discussion": "Results",
    "analysis": "Analysis", "comparative analysis": "Comparative Analysis",
    "case analysis": "Case Analysis", "technical analysis": "Technical Analysis",
    "cross-disciplinary analysis": "Cross-Disciplinary Analysis",
    "validation": "Validation",
    "position statement": "Position Statement", "supporting arguments": "Supporting Arguments",
    "discussion": "Discussion",
    "conclusion": "Conclusion", "conclusions": "Conclusion", "concluding remarks": "Conclusion",
    "references": "References", "bibliography": "References", "works cited": "References",
    "acknowledgements": "Acknowledgements", "acknowledgments": "Acknowledgements",
}


def _normalize_heading(line: str) -> str:
    """Strip leading/trailing numbering, markdown, bullets and colons; lowercase."""
    s = line.strip()
    s = re.sub(r"^[#>\*\-•\d\.\)\(\s:]+", "", s)   # leading "1.", "##", "-", etc.
    s = re.sub(r"[\*_#:.\s]+$", "", s)             # trailing markdown/punct: "**", "#", "."
    return s.strip().lower()


def split_sections(text: str) -> "OrderedDict[str, str]":
    """Split text into {canonical section name: body}. Empty if no headings found.

    Bodies under duplicate or synonymous headings (e.g. 'Methods' + 'Methodology',
    both canonical 'Methodology') are MERGED rather than overwritten, so no content
    is silently lost.
    """
    sections = OrderedDict()
    if not text:
        return sections
    current = None
    buf = []

    def _flush():
        if current is None:
            return
        joined = "\n".join(buf).strip()
        if current in sections:
            sections[current] = (sections[current] + "\n" + joined).strip()
        else:
            sections[current] = joined

    for line in text.split("\n"):
        norm = _normalize_heading(line)
        canon = SECTION_SYNONYMS.get(norm)
        # A heading is a short standalone line that maps to a known section.
        if canon and len(line.split()) <= 8:
            _flush()
            current = canon
            buf = []
        elif current is not None:
            buf.append(line)
    _flush()
    return sections


def detected_section_names(text: str):
    """Ordered list of canonical section names detected in the text."""
    return list(split_sections(text).keys())
