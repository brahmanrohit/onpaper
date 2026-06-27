"""
Citation Assistant backend.

Previously this returned hardcoded MOCK papers regardless of the query, which
is dangerous in a research tool (users could export fabricated references into
real work). It now delegates to reference_finder, which queries the free
CrossRef API for REAL papers with DOIs and formats them in APA / IEEE / MLA.
"""

from .reference_finder import search_references, format_reference


def suggest_citations(query, rows: int = 8):
    """Find real citations for a query via CrossRef.

    Returns a list of normalized reference dicts (title/authors/year/journal/
    doi/url). Returns an empty list if nothing is found or the service errors,
    so existing callers that slice/iterate the result keep working.
    """
    result = search_references(query, rows=rows)
    return result.get("references", [])


def format_citation(paper, style="APA"):
    """Format a reference dict (from suggest_citations) in APA / IEEE / MLA."""
    return format_reference(paper, style=style)
