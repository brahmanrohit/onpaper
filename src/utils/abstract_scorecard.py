"""
Abstract Quality Scorecard (offline).

Scores an abstract on the five expected "moves" (background, gap/aim, methods,
results, conclusion) plus length and readability, with a concrete tip per
missing move. If the pasted text has an "Abstract" heading it is used; otherwise
the whole input is treated as the abstract. No AI backend needed.
"""

import re
from nltk.tokenize import sent_tokenize
from .section_splitter import split_sections
from .readability_analyzer import analyze_text, compare_to_target

# Tighter, more specific cues — the earlier versions matched everyday words
# (e.g. "data", "using", "shows") and almost always fired, inflating the score.
MOVES = {
    "Background / context": re.compile(
        r"\b(increasingly important|in recent years|growing (?:interest|concern|body)|"
        r"has become|widely (?:used|recognized)|plays? a (?:critical|significant|key) role|"
        r"remains? a (?:challenge|problem)|emerging)\b", re.I),
    "Gap / aim": re.compile(
        r"\b(however,|this (?:paper|study|review|article|work) (?:aims|investigat|examin|present|propos|explor)|"
        r"we (?:propose|investigate|examine|present|explore|develop|aim)|the aim of|our (?:aim|objective|goal)|"
        r"to address|research gap|lack of|remains? (?:unclear|unknown|poorly understood))", re.I),
    "Methods": re.compile(
        r"\b(we (?:analy[sz]ed?|conducted|collected|used|developed|surveyed|measured|trained)|"
        r"using a|based on (?:a|data)|data (?:were|was|from|set)|a (?:dataset|survey|experiment|"
        r"framework|model|method|methodology) (?:of|was|is)|participants|sample of|"
        r"experiments?|methodology)", re.I),
    "Results": re.compile(
        r"\b(results? (?:show|indicate|suggest|reveal)|we (?:find|found|observe[d]?|show)|"
        r"findings? (?:show|indicate|suggest)|showed that|revealed that|demonstrated that|"
        r"outperform|achieved (?:an?|\d)|significant(?:ly)? (?:higher|lower|improve|increase|decrease))", re.I),
    "Conclusion / implication": re.compile(
        r"\b(we conclude|in conclusion|conclusion[s]?:?|implications?|we suggest|"
        r"contribut|future (?:work|research)|in summary|these (?:results|findings) (?:suggest|imply|highlight)|"
        r"underscore)\b", re.I),
}

TIPS = {
    "Background / context": "Open with why the topic matters / its context.",
    "Gap / aim": "State the research gap and the paper's aim or question explicitly.",
    "Methods": "Briefly say how the work was done (approach, data, or method).",
    "Results": "State the main finding or outcome.",
    "Conclusion / implication": "End with the takeaway / implication or contribution.",
}


def score_abstract(text: str) -> dict:
    secs = split_sections(text)
    if "Abstract" in secs:
        abstract = (secs["Abstract"] or "").strip()
        if not abstract:
            return {"error": "An 'Abstract' heading was found but has no text under it."}
        had_heading = True
    else:
        abstract = (text or "").strip()
        had_heading = False
    if not abstract:
        return {"error": "No abstract text found."}

    moves = {name: bool(rx.search(abstract)) for name, rx in MOVES.items()}
    present = sum(moves.values())
    missing_tips = [TIPS[name] for name, ok in moves.items() if not ok]

    a = analyze_text(abstract)
    if isinstance(a, dict) and a.get("word_count") is not None:
        wc = a["word_count"]
    else:  # analyze_text returned an error dict (e.g. no real words)
        wc = len([w for w in abstract.split() if any(c.isalnum() for c in w)])
    length = compare_to_target(wc, 150, 250)
    flesch = a.get("flesch_reading_ease") if isinstance(a, dict) else None

    # Score: 70% from moves coverage, 30% from length being on target.
    length_factor = 1.0 if length["status"] == "ok" else 0.5
    score = round((present / len(MOVES)) * 70 + length_factor * 30)

    return {
        "word_count": wc,
        "moves": moves,
        "moves_present": present,
        "moves_total": len(MOVES),
        "length_status": length["status"],
        "length_message": length["message"],
        "flesch_reading_ease": flesch,
        "score": score,
        "tips": missing_tips,
        "had_abstract_heading": had_heading,
        "error": None,
    }
