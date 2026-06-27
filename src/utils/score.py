"""
AI Peer Reviewer — paper scoring engine.

A faithful port of ResearchClawBench's `evaluation/score.py`
(github.com/Ethara-Ai/ResearchClawBench, MIT, InternScience) to OnPaper's stack.

What is kept verbatim from the original:
  * the strict-scientific-peer-reviewer RUBRIC, including both scoring modes
    (Mode A = quantitative metrics, Mode B = qualitative reasoning) and their full
    0-100 bands where **50 = as good as the reference/standard**;
  * per-criterion judging that returns {"reasoning", "score"};
  * weighted aggregation: total = sum(score * weight) / sum(weight).

What is adapted for OnPaper:
  * `structai.LLMAgent` / `multi_thread`  ->  the shared Groq/Ollama gateway
    (`generate_text`), so it routes through the same free backend as every other
    feature and degrades gracefully offline;
  * agent run-workspaces + a hand-authored `checklist.json`  ->  a user's paper
    text plus a rubric that is either auto-generated from the paper or a built-in
    academic-quality default (real OnPaper users do not ship expert checklists);
  * reference-based scoring is optional: with a reference paper the 50 baseline is
    "as good as that paper" (RCB's original semantics); without one, 50 is "a solid,
    competent, publishable paper in this field".

Image/figure criteria from RCB require a vision judge; OnPaper's gateway is text
only, so an `image` item is scored on whether the paper's text describes a figure
that satisfies the criterion (clearly noted in the reasoning), not on pixels.
"""

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from .gemini_helper import generate_text, is_unavailable_response

# --- RUBRIC (kept faithful to ResearchClawBench/evaluation/score.py) --------------
RUBRIC = """You are a strict scientific peer reviewer evaluating a research paper or report against a specific quality criterion.

You are given:
1. Background about the paper (its topic and task).
2. The paper / report text to evaluate.
3. A specific evaluation criterion.

IMPORTANT: Your role is ONLY to score the paper against the criterion. Do NOT attempt to rewrite the paper or solve its research task yourself. Focus solely on evaluating what is written.

## Evaluation Modes

Each criterion falls into one of two categories. Determine which applies based on the criterion's nature:

### Mode A: Objective Evaluation (Metric Optimization / Quantitative Results)
Use this when the criterion involves specific numerical results, metrics, benchmarks, or quantitative outcomes.

- **0**: The criterion is completely absent from the report.
- **1-10**: Mentioned but no quantitative results provided.
- **11-20**: Quantitative results given but the methodology has fundamental errors.
- **21-30**: Methodology has significant flaws; metrics deviate severely from the reference.
- **31-40**: Methodology is mostly correct but metrics are notably worse than the reference.
- **41-50**: Metrics are roughly comparable to the reference.
- **51-60**: Metrics are slightly better than the reference.
- **61-70**: Metrics are clearly better than the reference.
- **71-80**: Both methodology and metrics show substantial improvements over the reference.
- **81-90**: Metrics dramatically surpass the reference.
- **91-100**: Breakthrough results far exceeding the reference.

### Mode B: Subjective Evaluation (Mechanism Analysis / Qualitative Reasoning)
Use this when the criterion involves theoretical explanations, mechanistic insights, logical arguments, or interpretive analysis.

- **0**: The criterion is completely absent from the report.
- **1-10**: Mentioned only with vague, generic statements.
- **11-20**: Some description present but no substantive analysis.
- **21-30**: Some analysis attempted but evidence is insufficient or reasoning has logical gaps.
- **31-40**: Analysis direction is correct but lacks depth; key arguments are missing.
- **41-50**: Analysis depth and logical rigor are roughly comparable to the reference.
- **51-60**: More supporting evidence provided than the reference.
- **61-70**: More complete logical chain and more rigorous argumentation than the reference.
- **71-80**: Significantly deeper analysis; raises valuable insights not covered in the reference.
- **81-90**: Analysis depth far exceeds the reference.
- **91-100**: Original contributions with breakthrough insights beyond the reference.

## CRITICAL RULES
- 50 means "as good as the reference standard" (a solid, competent, publishable paper) -- this is a high bar.
- First determine if the criterion is Objective (Mode A) or Subjective (Mode B), then apply the corresponding rubric.
- No credit for vague or generic statements. Must demonstrate specific, concrete analysis.
- No inflation for well-written but shallow content. Substance over style. Longer does not mean better.
- Be highly skeptical of AI-generated content: it may sound plausible but contain factual errors, fabricated numbers, or unsupported conclusions. Verify claims against the criterion carefully.
- Be strict but fair.
"""

# Built-in academic-quality rubric used when the user does not auto-generate one.
# Weights sum to 1.0; each is a "text" criterion with keywords the judge must verify.
DEFAULT_RUBRIC = [
    {"type": "text", "weight": 0.15,
     "content": "The paper states a clear, specific research question or objective and explains why it matters.",
     "keywords": ["explicit research question or thesis", "motivation / significance", "scope of the study"]},
    {"type": "text", "weight": 0.20,
     "content": "The methodology is rigorous, appropriate to the question, and described in enough detail to be reproducible.",
     "keywords": ["data sources and sample", "method / procedure", "reproducibility", "justification of approach"]},
    {"type": "text", "weight": 0.20,
     "content": "The results are presented clearly and are genuinely supported by the data or evidence, not asserted.",
     "keywords": ["concrete results", "evidence backing each claim", "no unsupported or fabricated numbers"]},
    {"type": "text", "weight": 0.15,
     "content": "The discussion interprets the results with real analytical depth rather than restating them.",
     "keywords": ["interpretation of findings", "depth of analysis", "implications", "alternative explanations"]},
    {"type": "text", "weight": 0.10,
     "content": "The work is positioned against relevant prior literature with accurate, specific references.",
     "keywords": ["related work", "accurate citations", "contribution relative to prior work"]},
    {"type": "text", "weight": 0.10,
     "content": "The writing is clear, well structured, and uses precise academic language.",
     "keywords": ["logical structure", "clarity", "academic tone", "consistent terminology"]},
    {"type": "text", "weight": 0.10,
     "content": "Limitations are acknowledged honestly and sensible future work is identified.",
     "keywords": ["stated limitations", "threats to validity", "future directions"]},
]


def _clamp_score(value) -> int:
    """0-100 integer, truncating like RCB's int(); rejects non-finite (Infinity/NaN)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(f):
        return 0
    return max(0, min(100, int(f)))


def _strip_one_fence(s: str) -> str:
    """Unwrap a single whole-string ```...``` fence if present."""
    m = re.match(r"^```[a-zA-Z0-9]*\s*(.*?)\s*```$", s.strip(), re.S)
    return m.group(1).strip() if m else s.strip()


def _json_candidates(s: str, open_ch: str, close_ch: str):
    """Yield balanced open_ch..close_ch substrings, ignoring delimiters inside JSON
    strings, so braces/brackets in reasoning text do not confuse the scan."""
    n = len(s)
    for i in range(n):
        if s[i] != open_ch:
            continue
        depth = 0
        in_str = False
        esc = False
        for j in range(i, n):
            ch = s[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    yield s[i:j + 1]
                    break


def _best_score_object(raw: str):
    """Find the judge's intended {reasoning, score} object. Prefers an object that has
    BOTH keys (the real answer) over a bare {score: ...} that may appear earlier."""
    s = _strip_one_fence(raw or "")
    dicts = []
    try:
        d = json.loads(s)
        if isinstance(d, dict):
            dicts.append(d)
    except (json.JSONDecodeError, ValueError):
        pass
    if not dicts:
        for cand in _json_candidates(s, "{", "}"):
            try:
                d = json.loads(cand)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(d, dict):
                dicts.append(d)
    with_score = [d for d in dicts if "score" in d]
    both = [d for d in with_score if "reasoning" in d]
    if both:
        return both[0]
    if with_score:
        return with_score[0]
    return None


def _parse_score(raw: str) -> dict:
    """Pull {"reasoning", "score"} out of the judge's reply.

    Returns a `parsed` flag that is True only when a real JSON score object was found,
    so the caller can distinguish a genuine judge reply (even one that opens with
    'Error:') from a backend-outage string that has no score at all.
    """
    raw = raw or ""
    obj = _best_score_object(raw)
    if obj is not None:
        return {"score": _clamp_score(obj.get("score", 0)),
                "reasoning": _clean_str(obj.get("reasoning")) or "No reasoning returned.",
                "evidence": _clean_str(obj.get("evidence")),
                "location": _clean_str(obj.get("location")),
                "severity": _norm_severity(obj.get("severity", "")),
                "fix": _clean_str(obj.get("fix")),
                "parsed": True}
    # Fallback: prefer an anchored 'score: <n>', and the LAST such match (the final
    # stated score), so band text like '51-60' earlier in the prose is not grabbed.
    ms = list(re.finditer(r'"?score"?\s*[:=]\s*(\d{1,3})', raw, re.I))
    if not ms:
        ms = list(re.finditer(r"score\D{0,6}(\d{1,3})", raw, re.I))
    score = _clamp_score(ms[-1].group(1)) if ms else 0
    return {"score": score, "reasoning": raw.strip()[:400] or "Failed to parse scoring response.",
            "evidence": "", "location": "", "severity": "", "fix": "", "parsed": False}


def _baseline_line(reference_text: str) -> str:
    if (reference_text or "").strip():
        return ("A reference paper is provided below. Treat the 50 baseline as 'as good as that "
                "reference paper'.")
    return ("No reference paper was provided. Treat the 50 baseline as the standard of a solid, "
            "competent, publishable paper in this field.")


# Severity model + evidence/fix discipline, adapted from unified-personas/REVIEW.md
# (adversarial skeptic: quote verbatim evidence, cite the location, classify severity,
# emit a concrete fix). Shared by the text and image judging prompts.
_JUDGMENT_INSTRUCTIONS = """Classify the severity of the MAIN weakness against this criterion:
- Critical: the criterion is fundamentally unmet, factually wrong, or self-contradictory; a reviewer would reject on this point.
- Major: a clear gap that would draw a required-revision request (missing evidence, unsupported claim, weak method).
- Minor: a small, easily fixed weakness a careful reviewer flags but that survives a casual read.
- None: the criterion is genuinely well met with no material weakness.

Quote the single most relevant VERBATIM sentence from the paper as evidence (use "not addressed" if the criterion is absent). Name the section where it appears. Give one concrete, actionable fix the author can apply.

Return your answer as a JSON object:
{"reasoning": "<2-3 sentences>", "score": <0-100>, "evidence": "<verbatim quote or 'not addressed'>", "location": "<section/where in the paper>", "severity": "<Critical|Major|Minor|None>", "fix": "<one actionable fix>"}"""

_SEV_RANK = {"Critical": 0, "Major": 1, "Minor": 2, "None": 4, "": 3}


def _clean_str(value) -> str:
    """Stringify a JSON field, mapping null/None to '' so it stays falsy (the model
    may return null for evidence/fix on a well-met criterion)."""
    return "" if value is None else str(value).strip()


def _norm_severity(value) -> str:
    s = str(value or "").strip().lower()
    for label in ("critical", "major", "minor", "none"):
        if s == label or s.startswith(label):
            return label.capitalize()
    return ""


def _safe(s: str) -> str:
    """Neutralize the triple-quote delimiter so embedded text can't break out of it."""
    return (s or "").replace('"""', "'''")


def _build_text_prompt(paper_text: str, item: dict, background: str, reference_text: str) -> str:
    criteria = item.get("content", "")
    keywords = item.get("keywords", [])
    keywords_str = ", ".join(keywords) if keywords else "None specified"
    ref_block = f"\n## Reference Paper (the 50 baseline)\n{reference_text[:8000]}\n" if (reference_text or "").strip() else ""
    bg_block = f"\n## Paper Background\n{background}\n" if (background or "").strip() else ""
    return f"""{RUBRIC}

{_baseline_line(reference_text)}
{bg_block}{ref_block}
## Evaluation Criterion
{criteria}

## Key Aspects to Verify
{keywords_str}

## Paper / Report Text
{paper_text}

## Task
Rate how well the paper addresses the criterion. First determine if this criterion is Objective (Mode A) or Subjective (Mode B), then apply the corresponding rubric strictly.

{_JUDGMENT_INSTRUCTIONS}"""


def _build_image_prompt(paper_text: str, item: dict, background: str, reference_text: str) -> str:
    criteria = item.get("content", "")
    keywords = item.get("keywords", [])
    keywords_str = ", ".join(keywords) if keywords else "None specified"
    bg_block = f"\n## Paper Background\n{background}\n" if (background or "").strip() else ""
    return f"""{RUBRIC}

{_baseline_line(reference_text)}
{bg_block}
## Evaluation Criterion (figure / visual)
{criteria}

## Key Visual/Technical Aspects to Verify
{keywords_str}

## Paper / Report Text
{paper_text[:12000]}

## Task
This criterion concerns a figure or visual result. Visual inspection is unavailable here, so judge it from how the text describes the figure: whether the paper reports a figure with the right variables, scale, trend, and data to satisfy the criterion. State in your reasoning that this was assessed from the text description. First determine Mode A or Mode B, then apply the rubric strictly. A figure that is only mentioned, or described with wrong scales/missing data, should score low.

{_JUDGMENT_INSTRUCTIONS}"""


def _score_single_item(paper_text: str, item: dict, background: str, reference_text: str) -> dict:
    """Score one checklist item (text or image) via the gateway."""
    if item.get("type") == "image":
        prompt = _build_image_prompt(paper_text, item, background, reference_text)
    else:
        prompt = _build_text_prompt(paper_text, item, background, reference_text)
    raw = generate_text(prompt)
    parsed = _parse_score(raw)
    # Only treat as a backend outage when there is NO real score in the reply. A
    # judge reply that merely opens with 'Error:' still carries valid JSON and must
    # not be dropped as "unavailable".
    if not parsed.get("parsed") and is_unavailable_response(raw):
        return {"score": 0, "reasoning": "AI backend unavailable, item not scored.",
                "evidence": "", "location": "", "severity": "", "fix": "", "unavailable": True}
    return parsed


def score_paper(paper_text: str, checklist=None, reference_text: str = "",
                background: str = "", max_workers: int = 4) -> dict:
    """Score a paper against a weighted checklist. Mirrors RCB's score_run, but takes
    text in directly instead of reading a run workspace.

    Returns {items, total_score, total_weight, mode, error}. `mode` is
    'reference' when a reference paper was supplied, else 'standard'.
    """
    paper_text = (paper_text or "").strip()
    if not paper_text:
        return {"items": [], "total_score": 0, "total_weight": 0, "mode": "standard",
                "error": "Please provide the paper text to review."}

    checklist = checklist if checklist else DEFAULT_RUBRIC
    checklist = [c for c in checklist if isinstance(c, dict) and c.get("content")]
    if not checklist:
        return {"items": [], "total_score": 0, "total_weight": 0, "mode": "standard",
                "error": "The rubric is empty."}

    mode = "reference" if (reference_text or "").strip() else "standard"

    # RCB scores items with 16-way parallelism via structai.multi_thread. We use a
    # small bounded pool so the shared free-tier gateway is not hammered, while still
    # scoring criteria concurrently. generate_text makes stateless HTTP calls.
    workers = max(1, min(max_workers, len(checklist)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        raw_results = list(pool.map(
            lambda it: _score_single_item(paper_text, it, background, reference_text),
            checklist,
        ))

    if raw_results and all(r.get("unavailable") for r in raw_results):
        return {"items": [], "total_score": 0, "total_weight": 0, "mode": mode,
                "warning": None, "scored_items": 0, "skipped_items": len(raw_results),
                "error": "AI features are not available right now. Please try again later."}

    items = []
    total_weighted = 0.0
    total_weight = 0.0
    plain_scores = []          # for an unweighted fallback if every weight is zero
    skipped = 0
    for i, (item, sr) in enumerate(zip(checklist, raw_results)):
        score = sr.get("score", 0)
        unavailable = bool(sr.get("unavailable"))
        try:
            weight = float(item.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        if not math.isfinite(weight) or weight < 0:  # reject Infinity/NaN/negative
            weight = 0.0
        items.append({
            "index": i,
            "type": item.get("type", "text"),
            "content": item.get("content", ""),
            "weight": weight,
            "score": score,
            "reasoning": sr.get("reasoning", ""),
            "evidence": sr.get("evidence", ""),
            "location": sr.get("location", ""),
            "severity": sr.get("severity", ""),
            "fix": sr.get("fix", ""),
            "unavailable": unavailable,
        })
        # Items the backend could not score are EXCLUDED from the total (not counted
        # as weight-bearing zeros), so a partial outage cannot silently deflate it.
        if unavailable:
            skipped += 1
            continue
        plain_scores.append(score)
        total_weighted += score * weight
        total_weight += weight

    if total_weight > 0:
        final_score = round(total_weighted / total_weight, 2)
    elif plain_scores:
        # Every contributing weight was zero; fall back to an unweighted mean rather
        # than reporting a misleading 0.
        final_score = round(sum(plain_scores) / len(plain_scores), 2)
    else:
        final_score = 0

    warning = None
    if skipped:
        warning = (f"{skipped} of {len(checklist)} criteria could not be scored (the AI backend was "
                   f"unavailable for them) and were excluded from the total.")

    return {"items": items, "total_score": final_score, "total_weight": round(total_weight, 4),
            "mode": mode, "scored_items": len(plain_scores), "skipped_items": skipped,
            "warning": warning, "error": None}


def generate_checklist(paper_text: str, num_items: int = 7) -> dict:
    """Auto-build a weighted rubric from the paper (RCB's checklists are expert-authored;
    OnPaper users have none, so we derive one). Returns {checklist, error}.

    Falls back to DEFAULT_RUBRIC on any failure so scoring still works offline.
    """
    paper_text = (paper_text or "").strip()
    if not paper_text:
        return {"checklist": [], "error": "Please provide the paper text first."}

    num_items = max(3, min(12, int(num_items) if str(num_items).isdigit() else 7))
    prompt = (
        "You are a journal editor designing an evaluation rubric for the paper below. "
        f"Produce exactly {num_items} weighted criteria a strict peer reviewer would score it on. "
        "Cover the research question, methodology rigor, results and evidence, analytical depth, "
        "related work, writing quality, and limitations as relevant to THIS paper. The weights must "
        "be positive numbers that sum to 1.0.\n\n"
        "Respond with ONLY a JSON array, each element exactly:\n"
        '{"content": "<the criterion, one specific testable sentence>", '
        '"keywords": ["<technical aspect to verify>", "..."], '
        '"weight": <number>, "type": "text"}\n\n'
        f"Paper:\n\"\"\"{paper_text[:12000]}\"\"\""
    )
    raw = generate_text(prompt)
    if is_unavailable_response(raw):
        return {"checklist": DEFAULT_RUBRIC, "error": None, "fallback": True}

    parsed = _best_array(raw)
    if not isinstance(parsed, list):
        return {"checklist": DEFAULT_RUBRIC, "error": None, "fallback": True}

    checklist = []
    for it in parsed:
        if not isinstance(it, dict) or not it.get("content"):
            continue
        kws = it.get("keywords") or []
        if not isinstance(kws, list):
            kws = [str(kws)]
        try:
            w = float(it.get("weight", 0))
        except (TypeError, ValueError):
            w = 0.0
        if not math.isfinite(w) or w <= 0:  # reject Infinity/NaN/non-positive
            w = 1.0
        checklist.append({
            "type": "image" if str(it.get("type", "")).lower() == "image" else "text",
            "content": str(it["content"]).strip(),
            "keywords": [str(k).strip() for k in kws if str(k).strip()],
            "weight": w,
        })

    if not checklist:
        return {"checklist": DEFAULT_RUBRIC, "error": None, "fallback": True}
    return {"checklist": checklist, "error": None, "fallback": False}


def _best_array(raw: str):
    """Extract the first valid JSON array from the reply, tolerant of code fences,
    leading preamble, and trailing prose that itself contains brackets (e.g. '[12]')."""
    s = _strip_one_fence(raw or "")
    try:
        d = json.loads(s)
        if isinstance(d, list):
            return d
    except (json.JSONDecodeError, ValueError):
        pass
    for cand in _json_candidates(s, "[", "]"):
        try:
            d = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(d, list):
            return d
    return None


def build_revision_plan(result: dict, limit: int = 6) -> list:
    """Prioritized, highest-impact-first list of weaknesses to fix (no LLM call).

    Orders scored criteria by severity (Critical -> Major -> Minor) then by
    weight * (100 - score), and keeps those with a real weakness (a Critical/Major/
    Minor severity, or a score below 60). Each entry carries the actionable fix.
    """
    items = [it for it in (result.get("items") or []) if not it.get("unavailable")]

    def _key(it):
        sev = it.get("severity", "")
        impact = it.get("weight", 0.0) * (100 - it.get("score", 0))
        return (_SEV_RANK.get(sev, 3), -impact)

    plan = []
    for it in sorted(items, key=_key):
        sev = it.get("severity", "")
        if sev in ("Critical", "Major", "Minor") or it.get("score", 0) < 60:
            plan.append({
                "content": it.get("content", ""),
                "score": it.get("score", 0),
                "severity": sev,
                "fix": it.get("fix", ""),
                "evidence": it.get("evidence", ""),
                "location": it.get("location", ""),
            })
        if len(plan) >= max(1, limit):
            break
    return plan


def revise_paper(paper_text: str, plan: list) -> dict:
    """Produce a revised draft that addresses the reviewer findings in `plan`.

    One gateway call. Preserves the author's content and voice, never invents data,
    results, or citations (inserts a bracketed [ADD: ...] note where the author must
    supply specifics). Returns {revised, error}.
    """
    paper_text = (paper_text or "").strip()
    if not paper_text:
        return {"revised": "", "error": "Nothing to revise."}
    plan = plan or []
    if not plan:
        return {"revised": "", "error": "No revisions needed; nothing to address."}

    findings = []
    for i, p in enumerate(plan, 1):
        sev = f"[{p.get('severity')}] " if p.get("severity") else ""
        line = f"{i}. {sev}{p.get('content', '')}\n   Fix: {(_clean_str(p.get('fix')) or 'strengthen this point.')}"
        ev = _clean_str(p.get("evidence"))
        if ev and ev.lower() != "not addressed":
            line += f"\n   Current text: \"{ev}\""
        findings.append(line)

    prompt = (
        "You are revising an academic paper to address a reviewer's findings. Apply each fix while "
        "preserving the author's content, structure, and voice. Do NOT invent data, results, "
        "statistics, or citations; where a fix needs information only the author has, insert a clearly "
        "bracketed note like [ADD: specific result needed here] instead of fabricating it. "
        "Return ONLY the full revised paper text, with no preamble or commentary.\n\n"
        f"Reviewer findings to address:\n" + "\n".join(findings) + "\n\n"
        f'Paper:\n"""{_safe(paper_text)}"""'
    )
    out = generate_text(prompt)
    if is_unavailable_response(out):
        return {"revised": "", "error": out or "AI features are not available."}
    revised = _strip_one_fence((out or "").strip()).strip()
    if not revised:
        return {"revised": "", "error": "The model returned an empty revision. Please try again."}
    return {"revised": revised, "error": None}
