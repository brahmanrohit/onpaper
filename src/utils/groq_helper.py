"""
Groq AI Helper for Research Paper Assistant.

Groq exposes an OpenAI-compatible Chat Completions API and a very fast,
generous free tier — a good default backend for this app's many per-section
generation calls. Uses `requests` (already a dependency); no SDK needed.

Configure via .env:
    GROQ_API_KEY=gsk_...                      # from https://console.groq.com/keys
    GROQ_MODEL=llama-3.3-70b-versatile        # optional, this is the default
    GROQ_BASE_URL=https://api.groq.com/openai/v1   # optional
    GROQ_MAX_TOKENS=4096                       # optional
"""

import os
import re
import time
import requests

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MAX_TOKENS = 4096

# Free-tier Groq enforces a tokens-per-minute (TPM) limit, and this app fires
# many calls per paper (section-by-section). Retrying transient 429s/5xx with
# the server-suggested wait keeps full-paper generation from failing mid-run.
MAX_RETRIES = 4
MAX_RETRY_WAIT = 30.0  # seconds — never block the UI longer than this per wait


def _get_api_key() -> str:
    """Return the configured Groq key, or '' if missing/placeholder."""
    key = os.getenv("GROQ_API_KEY", "")
    if not key or key.strip() in ("", "your_groq_api_key_here"):
        return ""
    return key.strip()


def is_groq_available() -> bool:
    """True if a Groq API key is configured.

    Cheap, no network call (unlike Ollama's probe) — keeps app startup fast.
    Actual reachability/validity is surfaced as an error string at call time.
    """
    return bool(_get_api_key())


def get_groq_model() -> str:
    """The model id Groq will use (env override or sensible default)."""
    return os.getenv("GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def generate_text_with_groq(prompt: str) -> str:
    """Generate text via Groq's OpenAI-compatible Chat Completions endpoint.

    Returns the generated text on success, or a string starting with "Error"
    on failure (so gemini_helper.is_unavailable_response() / 'auto' fallback
    treats it correctly and surfaces the real cause).
    """
    api_key = _get_api_key()
    if not api_key:
        return "Error: GROQ_API_KEY not set. Add it to your .env file."

    base_url = os.getenv("GROQ_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    model = get_groq_model()
    try:
        max_tokens = int(os.getenv("GROQ_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
    except ValueError:
        max_tokens = DEFAULT_MAX_TOKENS

    payload = {
        "model": model,
        # Send only the user prompt (no system message) so behaviour matches
        # the Gemini path exactly — feature prompts (e.g. grammar JSON output)
        # are already self-contained.
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_detail = ""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
        except requests.exceptions.RequestException as e:
            # Transient network error — back off and retry.
            last_detail = f"could not reach Groq API: {str(e)}"
            if attempt < MAX_RETRIES - 1:
                time.sleep(min(2 ** attempt, MAX_RETRY_WAIT))
                continue
            return f"Error: {last_detail}"

        if response.status_code == 200:
            try:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return content if content else "Error: Empty response from Groq"
            except (KeyError, IndexError, ValueError) as e:
                return f"Error: unexpected Groq response format: {str(e)}"

        # Non-200: capture the real API message (401 invalid key, 429 TPM, etc.)
        try:
            last_detail = response.json().get("error", {}).get("message", "")
        except Exception:
            last_detail = response.text[:300]

        # Retry transient failures (rate limit / server errors); the rest are
        # terminal (e.g. 401 invalid key) and should fail fast.
        if response.status_code == 429 or response.status_code >= 500:
            if attempt < MAX_RETRIES - 1:
                time.sleep(_retry_wait(response, last_detail, attempt))
                continue

        return f"Error: Groq API returned {response.status_code}: {last_detail}"

    return f"Error: Groq API failed after {MAX_RETRIES} attempts: {last_detail}"


def _retry_wait(response, detail: str, attempt: int) -> float:
    """How long to wait before retrying, in seconds.

    Prefers the server's `Retry-After` header, then the wait hinted in the
    error message ("try again in 1.34s"), then exponential backoff — all
    capped at MAX_RETRY_WAIT so the UI never hangs.
    """
    # 1. Retry-After header (seconds).
    header = response.headers.get("Retry-After") or response.headers.get("retry-after")
    if header:
        try:
            return min(float(header), MAX_RETRY_WAIT)
        except ValueError:
            pass
    # 2. Wait hinted in the message body.
    match = re.search(r"try again in ([\d.]+)\s*s", detail)
    if match:
        try:
            return min(float(match.group(1)) + 0.25, MAX_RETRY_WAIT)
        except ValueError:
            pass
    # 3. Exponential backoff.
    return min(2 ** attempt, MAX_RETRY_WAIT)
