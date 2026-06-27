"""
Central AI gateway for the Research Paper Assistant.

NOTE: the module is still named `gemini_helper` for backward compatibility
(every feature module imports `generate_text` from here), but Gemini has been
removed. The app now uses open-source models only:

  * Groq   — fast, free cloud (Llama 3.3 70B etc.), the default primary.
  * Ollama — fully local, open-source, unlimited & offline (fallback).

`generate_text(prompt, backend="auto")` is the single entry point used by every
generation feature. In "auto" mode it tries Groq first (fast), then falls back
to a local Ollama server if one is reachable.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from .ollama_helper import get_ollama_helper, is_ollama_available
from .groq_helper import generate_text_with_groq, is_groq_available, get_groq_model

# Make console output robust on Windows terminals (cp1252) so that printing
# unicode glyphs (e.g. status symbols) never raises UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# Load environment variables from multiple possible locations
def load_env_file():
    """Load .env file from multiple possible locations."""
    current_dir = Path(__file__).parent
    possible_paths = [
        current_dir / '.env',  # In utils directory
        current_dir.parent.parent / '.env',  # In project root (onpaperfixed)
        current_dir.parent.parent.parent / '.env',  # In parent directory
        Path.cwd() / '.env',  # In current working directory
        Path.home() / '.env',  # In user home directory
    ]

    for env_path in possible_paths:
        if env_path.exists():
            print(f"Loading .env from: {env_path}")
            load_dotenv(env_path, override=True)
            return True

    print("Warning: No .env file found in any of the expected locations")
    return False


# Load environment variables
load_env_file()

# Report backend availability at import (informational only).
if is_groq_available():
    print(f"Groq backend ready (model: {get_groq_model()}).")
elif is_ollama_available():
    print("Ollama backend ready (local).")
else:
    print("Warning: No AI backend configured. Generation/grammar/paraphrase will be limited.")
    print("Enable one of:")
    print("  - Groq (free, fast):   set GROQ_API_KEY in .env  (https://console.groq.com/keys)")
    print("  - Ollama (local, free): install Ollama and run `ollama pull llama3.2`")


def is_unavailable_response(result) -> bool:
    """True if a generate_text() result is an error / 'backend unavailable' string.

    Centralized here so callers don't each re-implement the same checks.
    """
    if not result:
        return True
    lowered = str(result).strip().lower()
    # Match only the specific sentinel messages this gateway / the backends
    # produce — NOT any prose that happens to start with "Error" or contain
    # "is not available" (which are valid in real generated/rewritten text).
    return (
        lowered.startswith("ai generation failed")
        or lowered.startswith("ai features are not available")
        or lowered.startswith("error:")          # groq_helper sentinels all use "Error:"
        or lowered.startswith("groq error")
        or lowered.startswith("ollama error")
        or lowered.startswith("gemini error")
        or lowered.startswith("backend '")
        or lowered.startswith("ollama is not available")
        or lowered.startswith("gemini ai is not available")
    )


def generate_text(prompt, backend="auto"):
    """Generate text using an available AI backend (Groq or Ollama).

    "auto" tries Groq first (fast cloud), then falls back to a local Ollama
    server. Set AI_BACKEND in .env, or pass backend="groq"/"ollama" explicitly.
    """
    # Get backend preference from environment or parameter
    if backend == "auto":
        backend = os.getenv("AI_BACKEND", "auto").lower()

    # Track the most recent backend error so 'auto' mode can report the real
    # cause instead of a generic "not available" message.
    last_error = None

    # Try Groq first (fast, free cloud) if available and preferred.
    if backend in ["auto", "groq"] and is_groq_available():
        try:
            result = generate_text_with_groq(prompt)
            if not result.startswith("Error"):
                return result
            elif backend == "groq":
                return result
            else:
                last_error = result
        except Exception as e:
            if backend == "groq":
                return f"Groq error: {str(e)}"
            last_error = f"Groq error: {str(e)}"

    # Fall back to a local Ollama server (unlimited, offline) if reachable.
    if backend in ["auto", "ollama"] and is_ollama_available():
        try:
            ollama_helper = get_ollama_helper()
            result = ollama_helper.generate_text(prompt)
            if not result.startswith("Error") and not result.startswith("Ollama is not available"):
                return result
            elif backend == "ollama":
                return result
            else:
                last_error = result
        except Exception as e:
            if backend == "ollama":
                return f"Ollama error: {str(e)}"
            last_error = f"Ollama error: {str(e)}"

    # If we get here, no backend produced a result.
    if last_error:
        return (
            "AI generation failed. Please check your GROQ_API_KEY or your Ollama "
            f"setup. Details: {last_error}"
        )
    if backend == "auto":
        return ("AI features are not available. Set GROQ_API_KEY in .env "
                "(https://console.groq.com/keys) or run a local Ollama server.")
    else:
        return f"Backend '{backend}' is not available. Please check your configuration."


def generate_text_with_ollama(prompt):
    """Generate text using only Ollama."""
    if not is_ollama_available():
        return "Ollama is not available. Please ensure Ollama is running."

    try:
        ollama_helper = get_ollama_helper()
        return ollama_helper.generate_text(prompt)
    except Exception as e:
        return f"Error generating text with Ollama: {str(e)}"


def get_available_backends():
    """Get list of available AI backends."""
    backends = []
    if is_groq_available():
        backends.append("groq")
    if is_ollama_available():
        backends.append("ollama")
    return backends


def get_backend_status():
    """Get status of all AI backends."""
    status = {
        "groq": {
            "available": is_groq_available(),
            "model": get_groq_model() if is_groq_available() else None
        },
        "ollama": {
            "available": is_ollama_available(),
            "models": get_ollama_helper().list_models() if is_ollama_available() else []
        }
    }
    return status
