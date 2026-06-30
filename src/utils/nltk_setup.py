"""Idempotent NLTK data bootstrap.

nltk >= 3.9 routes ``sent_tokenize`` / ``word_tokenize`` through the
``punkt_tab`` resource, NOT the legacy ``punkt``. Code that only downloads
``punkt`` (as this project's modules used to) raises
``LookupError: Resource punkt_tab not found`` on a clean environment such as a
fresh Streamlit Community Cloud container — it only "works" locally because
``punkt_tab`` happens to already be cached.

This single helper ensures every NLTK resource the app tokenizes with is
present, downloads each only if missing, never raises, and runs its work at
most once per process. Every module that tokenizes calls ``ensure_nltk_data()``
instead of rolling its own download block.
"""

import nltk

# (find-path, download-name) for each resource the app relies on.
# punkt_tab is the nltk>=3.9 tokenizer table; punkt is kept as a fallback for
# older nltk; stopwords is used by the plagiarism preprocessor.
_RESOURCES = (
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("tokenizers/punkt", "punkt"),
    ("corpora/stopwords", "stopwords"),
)

_done = False


def ensure_nltk_data():
    """Download the NLTK corpora the app needs; safe to call repeatedly.

    Each resource is fetched only if absent. Network/offline failures are
    swallowed so a restricted environment degrades gracefully instead of
    crashing at import time.
    """
    global _done
    if _done:
        return
    for find_path, name in _RESOURCES:
        try:
            nltk.data.find(find_path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception:
                pass
    _done = True
