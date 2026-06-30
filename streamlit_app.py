"""Streamlit Community Cloud entry point.

Streamlit Cloud auto-detects a repo-root ``streamlit_app.py`` and runs it with
the repo root as the working directory. The real UI lives in ``main/main.py``;
this thin shim puts the project root (and ``main/``) on ``sys.path`` and executes
it, so the cloud deploy works with zero dashboard configuration.

Local development is unchanged — keep using ``python run.py`` (which ``cd``s into
``main/`` and runs ``streamlit run main.py``).
"""

import os
import runpy
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for path in (ROOT, os.path.join(ROOT, "main")):
    if path not in sys.path:
        sys.path.insert(0, path)

# Execute main/main.py as if it were the launched script (run_name="__main__"
# so its `if __name__ == "__main__"`-style top-level code runs normally).
runpy.run_path(os.path.join(ROOT, "main", "main.py"), run_name="__main__")
