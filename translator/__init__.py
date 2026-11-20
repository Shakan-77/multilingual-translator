import os
import sys
import tempfile
from pathlib import Path

# Windows consoles default to a legacy code page; print Unicode safely.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Keep every file this app writes (downloaded models, Gradio uploads, temp
# files) inside the project folder rather than the user's system drive.
# These folders are gitignored. A user-chosen HF_HOME / GRADIO_TEMP_DIR wins.
PROJECT_DIR = Path(__file__).resolve().parent.parent
_TMP_DIR = PROJECT_DIR / ".tmp"
_TMP_DIR.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(PROJECT_DIR / ".hf_cache"))
os.environ.setdefault("GRADIO_TEMP_DIR", str(_TMP_DIR / "gradio"))
os.environ.setdefault("MPLCONFIGDIR", str(_TMP_DIR / "matplotlib"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
tempfile.tempdir = str(_TMP_DIR)

from .engine import TranslationResult, Translator, split_sentences  # noqa: E402
from .languages import LANGUAGES, route  # noqa: E402

__all__ = ["LANGUAGES", "PROJECT_DIR", "TranslationResult", "Translator", "route", "split_sentences"]
