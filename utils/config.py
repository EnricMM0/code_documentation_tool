import builtins
import os

import dotenv

dotenv.load_dotenv()

# ── API ────────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Language system prompts ────────────────────────────────────────────────────
LANGUAGE_SYSTEM_PROMPTS = {
    "py":      "You are an expert Python documentation generator.",
    "js":      "You are an expert JavaScript/TypeScript documentation generator.",
    "c":       "You are an expert C documentation generator.",
    "default": "You are an expert code documentation generator.",
}

# ── Parser / pipeline settings ─────────────────────────────────────────────────
# Extensions that receive per-function/block documentation (vs. a plain summary)
STRUCTURED_EXTENSIONS = {"py", "js", "c"}

# Python built-in names (used by the call-graph visitor to filter noise)
BUILTINS = set(dir(builtins))

# Maximum call-graph edges rendered in either single-file or folder mode
CALL_GRAPH_DISPLAY_LIMIT = 100

# Directories skipped when walking a project folder
IGNORE_DIRS = {
    "test", "tests", "node_modules", "__pycache__",
    ".git", ".venv", "venv", "dist", "build",
}
