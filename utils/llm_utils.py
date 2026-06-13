"""
llm_utils.py
------------
All LLM interactions: raw API call, documentation generation, summarisation,
and README generation. Every outbound request goes through _call_groq().
"""

import re

import requests

from .config import GROQ_API_KEY, LANGUAGE_SYSTEM_PROMPTS


# ── Core API call ──────────────────────────────────────────────────────────────

def _call_groq(messages: list, model: str, temperature: float = 0.1,
               max_tokens: int = 1000, timeout: int = 20) -> str:
    """Single entry-point for all Groq API calls."""
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ── Response cleaning ──────────────────────────────────────────────────────────

def clean_llm_code(code: str) -> str:
    """Strip markdown code fences that LLMs sometimes add to code output."""
    code = re.sub(r"^```(?:\w+)?\s*", "", code, flags=re.MULTILINE)
    code = re.sub(r"^```\s*",         "", code, flags=re.MULTILINE)
    return code.strip()


def _strip_doc_fence(block: str) -> str:
    """
    Remove code-fence wrappers the LLM sometimes adds around doc blocks.

    Handles two patterns:

    1. Fenced with duplicate header:
        ### Function: foo      ← header (kept)
        ```                    ← opening fence (removed)
        ### Function: foo      ← duplicate header (removed)
        description            ← content (kept)
        ```                    ← closing fence (removed)

    2. Plain description wrapped in a stray closing fence:
        ### Function: foo      ← header (kept)
        description            ← content (kept)
        ```                    ← stray closing fence (removed)
    """
    lines = block.splitlines()
    while lines and re.match(r"^`+\s*$", lines[-1].strip()):
        lines.pop()
    if not lines:
        return ""
    if len(lines) >= 2 and re.match(r"^```", lines[1].strip()):
        inner = lines[2:]
        if inner and inner[0].strip().startswith("### "):
            inner = inner[1:]
        lines = [lines[0]] + inner
    return "\n".join(lines).strip()


# ── Per-block documentation ────────────────────────────────────────────────────

def _doc_single_block(cls: str | None, name: str, code: str,
                      model: str, language: str) -> str:
    """
    Generate a one-line Markdown documentation entry for a single code block.
    Returns a '### Function/Class: label\\ndescription' string.
    """
    kind  = "Class" if name == "__class__" else "Function"
    label = f"{cls}.{name}" if cls and name != "__class__" else (cls or name)
    system_prompt = LANGUAGE_SYSTEM_PROMPTS.get(language, LANGUAGE_SYSTEM_PROMPTS["default"])
    prompt = (
        f"Write one sentence (max 25 words) describing what this {kind.lower()} does.\n"
        "Return ONLY the sentence — no headers, no markdown, no punctuation beyond the sentence.\n\n"
        f"```\n{code}\n```"
    )
    try:
        desc = _call_groq(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt},
            ],
            model=model,
            max_tokens=80,
            timeout=20,
        )
        return f"### {kind}: {label}\n{desc.strip()}"
    except Exception:
        return f"### {kind}: {label}"


def generate_file_docs(blocks: list[tuple], model: str = "llama-3.1-8b-instant",
                       language: str = "py") -> list[str]:
    """
    Generate Markdown documentation for all blocks in a file in a single LLM
    call. Returns a list of per-block doc strings.
    """
    system_prompt = LANGUAGE_SYSTEM_PROMPTS.get(language, LANGUAGE_SYSTEM_PROMPTS["default"])

    sections = []
    for cls, name, code in blocks:
        label = f"{cls}.{name}" if cls and name != "__class__" else (cls or name)
        sections.append(f"### {label}\n```\n{code}\n```")

    prompt = (
        "Generate Markdown documentation for each code block below.\n"
        "For EACH block, start with the exact header '### Function: <name>' "
        "(or '### Class: <name>' for classes).\n"
        "Max 40 words per block. No examples. No explanations outside the doc.\n"
        "Return ONLY plain Markdown text. Do NOT wrap any block in code fences.\n\n"
        + "\n\n".join(sections)
    )

    raw = _call_groq(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        model=model,
        max_tokens=4000,
        timeout=60,
    )

    raw_parts = re.findall(r"###[^\n]+(?:\n(?!###)[^\n]*)*", raw)

    seen: dict[str, str] = {}
    for p in raw_parts:
        p = _strip_doc_fence(p.strip())
        if not p:
            continue
        m = re.match(r"###[^:]+:\s+(\w[\w.]*)", p)
        key = m.group(1) if m else p[:40]
        if key not in seen or len(p) > len(seen[key]):
            seen[key] = p
    return list(seen.values())


def generate_func_def(class_name: str | None, code: str,
                      model: str = "llama-3.1-8b-instant",
                      add_docstrings: bool = False,
                      comment_code: bool = False,
                      language: str = "py") -> str:
    """Annotate a single code block with docstrings/comments (edit mode)."""
    system_prompt = LANGUAGE_SYSTEM_PROMPTS.get(language, LANGUAGE_SYSTEM_PROMPTS["default"])

    instructions = [
        f"You will receive {language.upper()} code.",
        "Return the SAME code, but edited.",
        "Do NOT remove any logic.",
        "Do NOT change function signatures.",
        f"Return ONLY valid {language.upper()} code, without markdown formatting.",
    ]
    if add_docstrings:
        instructions.append("Add concise docstrings/doc-comments to all functions and classes.")
    if comment_code:
        instructions.append("Add concise inline comments explaining non-obvious logic.")

    prompt = (
        f"{' | '.join(instructions)}\n\n"
        f"Return ONLY the result.\n\n"
        f"{'Class: ' + class_name + chr(10) if class_name else ''}"
        f"Code:\n{code}"
    )

    return _call_groq(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        model=model,
        max_tokens=3000,
        timeout=60,
    )


# ── Summarisation ──────────────────────────────────────────────────────────────

def generate_example_usage_llm(func_names: list[str],
                                model: str = "llama-3.1-8b-instant") -> str:
    prompt = (
        "Generate a short example showing how these functions might be used together.\n"
        "Do NOT explain anything.\n"
        "Return only a code block.\n\n"
        f"Functions:\n{', '.join(func_names[:8])}"
    )
    return _call_groq(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=120,
        timeout=10,
    )


def summarize_file_llm(file_content: str, model: str = "llama-3.1-8b-instant") -> str:
    prompt = (
        "Summarize this code documentation in 3-5 sentences.\n"
        "Focus on what the file does and its main components, "
        "mentioning briefly the main classes and functions.\n\n"
        f"Documentation:\n{file_content}"
    )
    return _call_groq(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        timeout=20,
    )


def summarize_non_code_file(content: str, model: str = "llama-3.1-8b-instant") -> str:
    prompt = (
        "Summarize this document in 3-5 sentences, "
        "focusing on its content and purpose.\n\n"
        f"Content:\n{content}"
    )
    return _call_groq(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        timeout=20,
    )


# ── README generation ──────────────────────────────────────────────────────────

def generate_readme_llm(full_documentation: str,
                        model: str = "llama-3.1-8b-instant") -> None:
    """Generate a single-file README.md from documentation."""
    prompt = (
        "Generate a concise README.md for this file. "
        "Start with a brief description of what the file does and its main purpose.\n"
        "Include a section listing the key functions and classes by name "
        "with a one-line description each — only those actually present in the documentation.\n"
        "Do not include any introduction to the task, just return the README content.\n"
        "No badges, no emojis. "
        "Do not add information that is not present (Contributing, License, etc.).\n\n"
        f"Documentation:\n{full_documentation}"
    )
    readme = _call_groq(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.1,
        max_tokens=2000,
        timeout=30,
    )
    with open("README.md", "w") as f:
        f.write(readme)


def combine_readme_llm(full_documentation: str, tree_str: str,
                       model: str = "llama-3.1-8b-instant") -> None:
    """Generate a project-level README.md from multi-file documentation."""
    tree_section = f"\n\nCall tree:\n{tree_str}" if tree_str else ""
    prompt = (
        "Given a description of the files present in this project, generate a concise README.md.\n"
        "No badges, no emojis. "
        "Do not add information that is not present (Links, Contributing, License, etc.).\n"
        "Do not list all functions — give a high-level overview and a brief example usage section.\n"
        "Only return the README.md content. Do not include any notes or explanations.\n\n"
        f"Documentation:\n{full_documentation}"
        + tree_section
    )
    readme = _call_groq(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=2000,
        timeout=30,
    )
    with open("README.md", "w") as f:
        f.write(readme)
