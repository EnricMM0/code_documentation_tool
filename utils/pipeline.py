"""
pipeline.py
-----------
Per-file documentation pipeline and output helpers.
Orchestrates parsing, LLM calls, and code reconstruction for a single file.
"""

import re
from concurrent.futures import ThreadPoolExecutor

from .graph_utils import (
    build_call_graph,
    extract_call_graph,
    find_roots,
    render_call_tree,
)
from .llm_utils import _doc_single_block, clean_llm_code, generate_func_def
from .parser_utils import group_blocks_by_class, rebuild_code, split_blocks


# ── Output helpers ─────────────────────────────────────────────────────────────

def build_markdown(tree: str, func_docs: list[str], example: str) -> str:
    md = ["# Project Documentation\n"]
    if tree:
        md += ["## Call Graph\n", "```", tree, "```\n"]
    md.append("## Functions\n")
    for doc in func_docs:
        md.append(doc.strip())
        md.append("\n")
    if example:
        md += ["## Example Usage\n", f"```\n{example.strip()}\n```\n"]
    return "\n".join(md)


# ── Per-file pipeline ──────────────────────────────────────────────────────────

def process_structured_file(code: str, ext: str, file_path_label: str,
                             model: str, add_docstrings: bool,
                             comment_code: bool,
                             executor: ThreadPoolExecutor) -> dict:
    """
    Full documentation pipeline for .py, .js, and .c files.

    Doc mode  → one focused LLM call per block via _doc_single_block() (parallel).
    Edit mode → one LLM call per block via generate_func_def() (parallel).
    Call graph is extracted for all supported languages.

    Returns a dict with keys: doc_blocks, edited_blocks, call_graph_edges, tree_str.
    Raises ValueError on parse failure so callers can display a warning and continue.
    """
    try:
        raw_blocks = split_blocks(code, ext)
    except SyntaxError as e:
        raise ValueError(f"Could not parse {ext} file '{file_path_label}': {e}") from e

    if not raw_blocks:
        return {
            "doc_blocks":       [],
            "edited_blocks":    [],
            "call_graph_edges": [],
            "tree_str":         "",
        }

    grouped = group_blocks_by_class(raw_blocks)

    # ── Call graph ────────────────────────────────────────────────────────────
    call_graph_edges: list = []
    tree_str = ""
    call_graph_edges = extract_call_graph(code, ext, raw_blocks)
    if call_graph_edges:
        graph_struct = build_call_graph(call_graph_edges)
        roots = find_roots(call_graph_edges)
        tree_str = "\n".join(render_call_tree(graph_struct, roots))

    # ── Documentation generation ──────────────────────────────────────────────
    if add_docstrings or comment_code:
        futures = []
        for class_name, data in grouped.items():
            class_code = data["class_code"]
            if class_code is None:
                for _func_name, func_code in data["functions"]:
                    futures.append(executor.submit(
                        generate_func_def,
                        None, func_code, model, add_docstrings, comment_code, ext,
                    ))
            else:
                futures.append(executor.submit(
                    generate_func_def,
                    class_name, class_code, model, add_docstrings, comment_code, ext,
                ))
        results = [clean_llm_code(f.result()) for f in futures]
        return {
            "doc_blocks":       [],
            "edited_blocks":    results,
            "call_graph_edges": call_graph_edges,
            "tree_str":         tree_str,
        }
    else:
        futures = [
            executor.submit(_doc_single_block, cls, name, code, model, ext)
            for cls, name, code in raw_blocks
        ]
        doc_blocks = [f.result() for f in futures]
        return {
            "doc_blocks":       doc_blocks,
            "edited_blocks":    [],
            "call_graph_edges": call_graph_edges,
            "tree_str":         tree_str,
        }
