"""
parser_utils.py
---------------
Language-specific source parsers and code reconstructors.

Each parser returns (class_name | None, function_name, source) tuples.
Class-level blocks carry function_name == '__class__'.
Supported languages: Python (AST), JavaScript/TypeScript (regex), C (regex).
"""

import ast
import re
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════════════════
# Python
# ══════════════════════════════════════════════════════════════════════════════

def split_code_by_ast(code: str) -> list[tuple]:
    """
    Parse Python source into (class_name | None, function_name, source) tuples.
    Class blocks are tagged with function_name == '__class__'.
    """
    tree = ast.parse(code)
    lines = code.splitlines()
    blocks = []

    class ASTVisitor(ast.NodeVisitor):
        def __init__(self):
            self.current_class = None

        def visit_ClassDef(self, node):
            prev = self.current_class
            self.current_class = node.name
            class_code = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            blocks.append((node.name, "__class__", class_code))
            self.generic_visit(node)
            self.current_class = prev

        def visit_FunctionDef(self, node):
            func_code = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            blocks.append((self.current_class, node.name, func_code))

        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)

    ASTVisitor().visit(tree)
    return blocks


def group_blocks_by_class(split_blocks: list[tuple]) -> dict:
    """
    Group (class_name, func_name, code) tuples into a nested dict:
    { class_name_or_None: { "class_code": str|None, "functions": [(name, code)] } }
    """
    grouped = defaultdict(lambda: {"class_code": None, "functions": []})
    for cls, name, code in split_blocks:
        if name == "__class__":
            grouped[cls]["class_code"] = code
        else:
            grouped[cls]["functions"].append((name, code))
    return grouped


def rebuild_code_ast(original_code: str, edited_blocks: list[str]) -> str:
    """
    Merge LLM-edited blocks back into the original Python source using AST line
    ranges. Preserves imports, module-level statements, and comments outside
    function/class definitions.
    """
    tree = ast.parse(original_code)
    lines = original_code.splitlines()
    line_occupied = [False] * len(lines)
    replacements = []

    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start, end = node.lineno - 1, node.end_lineno
        edited_blk = next(
            (blk for blk in edited_blocks
             if any(node.name in ln for ln in blk.splitlines()[:5])),
            None,
        )
        if edited_blk:
            replacements.append((start, end, edited_blk))
            for i in range(start, end):
                line_occupied[i] = True

    output = []
    for i, line in enumerate(lines):
        if line_occupied[i]:
            first_line = next((s for s, e, _ in replacements if s <= i < e), None)
            if first_line == i:
                blk = next(blk for s, _, blk in replacements if s == first_line)
                output.append(blk.rstrip())
        else:
            output.append(line)

    return "\n".join(output)


def _rebuild_by_text(original_code: str, orig_blocks: list[tuple],
                     edited_blocks: list[str]) -> str:
    """
    Reconstruct source by replacing each parsed block with its LLM-edited
    counterpart, matched by function/class name in the first five lines.
    Only top-level entries (class blocks or free functions) are replaced
    directly; methods are already included in their class-level edited block.
    """
    result = original_code
    top_level = [
        (cls, name, src) for cls, name, src in orig_blocks
        if name == "__class__" or cls is None
    ]
    for cls, name, orig_src in top_level:
        search_name = cls if name == "__class__" else name
        edited = next(
            (blk for blk in edited_blocks
             if any(search_name in ln for ln in blk.splitlines()[:5])),
            None,
        )
        if edited and orig_src in result:
            result = result.replace(orig_src, edited.strip(), 1)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# JavaScript / TypeScript
# ══════════════════════════════════════════════════════════════════════════════

def split_js_blocks(code: str) -> list[tuple]:
    """
    Extract top-level functions and classes from JavaScript/TypeScript source.

    Returns (class_name | None, function_name, source) tuples.
    Handles function declarations, arrow functions, function expressions,
    prototype method assignments, class declarations, and export variants.
    Uses brace counting to find block end correctly.
    """
    blocks = []
    lines = code.splitlines()
    n = len(lines)

    func_pattern     = re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(")
    arrow_pattern    = re.compile(r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(.*?\)\s*=>")
    func_expr_pattern = re.compile(r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function\b")
    proto_pattern    = re.compile(r"^(?:\w+)\.prototype\.(\w+)\s*=\s*(?:async\s+)?function\b")
    class_pattern    = re.compile(r"^(?:export\s+)?class\s+(\w+)")
    proto_owner      = re.compile(r"^(\w+)\.prototype\.")

    def extract_block(start_line: int) -> tuple[str, int]:
        depth, started = 0, False
        for idx in range(start_line, n):
            for ch in lines[idx]:
                if ch == "{":
                    depth += 1
                    started = True
                elif ch == "}" and started:
                    depth -= 1
            if started and depth == 0:
                return "\n".join(lines[start_line : idx + 1]), idx + 1
        return lines[start_line], start_line + 1

    i = 0
    while i < n:
        stripped = lines[i].lstrip()

        m_class = class_pattern.match(stripped)
        if m_class:
            class_name = m_class.group(1)
            block_src, end_i = extract_block(i)
            blocks.append((class_name, "__class__", block_src))
            SKIP_METHOD_NAMES = {"constructor", "if", "for", "while", "switch", "catch"}
            method_pat = re.compile(r"^\s+(?:static\s+)?(?:async\s+)?(?:get\s+|set\s+)?(\w+)\s*\(")
            for j, src_line in enumerate(lines[i : end_i], start=i):
                mm = method_pat.match(src_line)
                if mm and mm.group(1) not in SKIP_METHOD_NAMES:
                    method_src, _ = extract_block(j)
                    blocks.append((class_name, mm.group(1), method_src))
            i = end_i
            continue

        m_proto = proto_pattern.match(stripped)
        if m_proto:
            method_name = m_proto.group(1)
            owner_m = proto_owner.match(stripped)
            owner = owner_m.group(1) if owner_m else None
            if "{" in "\n".join(lines[i : i + 3]):
                block_src, end_i = extract_block(i)
                blocks.append((owner, method_name, block_src))
                i = end_i
                continue

        m_func = func_pattern.match(stripped)
        if m_func:
            block_src, end_i = extract_block(i)
            blocks.append((None, m_func.group(1), block_src))
            i = end_i
            continue

        m_func_expr = func_expr_pattern.match(stripped)
        if m_func_expr:
            if "{" in "\n".join(lines[i : i + 3]):
                block_src, end_i = extract_block(i)
                blocks.append((None, m_func_expr.group(1), block_src))
                i = end_i
                continue

        m_arrow = arrow_pattern.match(stripped)
        if m_arrow:
            block_src, end_i = extract_block(i)
            blocks.append((None, m_arrow.group(1), block_src))
            i = end_i
            continue

        i += 1

    return blocks


def rebuild_code_js(original_code: str, edited_blocks: list[str]) -> str:
    """Reconstruct an edited JavaScript/TypeScript file from LLM-annotated blocks."""
    return _rebuild_by_text(original_code, split_js_blocks(original_code), edited_blocks)


# ══════════════════════════════════════════════════════════════════════════════
# C
# ══════════════════════════════════════════════════════════════════════════════

def split_c_blocks(code: str) -> list[tuple]:
    """
    Extract top-level function definitions from C source code.

    Handles pointer return types, qualifier prefixes, and multi-word type names.
    Skips preprocessor directives, forward declarations, and control-flow keywords.
    Returns (None, function_name, source) tuples (C has no classes).
    """
    blocks = []
    lines = code.splitlines()
    n = len(lines)

    SKIP_KEYWORDS = {"if", "else", "for", "while", "do", "switch",
                     "struct", "enum", "union", "typedef"}

    func_start = re.compile(
        r"^(?:(?:static|inline|extern|const|unsigned|signed|volatile)\s+)*"
        r"(?:[\w\s]+?)\s+"
        r"(\*+\s*)?(\w+)\s*\("
    )

    def extract_block(start: int) -> tuple[str, int]:
        depth, started = 0, False
        for idx in range(start, n):
            for ch in lines[idx]:
                if ch == "{":
                    depth += 1
                    started = True
                elif ch == "}" and started:
                    depth -= 1
            if started and depth == 0:
                return "\n".join(lines[start : idx + 1]), idx + 1
        return lines[start], start + 1

    i = 0
    while i < n:
        stripped = lines[i].strip()
        if (not stripped
                or stripped.startswith("#")
                or stripped.startswith("//")
                or stripped.endswith(";")):
            i += 1
            continue

        m = func_start.match(stripped)
        if m:
            func_name = m.group(2)
            if func_name in SKIP_KEYWORDS:
                i += 1
                continue
            # Only treat as a definition if '{' appears within 5 lines
            if "{" not in "\n".join(lines[i : i + 5]):
                i += 1
                continue
            block_src, end_i = extract_block(i)
            blocks.append((None, func_name, block_src))
            i = end_i
            continue

        i += 1

    return blocks


def rebuild_code_c(original_code: str, edited_blocks: list[str]) -> str:
    """Reconstruct an edited C source file from LLM-annotated blocks."""
    return _rebuild_by_text(original_code, split_c_blocks(original_code), edited_blocks)


# ══════════════════════════════════════════════════════════════════════════════
# Language dispatch
# ══════════════════════════════════════════════════════════════════════════════

def split_blocks(code: str, ext: str) -> list[tuple]:
    """Route source code to the appropriate parser for the given extension."""
    if ext == "py":
        return split_code_by_ast(code)
    if ext == "js":
        return split_js_blocks(code)
    if ext == "c":
        return split_c_blocks(code)
    return []


def rebuild_code(original_code: str, edited_blocks: list[str], ext: str) -> str:
    """Dispatch to the appropriate reconstructor for the given file extension."""
    if ext == "py":
        return rebuild_code_ast(original_code, edited_blocks)
    if ext == "js":
        return rebuild_code_js(original_code, edited_blocks)
    if ext == "c":
        return rebuild_code_c(original_code, edited_blocks)
    return "\n\n".join(edited_blocks)
