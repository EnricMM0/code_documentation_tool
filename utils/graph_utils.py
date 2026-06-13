"""
graph_utils.py
--------------
Call-graph extraction (Python AST, C regex, JavaScript regex) and rendering
(text tree + interactive vis-network HTML widget).
"""

import ast
import json
import re
from collections import defaultdict

from graphviz import Digraph


# ── Python AST helpers ─────────────────────────────────────────────────────────

def _get_defined_names(tree: ast.AST) -> set[str]:
    """Return all function and class names defined in a Python AST."""
    return {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


class CallGraphVisitor(ast.NodeVisitor):
    """
    Collect (caller, callee) edges by walking a Python AST.
    Only records calls to user-defined names, filtering stdlib/builtin/third-party.
    """

    def __init__(self, defined_names: set[str]):
        self.current_function = None
        self.edges: list[tuple] = []
        self.defined_names = defined_names

    def visit_FunctionDef(self, node):
        prev = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = prev

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        if self.current_function is None:
            self.generic_visit(node)
            return
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        if func_name and func_name in self.defined_names:
            self.edges.append((self.current_function, func_name))
        self.generic_visit(node)


# ── Graph structure helpers ────────────────────────────────────────────────────

def build_call_graph(edges: list[tuple]) -> dict:
    graph = defaultdict(list)
    for caller, callee in edges:
        graph[caller].append(callee)
    return graph


def find_roots(edges: list[tuple]) -> list[str]:
    callers = {c for c, _ in edges}
    callees = {c for _, c in edges}
    return list(callers - callees)


def render_call_tree(graph: dict, roots: list[str],
                     indent: int = 0, visited: set | None = None) -> list[str]:
    if visited is None:
        visited = set()
    lines = []
    for root in roots:
        lines.append("  " * indent + root)
        if root in visited:
            continue
        visited.add(root)
        for child in graph.get(root, []):
            lines.extend(render_call_tree(graph, [child], indent + 1, visited))
    return lines


def merge_call_graphs(edge_lists: list[list]) -> list[tuple]:
    return list({edge for edges in edge_lists for edge in edges})


def prune_trivial_nodes(edges: list[tuple], min_degree: int = 2) -> list[tuple]:
    in_deg: dict = defaultdict(int)
    out_deg: dict = defaultdict(int)
    for src, dst in edges:
        out_deg[src] += 1
        in_deg[dst] += 1
    return [
        (src, dst) for src, dst in edges
        if (in_deg[src] + out_deg[src] >= min_degree
            and in_deg[dst] + out_deg[dst] >= min_degree)
    ]


def collapse_by_module(edges: list[tuple]) -> list[tuple]:
    return list({
        (src.split(".")[0], dst.split(".")[0])
        for src, dst in edges
        if src.split(".")[0] != dst.split(".")[0]
    })


# ── Language-specific edge extraction ─────────────────────────────────────────

def extract_call_graph_c(blocks: list[tuple]) -> list[tuple]:
    """Extract (caller, callee) edges from parsed C blocks."""
    defined = {name for _, name, _ in blocks}
    SKIP = {
        "if", "for", "while", "switch", "do", "sizeof", "typeof",
        "return", "printf", "fprintf", "sprintf", "malloc", "calloc",
        "realloc", "free", "memset", "memcpy", "memmove", "strlen",
        "strcmp", "strcpy", "strcat", "fopen", "fclose", "exit", "abort",
    }
    call_pat = re.compile(r"\b(\w+)\s*\(")
    edges: set[tuple] = set()
    for _, func_name, src in blocks:
        body_start = src.find("{")
        body = src[body_start:] if body_start != -1 else src
        for m in call_pat.finditer(body):
            callee = m.group(1)
            if callee in defined and callee != func_name and callee not in SKIP:
                edges.add((func_name, callee))
    return list(edges)


def extract_call_graph_js(blocks: list[tuple]) -> list[tuple]:
    """Extract (caller, callee) edges from parsed JavaScript/TypeScript blocks."""
    defined = {name for _, name, _ in blocks if name != "__class__"}
    SKIP = {
        "if", "for", "while", "switch", "catch", "new", "return", "throw",
        "typeof", "instanceof", "void", "delete", "await", "yield",
        "console", "setTimeout", "setInterval", "clearTimeout", "clearInterval",
        "Promise", "Array", "Object", "String", "Number", "Boolean", "JSON",
        "Math", "Date", "Map", "Set", "Error", "fetch", "require", "exports",
        "module", "document", "window", "process", "Symbol", "WeakMap", "WeakSet",
    }
    call_pat = re.compile(r"\b(\w+)\s*\(")
    edges: set[tuple] = set()
    for _, func_name, src in blocks:
        if func_name == "__class__":
            continue
        body_start = src.find("{")
        body = src[body_start:] if body_start != -1 else src
        for m in call_pat.finditer(body):
            callee = m.group(1)
            if callee in defined and callee != func_name and callee not in SKIP:
                edges.add((func_name, callee))
    return list(edges)


def extract_call_graph(code: str, ext: str, blocks: list[tuple]) -> list[tuple]:
    """Dispatch call-graph extraction to the appropriate language implementation."""
    if ext == "py":
        tree = ast.parse(code)
        defined_names = _get_defined_names(tree)
        visitor = CallGraphVisitor(defined_names)
        visitor.visit(tree)
        return visitor.edges
    if ext == "c":
        return extract_call_graph_c(blocks)
    if ext == "js":
        return extract_call_graph_js(blocks)
    return []


# ── Rendering ──────────────────────────────────────────────────────────────────

def render_tree_graph(edges: list[tuple]) -> Digraph:
    dot = Digraph(engine="dot")
    dot.attr(rankdir="TB", nodesep="0.6", ranksep="0.8")
    for parent, child in edges:
        dot.edge(parent, child)
    return dot


def render_interactive_graph(edges: list[tuple]) -> str:
    """
    Render a call graph as a fully interactive HTML widget using vis-network.
    Supports zoom, pan, and drag. Direction and spacing adapt automatically to
    graph shape. Node size encodes call depth (roots largest, leaves smallest).
    Returns an HTML string for st.components.v1.html().
    """
    node_ids   = sorted({n for edge in edges for n in edge})
    nodes_data = [{"id": n, "label": n} for n in node_ids]
    edges_data = [{"from": src, "to": dst, "arrows": "to"} for src, dst in edges]

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link href="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/dist/vis-network.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/dist/vis-network.min.js"></script>
<style>
  body {{ margin: 0; }}
  #graph {{
    width: 100%; height: 500px;
    border: 1px solid #e0e0e0; border-radius: 6px;
    background: #fafafa;
  }}
</style>
</head>
<body>
<div id="graph"></div>
<script>
  var nodesRaw = {json.dumps(nodes_data)};
  var edgesRaw = {json.dumps(edges_data)};

  // BFS depth from roots (in-degree 0 nodes).
  var inDeg = {{}}, children = {{}};
  nodesRaw.forEach(function(n) {{ inDeg[n.id] = 0; children[n.id] = []; }});
  edgesRaw.forEach(function(e) {{
    inDeg[e.to] = (inDeg[e.to] || 0) + 1;
    (children[e.from] = children[e.from] || []).push(e.to);
  }});
  var depth = {{}}, queue = [];
  nodesRaw.forEach(function(n) {{
    if (inDeg[n.id] === 0) {{ depth[n.id] = 0; queue.push(n.id); }}
  }});
  while (queue.length) {{
    var cur = queue.shift();
    (children[cur] || []).forEach(function(child) {{
      if (depth[child] === undefined) {{ depth[child] = depth[cur] + 1; queue.push(child); }}
    }});
  }}
  var maxDepth = 0;
  nodesRaw.forEach(function(n) {{
    if (depth[n.id] === undefined) depth[n.id] = 0;
    if (depth[n.id] > maxDepth) maxDepth = depth[n.id];
  }});

  // Count nodes per level to detect skewed graphs.
  var perLevel = {{}};
  nodesRaw.forEach(function(n) {{
    var d = depth[n.id];
    perLevel[d] = (perLevel[d] || 0) + 1;
  }});
  var maxWide  = Math.max.apply(null, Object.values(perLevel).concat([1]));
  var numLevels = maxDepth + 1;

  // Auto-direction: rotate to LR when graph is tall and thin.
  var direction = numLevels > maxWide * 1.8 ? 'LR' : 'UD';

  // Node spacing derived from label widths to prevent overlap.
  var maxLabelLen = Math.max.apply(null, nodesRaw.map(function(n) {{ return n.id.length; }}));
  var avgLabelLen = nodesRaw.reduce(function(s, n) {{ return s + n.id.length; }}, 0) / (nodesRaw.length || 1);
  var estMaxNodeW = maxLabelLen * 7 + 40;
  var estAvgNodeW = avgLabelLen * 7 + 40;
  var nodeSpacing = Math.max(estMaxNodeW, Math.round(560 / maxWide));
  var levelSep    = Math.max(70, Math.min(140, Math.round(560 / numLevels)));

  // Resize container to fit layout.
  var estPx = direction === 'UD'
    ? numLevels * levelSep + 80
    : maxWide   * (estAvgNodeW + nodeSpacing) + 80;
  document.getElementById('graph').style.height = Math.max(420, Math.min(780, estPx)) + 'px';

  // Depth → font size (roots largest, leaves smallest).
  nodesRaw.forEach(function(n) {{
    var d = depth[n.id];
    var fontSize = maxDepth === 0 ? 13 : Math.round(18 - (d / maxDepth) * 8);
    fontSize = Math.max(10, Math.min(18, fontSize));
    n.font = {{ size: fontSize, face: 'monospace' }};
    n.borderWidth = d === 0 ? 2.5 : 1;
    n.margin = d === 0 ? 10 : 6;
  }});

  var nodes   = new vis.DataSet(nodesRaw);
  var edges   = new vis.DataSet(edgesRaw);
  var network = new vis.Network(
    document.getElementById('graph'),
    {{ nodes: nodes, edges: edges }},
    {{
      layout: {{
        hierarchical: {{
          enabled: true,
          direction: direction,
          sortMethod: 'directed',
          nodeSpacing: nodeSpacing,
          levelSeparation: levelSep,
          treeSpacing: Math.round(nodeSpacing * 1.5),
          blockShifting: true,
          edgeMinimization: true,
          shakeTowards: 'roots',
        }},
      }},
      physics: {{ enabled: false }},
      interaction: {{ dragNodes: true, dragView: true, zoomView: true, hover: true }},
      nodes: {{
        shape: 'box',
        color: {{ background: '#dce8f5', border: '#3d7ab5',
                  highlight: {{ background: '#b8d4f0', border: '#1f4e79' }} }},
      }},
      edges: {{
        color: {{ color: '#3d7ab5', highlight: '#1f4e79' }},
        width: 1.5,
        arrows: {{ to: {{ enabled: true, scaleFactor: 0.6 }} }},
        smooth: {{ type: 'cubicBezier', forceDirection: direction === 'UD' ? 'vertical' : 'horizontal' }},
      }},
    }}
  );
  network.fit();
</script>
</body>
</html>"""
