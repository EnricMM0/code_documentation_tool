"""
main.py
-------
Streamlit entry point for the AI Code Documentation Generator.
Run with: streamlit run main.py
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
from streamlit.components.v1 import html

from utils.config import CALL_GRAPH_DISPLAY_LIMIT, IGNORE_DIRS, STRUCTURED_EXTENSIONS
from utils.graph_utils import (
    collapse_by_module,
    merge_call_graphs,
    prune_trivial_nodes,
    render_interactive_graph,
)
from utils.llm_utils import (
    combine_readme_llm,
    generate_example_usage_llm,
    generate_readme_llm,
    summarize_file_llm,
    summarize_non_code_file,
)
from utils.pipeline import build_markdown, process_structured_file
from utils.parser_utils import rebuild_code


def main():
    st.title("AI Code Documentation Generator")
    st.markdown(
        "Upload a code file or specify a project folder.  \n"
        "The app will use an LLM (Groq) to generate documentation of your code."
    )

    # ── API key guard ──────────────────────────────────────────────────────────
    from utils.config import GROQ_API_KEY
    if not GROQ_API_KEY:
        st.error(
            "GROQ_API_KEY not found. "
            "Create a `.env` file in this directory with:\n\n"
            "```\nGROQ_API_KEY=your_key_here\n```"
        )
        st.stop()

    # ── Options ────────────────────────────────────────────────────────────────
    model = st.selectbox(
        "Choose LLM Model",
        ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
    )

    input_type = st.radio("What do you want to document?", ["Single file", "Folder"])

    _ALLOWED_EXTS = {"py", "js", "c", "java", "cpp", "html", "css", "ipynb"}
    if input_type == "Single file":
        uploaded_file = st.file_uploader(
            "Upload a file",
            help="Supported: " + ", ".join(sorted(_ALLOWED_EXTS)).upper(),
        )
        if uploaded_file is not None:
            _ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
            if _ext not in _ALLOWED_EXTS:
                st.error(f"Unsupported file type: .{_ext}. "
                         f"Supported: {', '.join(sorted(_ALLOWED_EXTS))}")
                uploaded_file = None
        folder_path = None
        extensions  = None
    else:
        uploaded_file = None
        folder_path   = st.text_input("Enter a folder path", value=os.getcwd())
        ext_choice    = st.radio(
            "Process all files or select specific types?",
            ["All types", "Select types"],
        )
        all_exts = ["py", "js", "c", "java", "cpp", "html", "css",
                    "ipynb", "yaml", "yml", "csv", "json", "cfg", "ini",
                    "rst", "md"]
        extensions = (
            st.multiselect("Select file types to document", all_exts, default=["py"])
            if ext_choice == "Select types"
            else all_exts
        )

    generate_readme  = st.checkbox("Generate README.md", value=True)
    include_examples = False   # reserved for future toggle
    add_docstrings   = st.checkbox("Add docstrings per function/class")
    comment_code     = st.checkbox("Add inline comments")
    display_graph    = st.checkbox("Display call graph (Python, JavaScript, C)")

    if not (generate_readme or add_docstrings or comment_code):
        st.info("Enable at least one output option above, then click Generate.")
        return

    if not st.button("Generate Documentation"):
        return

    # ══════════════════════════════════════════════════════════════════════════
    # Single-file mode
    # ══════════════════════════════════════════════════════════════════════════
    if uploaded_file is not None:
        code = uploaded_file.read().decode("utf-8")
        ext  = uploaded_file.name.rsplit(".", 1)[-1].lower()
        doc_blocks, example_usage = [], ""

        if ext in STRUCTURED_EXTENSIONS:
            with st.spinner(f"Analysing {uploaded_file.name}…"):
                try:
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        result = process_structured_file(
                            code, ext, uploaded_file.name,
                            model, add_docstrings, comment_code, executor,
                        )
                except ValueError as e:
                    st.error(str(e))
                    return

            doc_blocks    = result["doc_blocks"]
            edited_blocks = result["edited_blocks"]
            tree_str      = result["tree_str"]
            edges         = result["call_graph_edges"]

            if edges and display_graph:
                st.subheader("Call Tree Visualization")
                html(render_interactive_graph(edges[:CALL_GRAPH_DISPLAY_LIMIT]), height=820)
                if len(edges) > CALL_GRAPH_DISPLAY_LIMIT:
                    st.caption(f"Showing {CALL_GRAPH_DISPLAY_LIMIT} of {len(edges)} edges.")

            if add_docstrings or comment_code:
                edited_code = rebuild_code(code, edited_blocks, ext)
                out_name = f"{uploaded_file.name.rsplit('.', 1)[0]}_edited.{ext}"
                with open(out_name, "w") as f:
                    f.write(edited_code)
                st.success(f"Annotated file saved as `{out_name}`")
                with st.expander("View annotated code", expanded=True):
                    st.code(edited_code, language=ext)
                full_documentation = edited_code

            elif not doc_blocks:
                with st.spinner(f"Summarising {uploaded_file.name}…"):
                    full_documentation = summarize_non_code_file(code, model=model)
                st.success(f"Summarised `{uploaded_file.name}` (no functions or classes found)")
                with st.expander("View summary", expanded=False):
                    st.markdown(full_documentation)

            else:
                func_names = [
                    m.group(1) for doc in doc_blocks
                    if (m := re.search(r"### (?:Function|Class): (\w+)", doc))
                ]
                example_usage = (
                    generate_example_usage_llm(func_names, model)
                    if include_examples else ""
                )
                full_documentation = build_markdown(tree_str, doc_blocks, example_usage)
                n_funcs   = sum(1 for b in doc_blocks if "### Function:" in b)
                n_classes = sum(1 for b in doc_blocks if "### Class:"    in b)
                parts = []
                if n_funcs:   parts.append(f"{n_funcs} function{'s' if n_funcs > 1 else ''}")
                if n_classes: parts.append(f"{n_classes} class{'es' if n_classes > 1 else ''}")
                st.success(f"Documented `{uploaded_file.name}` — {', '.join(parts)}")
                with st.expander("View documentation", expanded=False):
                    st.markdown(full_documentation)

        else:
            with st.spinner(f"Summarising {uploaded_file.name}…"):
                full_documentation = summarize_non_code_file(code, model=model)
            with st.expander(f"Summary — {uploaded_file.name}", expanded=True):
                st.markdown(full_documentation)

        if generate_readme:
            doc_for_readme = (
                build_markdown("", doc_blocks, example_usage)
                if not (add_docstrings or comment_code)
                   and ext in STRUCTURED_EXTENSIONS
                   and doc_blocks
                else full_documentation
            )
            with st.spinner("Writing README.md…"):
                generate_readme_llm(doc_for_readme, model=model)
            st.success("README.md saved")
            try:
                with open("README.md") as _f:
                    _readme_content = _f.read()
                with st.expander("View README.md", expanded=True):
                    st.markdown(_readme_content)
            except OSError:
                pass

    # ══════════════════════════════════════════════════════════════════════════
    # Folder mode
    # ══════════════════════════════════════════════════════════════════════════
    elif folder_path is not None and os.path.isdir(folder_path):
        st.info(f"Processing folder: {folder_path}")

        folder_path_norm = folder_path.rstrip("/\\")
        edited_root      = folder_path_norm + "_edited"

        def _walk_filtered(path):
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if d.lower() not in IGNORE_DIRS]
                yield root, files

        files_to_process = [
            os.path.join(root, fname)
            for root, files in _walk_filtered(folder_path)
            for fname in files
            if fname.rsplit(".", 1)[-1].lower() in extensions
        ]
        total_files = len(files_to_process)
        st.info(f"Found {total_files} files to process.")

        progress_bar  = st.progress(0)
        status_text   = st.empty()
        all_doc_blocks:  list = []
        all_call_graphs: list = []
        file_summaries:  dict = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            for i, file_path in enumerate(files_to_process, 1):
                ext = file_path.rsplit(".", 1)[-1].lower()
                status_text.text(f"Processing {file_path} ({i}/{total_files})")

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        code = f.read()
                except Exception as exc:
                    st.warning(f"Skipping {file_path}: {exc}")
                    progress_bar.progress(i / total_files)
                    continue

                if ext in STRUCTURED_EXTENSIONS:
                    try:
                        result = process_structured_file(
                            code, ext, file_path,
                            model, add_docstrings, comment_code, executor,
                        )
                    except ValueError as exc:
                        st.warning(str(exc))
                        progress_bar.progress(i / total_files)
                        continue

                    doc_blocks    = result["doc_blocks"]
                    edited_blocks = result["edited_blocks"]
                    tree_str      = result["tree_str"]
                    edges         = result["call_graph_edges"]

                    if edges:
                        module_name = os.path.splitext(os.path.basename(file_path))[0]
                        prefixed = [(f"{module_name}.{src}", f"{module_name}.{dst}")
                                    for src, dst in edges]
                        all_call_graphs.append(prefixed)

                    file_summaries[file_path] = (
                        f"**{os.path.basename(file_path)}**\n"
                        + summarize_file_llm(
                            "\n\n".join(doc_blocks or edited_blocks), model=model
                        )
                    )

                    if add_docstrings or comment_code:
                        edited_code = rebuild_code(code, edited_blocks, ext)
                        rel_path = os.path.relpath(file_path, folder_path)
                        out_name = os.path.join(edited_root, rel_path)
                        os.makedirs(os.path.dirname(out_name), exist_ok=True)
                        with open(out_name, "w", encoding="utf-8") as f:
                            f.write(edited_code)
                    else:
                        all_doc_blocks.extend(doc_blocks)

                else:
                    summary = summarize_non_code_file(code, model=model)
                    file_summaries[file_path] = (
                        f"**{os.path.basename(file_path)}**\n{summary}"
                    )

                progress_bar.progress(i / total_files)

        status_text.text("Folder processing complete!")
        if add_docstrings or comment_code:
            st.success(f"Annotated files saved to `{edited_root}/`")

        # ── Call graph ─────────────────────────────────────────────────────────
        if all_call_graphs:
            merged_edges = merge_call_graphs(all_call_graphs)
            safe_edges   = prune_trivial_nodes(merged_edges)
            safe_edges   = collapse_by_module(safe_edges)[:CALL_GRAPH_DISPLAY_LIMIT]
            if not safe_edges:
                safe_edges = prune_trivial_nodes(merged_edges)[:CALL_GRAPH_DISPLAY_LIMIT]
            if not safe_edges:
                safe_edges = merged_edges[:CALL_GRAPH_DISPLAY_LIMIT]

            if display_graph:
                if safe_edges:
                    st.subheader("Project Call Tree")
                    html(render_interactive_graph(safe_edges), height=820)
                    if len(safe_edges) > CALL_GRAPH_DISPLAY_LIMIT:
                        st.caption(
                            f"Showing {CALL_GRAPH_DISPLAY_LIMIT} of "
                            f"{len(safe_edges)} edges for readability."
                        )
                else:
                    st.info("No call graph edges found in the processed files.")
        else:
            safe_edges = []

        if not (add_docstrings or comment_code):
            func_names = [
                m.group(1) for doc in all_doc_blocks
                if (m := re.search(r"### (?:Function|Class): (\w+)", doc))
            ]
            example_usage      = (
                generate_example_usage_llm(func_names, model)
                if include_examples else ""
            )
            full_documentation = build_markdown("", all_doc_blocks, example_usage)
            out_path = os.path.join(folder_path, "project_documentation.md")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(full_documentation)
            st.success(f"Documentation written to {out_path}")

        if generate_readme:
            full_summary = "\n\n".join(file_summaries.values())
            combine_readme_llm(full_summary, "", model=model)
            st.success("README.md generated")
            try:
                with open("README.md") as _f:
                    _readme_content = _f.read()
                with st.expander("View README.md", expanded=True):
                    st.markdown(_readme_content)
            except OSError:
                pass

    else:
        st.warning("Please upload a file or specify a valid folder path.")


if __name__ == "__main__":
    main()
