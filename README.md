# AI Code Documentation Generator

A Streamlit web app that uses a Groq-hosted LLM to automatically document code projects. Supports single files and entire folders, with optional docstring injection, inline comments, and an interactive call graph.

## Supported languages

Python, JavaScript / TypeScript, C — with structured per-function documentation and call-graph extraction. All other file types (HTML, CSS, JSON, Markdown, …) receive a plain-text summary.

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your Groq API key** — create a file `.env` in the root directory and fill in your key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Run**
   ```bash
   streamlit run main.py
   ```

## Features

| Option | Description |
|---|---|
| Generate README | Produces a `README.md` summarising the file or project |
| Add docstrings | Injects docstrings into every function and class |
| Add inline comments | Annotates non-obvious logic with inline comments |
| Call graph | Renders an interactive, hierarchical call graph |

## Project structure

```
doc_tool/
├── main.py              # Streamlit UI
├── requirements.txt
├── .env                 #To be added
└── utils/
    ├── config.py        # Constants and environment variables
    ├── llm_utils.py     # Groq API calls, generation, summarisation
    ├── parser_utils.py  # Language parsers and code reconstructors
    ├── graph_utils.py   # Call-graph extraction and vis-network renderer
    └── pipeline.py      # Per-file documentation pipeline
```
