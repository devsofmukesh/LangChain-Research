# Agentic RAG with LangGraph

A **self-correcting Retrieval-Augmented Generation (RAG)** pipeline built with [LangGraph](https://langchain-ai.github.io/langgraph/), using local models via [Ollama](https://ollama.com/), [Chroma](https://www.trychroma.com/) as the vector store, and [Tavily](https://tavily.com/) as a web-search fallback.

Instead of a linear "retrieve → generate" chain, this project models RAG as a **stateful graph**. Retrieved documents are graded for relevance before generation, and if any document is found irrelevant, the graph automatically falls back to a live web search to fill in the gaps — making the pipeline more robust to poor retrieval results.

---

## How the Workflow Works

The graph is defined in `graph/graph.py` and orchestrates four nodes around a shared `GraphState`:

```
        ┌──────────┐
        │ RETRIEVE │
        └────┬─────┘
             │
             ▼
    ┌──────────────────┐
    │ GRADE_DOCUMENTS   │
    └────────┬──────────┘
             │
     (decide_to_generate)
             │
     ┌───────┴────────┐
     │                 │
     ▼                 ▼
┌───────────┐    ┌───────────┐
│ WEBSEARCH │    │ GENERATE  │
└─────┬─────┘    └─────┬─────┘
      │                │
      └───────►────────┘
               │
               ▼
              END
```

**Nodes**

| Node | File | Responsibility |
|---|---|---|
| `retrieve` | `graph/nodes/retrieve.py` | Pulls relevant chunks from the Chroma vector store for the incoming question. |
| `grade_documents` | `graph/nodes/grade_documents.py` | Uses an LLM-based binary relevance grader (`graph/chains/retrieval_grader.py`) to filter out irrelevant documents and flags whether a web search is needed. |
| `websearch` | `graph/nodes/web_search.py` | Runs a live Tavily search when retrieved documents aren't sufficient, and appends the results as an additional document. |
| `generate` | `graph/nodes/generate.py` | Produces the final answer via `graph/chains/generation.py`, grounded in the (possibly web-augmented) document context. |

**Conditional routing** (`decide_to_generate` in `graph/graph.py`): after grading, the graph checks the `web_search` flag in `GraphState` — if any document was irrelevant, it routes to `WEBSEARCH` before `GENERATE`; otherwise it goes straight to `GENERATE`.

**State** (`graph/state.py`) is a `TypedDict` carrying:
- `question` — the user's question
- `generation` — the LLM's final answer
- `web_search` — bool flag set by the grader
- `documents` — list of retrieved/graded/web-augmented documents

**Models used**
- LLM: `qwen3.5:9b` via `ChatOllama` (generation + document grading, structured output)
- Embeddings: `qwen3-embedding:8b` via `OllamaEmbeddings` (ingestion + retrieval)

---

## Project Structure

```
.
├── main.py                          # Entry point — runs the graph on a sample question
├── ingestion.py                     # Loads seed URLs, chunks them, builds/loads the Chroma store
├── utils.py                         # format_box() and format_time() helpers
├── schemas.py                       # Shared pydantic schemas / structured-output models
├── flow.png                         # Auto-generated Mermaid diagram of the compiled graph
├── pyproject.toml                   # Project metadata + dependencies (uv-managed)
├── uv.lock                          # Locked dependency versions
├── requirements.txt                 # Plain pip-installable dependency list (alt. to uv)
├── .env                             # Environment variables (not committed)
├── graph/
│   ├── __init__.py
│   ├── state.py                     # GraphState TypedDict
│   ├── consts.py                    # Node name constants
│   ├── graph.py                     # StateGraph definition + conditional routing
│   ├── nodes/
│   │   ├── __init__.py              # Re-exports node functions
│   │   ├── retrieve.py
│   │   ├── grade_documents.py
│   │   ├── generate.py
│   │   └── web_search.py
│   └── chains/
│       ├── __init__.py
│       ├── generation.py            # generation_chain (prompt | llm | parser)
│       ├── retrieval_grader.py      # retrieval_grader (structured output grader)
│       └── tests/
│           ├── __init__.py
│           └── test_chains.py       # pytest tests for grading + generation chains
```

---

## Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package/dependency manager
- **[Ollama](https://ollama.com/)** installed and running locally, with the required models pulled:
  ```bash
  ollama pull qwen3.5:9b
  ollama pull qwen3-embedding:8b
  ```
- A **Tavily API key** (for the web-search fallback node) — get one at [tavily.com](https://tavily.com/)

---

## Setup with uv

1. **Install uv** (if you don't already have it):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repo and move into it**:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

3. **Create a virtual environment and install dependencies**:
   ```bash
   uv sync
   ```
   This creates a `.venv/` and installs everything pinned in `pyproject.toml` / `uv.lock`.

   If you're starting from scratch without a `pyproject.toml` yet, initialize one and add the core dependencies:
   ```bash
   uv init
   uv add langgraph langchain-chroma langchain-ollama langchain-unstructured \
          langchain-community langchain-text-splitters langchain-tavily \
          langchain-core python-dotenv wcwidth tqdm pydantic
   uv add --dev pytest
   ```

4. **Create your `.env` file** in the project root:
   ```env
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

5. **Make sure Ollama is running** in the background:
   ```bash
   ollama serve
   ```

6. **Build the vector store (first run only)**:
   In `ingestion.py`, uncomment the `Chroma.from_documents(...)` block so the seed URLs get embedded and persisted to `./.chroma`. Run it once:
   ```bash
   uv run ingestion.py
   ```
   Then re-comment that block for subsequent runs so you don't re-embed every time — the retriever below it will simply load the persisted `./.chroma` store.

---

## Running the App

Run the full agentic RAG graph on the sample question ("What is agent memory?") defined in `main.py`:

```bash
uv run main.py
```

This will:
1. Retrieve documents from the Chroma store
2. Grade them for relevance
3. Fall back to a Tavily web search if needed
4. Generate a final answer
5. Pretty-print the answer and source documents in bordered boxes (via `format_box`)

It also prints an ASCII/Mermaid representation of the graph and saves a visual diagram to `flow.png`.

---

## Running Tests

Tests live in `test_chains.py` and cover the retrieval grader and generation chain directly against the live retriever and Ollama models (no mocking), so Ollama must be running and the vector store must already be built.

Run the full test suite with:

```bash
uv run pytest
```

Run a specific test, with verbose output:

```bash
uv run pytest -v -k test_generation_chain
```

**Tests included:**
- `test_retrieval_grader_answer_yes` — confirms a relevant document is graded `"yes"` for a matching question.
- `test_retrieval_grader_answer_no` — confirms the same document is graded `"no"` against an unrelated question.
- `test_generation_chain` — runs the generation chain end-to-end and prints the answer.

---

## Notes / Gotchas

- `graph/nodes/__init__.py` must explicitly re-export the node **functions** (`generate`, `grade_documents`, `retrieve`, `web_search`), not just the submodules — otherwise `graph/graph.py`'s `from graph.nodes import generate, ...` will bind to modules instead of callables and raise a `TypeError` when invoked as `generate(state)`.
- The Chroma store is persisted to `./.chroma` — delete this directory if you change the embedding model or want to re-ingest from scratch.
- `web_search` appends a single combined `Document` (all Tavily result contents joined) to the existing `documents` list rather than replacing it.