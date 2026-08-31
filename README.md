# Agentic RAG with LangGraph

A **self-correcting, adaptive Retrieval-Augmented Generation (RAG)** pipeline built with [LangGraph](https://langchain-ai.github.io/langgraph/), using local models via [Ollama](https://ollama.com/), [Chroma](https://www.trychroma.com/) as the vector store, and [Tavily](https://tavily.com/) as a web-search fallback.

Instead of a linear "retrieve → generate" chain, this project models RAG as a **stateful graph**. An LLM router first decides whether a question even belongs in the vector store; retrieved documents are graded for relevance before generation; and the generated answer itself is graded for hallucinations and question-relevance, looping back to retry or re-search when it falls short.

---

## How the Workflow Works

The graph is defined in `graph/graph.py` and orchestrates four nodes, one routing decision at the entry point, and two grading gates around a shared `GraphState`.

```
                    ┌───────────────┐
                    │ ROUTE_QUESTION│
                    └───────┬───────┘
                (route_question decision)
              ┌─────────────┴─────────────┐
              ▼                           ▼
        ┌──────────┐                ┌───────────┐
        │ RETRIEVE │                │ WEBSEARCH │◄────────────┐
        └────┬─────┘                └─────┬─────┘             │
             ▼                            │                   │
    ┌──────────────────┐                  │                   │
    │ GRADE_DOCUMENTS   │                 │                   │
    └────────┬──────────┘                 │                   │
     (decide_to_generate)                 │                   │
     ┌───────┴────────┐                   │                   │
     ▼                ▼                   │                   │
┌───────────┐    ┌───────────┐            │                   │
│ WEBSEARCH │───►│ GENERATE  │◄───────────┘                   │
└───────────┘    └─────┬─────┘                                │
                        ▼                                     │
      (grade_generation_grounded_in_documents_and_question)   │
       ┌────────────────┼────────────────┐                    │
       ▼                ▼                ▼                    │
 "unsupported"       "useful"       "not useful"──────────────┘
   (retry GENERATE)     │
       │                ▼
       └──────────►     END
```

**Nodes**

| Node | File | Responsibility |
|---|---|---|
| `retrieve` | `graph/nodes/retrieve.py` | Pulls relevant chunks from the Chroma vector store for the incoming question. |
| `grade_documents` | `graph/nodes/grade_documents.py` | Uses an LLM-based binary relevance grader (`graph/chains/retrieval_grader.py`) to filter out irrelevant documents and flags whether a web search is needed. |
| `websearch` | `graph/nodes/web_search.py` | Runs a live Tavily search when retrieved documents aren't sufficient (or when the router sends the question straight to the web), and appends the results as an additional document. |
| `generate` | `graph/nodes/generate.py` | Produces the final answer via `graph/chains/generation.py`, grounded in the (possibly web-augmented) document context. |

**Routing & grading chains**

| Chain | File | Responsibility |
|---|---|---|
| `question_router` | `graph/chains/router.py` | Structured-output LLM classifier that routes a question to `"vectorstore"` or `"websearch"` at the graph's **entry point**, based on whether the question matches the vector store's known topics (agents, prompt engineering, adversarial attacks on LLMs). |
| `retrieval_grader` | `graph/chains/retrieval_grader.py` | Binary relevance grader used inside `grade_documents` for each retrieved document. |
| `hallucination_grader` | `graph/chains/hallucination_grader.py` | Binary grader checking whether the generated answer is grounded in / supported by the retrieved documents. |
| `answer_grader` | `graph/chains/answer_grader.py` | Binary grader checking whether the generated answer actually resolves the user's question. |

**Conditional routing**

- **`route_question`** (graph entry point, in `graph/graph.py`): calls `question_router` and sends the question to `WEBSEARCH` or `RETRIEVE` before anything else has run.
- **`decide_to_generate`** (after `GRADE_DOCUMENTS`): checks the `web_search` flag in `GraphState` — if any document was irrelevant, it routes to `WEBSEARCH` before `GENERATE`; otherwise it goes straight to `GENERATE`.
- **`grade_generation_grounded_in_documents_and_question`** (after `GENERATE`): runs `hallucination_grader` then `answer_grader` in sequence:
  - `"unsupported"` — the answer isn't grounded in the documents → loop back to `GENERATE` and retry.
  - `"not useful"` — the answer is grounded but doesn't address the question → route back to `WEBSEARCH` for more context.
  - `"useful"` — the answer is grounded and answers the question → `END`.

**State** (`graph/state.py`) is a `TypedDict` carrying:
- `question` — the user's question
- `generation` — the LLM's final answer
- `web_search` — bool flag set by the document grader
- `documents` — list of retrieved/graded/web-augmented documents

**Models used**
- LLM: `qwen3.5:9b` via `ChatOllama` — shared across generation, document grading, question routing, hallucination grading, and answer grading (all structured output)
- Embeddings: `qwen3-embedding:8b` via `OllamaEmbeddings` (ingestion + retrieval)

---

## Project Structure

```
.
├── README.md                                          # Project overview, setup, and usage docs (this file)
├── flow.png                                           # Auto-generated Mermaid diagram of the compiled graph
├── graph/                                             # Core LangGraph package: state, nodes, chains, routing
│   ├── __init__.py                                    # Marks graph/ as a package
│   ├── chains/                                        # LLM-backed chains: routing + grading + generation
│   │   ├── __init__.py                                # Marks graph/chains/ as a package
│   │   ├── answer_grader.py                           # answer_grader — checks answer resolves the question
│   │   ├── generation.py                              # generation_chain (prompt | llm | parser)
│   │   ├── hallucination_grader.py                    # hallucination_grader — checks answer is grounded in documents
│   │   ├── retrieval_grader.py                        # retrieval_grader (structured output grader)
│   │   ├── router.py                                  # question_router — routes a question to vectorstore vs websearch
│   │   └── tests/                                     # Pytest suite for the chains above
│   │       ├── __init__.py                            # Marks graph/chains/tests/ as a package
│   │       └── test_chains.py                         # pytest tests for routing, grading, and generation chains
│   ├── consts.py                                      # Node name constants
│   ├── graph.py                                       # StateGraph definition, conditional entry point + edges, 
│   │                                                  # diagram export
│   ├── nodes/                                         # Graph node functions (one per pipeline step)
│   │   ├── __init__.py                                # Re-exports node functions for graph.py to import
│   │   ├── generate.py                                # generate node — produces the final answer
│   │   ├── grade_documents.py                         # grade_documents node — filters irrelevant retrieved docs
│   │   ├── retrieve.py                                # retrieve node — pulls chunks from the Chroma vector store
│   │   └── web_search.py                              # websearch node — Tavily fallback search
│   └── state.py                                       # GraphState TypedDict
├── ingestion.py                                       # Loads seed URLs, chunks them, builds/loads the Chroma store
├── main.py                                            # Entry point — runs the graph on a sample question
├── pyproject.toml                                     # Project metadata + dependencies (uv-managed)
├── requirements.txt                                   # Plain pip-installable dependency list (alt. to uv)
├── schemas.py                                         # Reflection / AnswerQuestion / ReviseAnswer pydantic schemas
│                                                      # (scaffolding for a future reflect-and-revise agent; not yet 
│                                                      # wired into the graph)
├── utils.py                                           # format_box() and format_time() helpers
└── uv.lock                                            # Locked dependency versions
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
          langchain-core grandalf python-dotenv wcwidth tqdm pydantic
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
1. Route the question to the vector store or straight to web search
2. Retrieve documents from the Chroma store (if routed there) and grade them for relevance
3. Fall back to a Tavily web search if documents are missing or insufficient
4. Generate an answer, then grade it for hallucinations and for actually answering the question — retrying generation or falling back to web search again if it doesn't pass
5. Pretty-print the final answer and source documents in bordered boxes (via `format_box`)

It also prints an ASCII/Mermaid representation of the graph and saves a visual diagram to `flow.png` (rendered via mermaid.ink with an ELK layout; a monkey-patch works around a `grandalf` crash on the graph's self-loop edges when rendering ASCII).

---

## Running Tests

Tests live in `graph/chains/tests/test_chains.py` and cover the router, retrieval grader, hallucination grader, and generation chain directly against the live retriever and Ollama models (no mocking), so Ollama must be running and the vector store must already be built.

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
- `test_hallucination_grader_answer_yes` — confirms a generation produced from retrieved documents is graded as grounded.
- `test_hallucination_grader_answer_no` — confirms an unrelated, fabricated generation is graded as not grounded.
- `test_router_to_vectorstore` — confirms an in-domain question (e.g. "agent memory") routes to `"vectorstore"`.
- `test_router_to_websearch` — confirms an out-of-domain question (e.g. "how to make pizza?") routes to `"websearch"`.

---

## Notes / Gotchas

- `graph/nodes/__init__.py` must explicitly re-export the node **functions** (`generate`, `grade_documents`, `retrieve`, `web_search`), not just the submodules — otherwise `graph/graph.py`'s `from graph.nodes import generate, ...` will bind to modules instead of callables and raise a `TypeError` when invoked as `generate(state)`.
- The Chroma store is persisted to `./.chroma` — delete this directory if you change the embedding model or want to re-ingest from scratch.
- `web_search` appends a single combined `Document` (all Tavily result contents joined) to the existing `documents` list rather than replacing it.
- The `"unsupported"` branch of `grade_generation_grounded_in_documents_and_question` loops back to `GENERATE` with **no retry limit** — a persistently ungrounded generation could loop indefinitely. Consider adding a retry counter to `GraphState` if this becomes an issue.
- `question_router`'s system prompt hard-codes the vector store's topic scope (agents, prompt engineering, adversarial attacks on LLMs) — update it if you ingest different seed URLs in `ingestion.py`.
- `schemas.py` (`Reflection`, `AnswerQuestion`, `ReviseAnswer`) is not yet consumed by the graph — it's scaffolding for a possible future reflect-and-revise node.