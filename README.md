# LangChain-Research

A self-reflective research agent built with LangChain and LangGraph. Given a question, it drafts an initial answer, critiques itself, searches the web to fill in gaps, and produces a revised, referenced answer — looping through the search-and-revise cycle a configurable number of times.

## How it works

1. **Draft** — the model writes an initial ~250 word answer, along with a self-critique and a list of search queries that would improve it.
2. **Execute Tools** — those search queries are run against Tavily's search API.
3. **Revise** — the model rewrites the answer using the search results, improving accuracy and adding references.
4. Steps 2–3 repeat until `MAX_ITERATIONS` is reached, then the final answer is printed.

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [Ollama](https://ollama.com/) running locally with a pulled model (default: `qwen3.5:9b`)
- A [Tavily](https://tavily.com/) API key

## Setup

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd LangChain-Research
```

### 2. Install dependencies

Using `uv` (recommended, matches the project's `pyproject.toml` / `uv.lock`):

```bash
uv sync
```

Or using pip with `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set up Ollama

Install [Ollama](https://ollama.com/download), then pull the model used in `chains.py`:

```bash
ollama pull qwen3.5:9b
```

Make sure Ollama is running (`ollama serve`, or it may already run as a background service after install).

### 4. Configure environment variables

Create a `.env` file in the project root:

```bash
TAVILY_API_KEY=your_tavily_api_key_here
```

You can get a Tavily API key from [tavily.com](https://tavily.com/).

## Usage

Run the agent with:

```bash
uv run main.py
```

or, if using a virtualenv/pip setup:

```bash
python main.py
```

This will:
- Print an ASCII and Mermaid diagram of the graph to the terminal
- Save a visual diagram of the workflow to `flow.png`
- Run the agent on the example question hardcoded in `main.py`
- Print the final revised answer

### Changing the question

Edit the `content` field in the `graph.invoke(...)` call at the bottom of `main.py`:

```python
response = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Your question here",
            }
        ]
    }
)
```

### Adjusting the number of revision loops

Change `MAX_ITERATIONS` at the top of `main.py` (default is `2`).

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Builds and runs the LangGraph workflow (draft → search → revise loop) |
| `chains.py` | Sets up the Ollama model and prompt templates for drafting and revising answers |
| `schemas.py` | Pydantic schemas defining the structured output (answer, critique, search queries, references) |
| `executor.py` | Runs the Tavily web searches requested during drafting/revision |
| `utils.py` | Small helpers for formatted terminal output (bordered text boxes, elapsed time) |
| `pyproject.toml` / `uv.lock` | Project metadata and locked dependencies (for `uv`) |
| `requirements.txt` | Alternative dependency list (for `pip`) |
| `flow.png` | Auto-generated diagram of the graph, regenerated each run |

## Notes

- The agent stops looping once the number of tool (search) messages exceeds `MAX_ITERATIONS`.
- Swap `ChatOllama` in `chains.py` for another provider (e.g. `langchain-openai`, `langchain-google-genai`) if you'd rather not run a local model.