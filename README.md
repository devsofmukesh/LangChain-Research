# LangChain Agent with Ollama and MCP

A small project for exploring LangChain agents powered by local Ollama models and MCP tools. The main example uses the `gemma4:e4b` model, starts an MCP math server over standard input/output (stdio), discovers its tools, and asks the agent to solve an arithmetic question.

The math server provides two tools:

- `add` adds two integers.
- `multiply` multiplies two integers.

There is also a standalone weather MCP server example. It uses SSE transport and currently returns a placeholder response; the main agent does not connect to it.

## Requirements and Setup

- Python 3.14 (see `.python-version`; `uv` installs it automatically)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://ollama.com/) with the `gemma4:e4b` model

From the repository root, install the project dependencies:

```bash
uv sync
```

Download the model if needed:

```bash
ollama pull gemma4:e4b
```

Make sure the Ollama service is running before starting the agent.

## Run the Agent

From the repository root, run:

```bash
uv run main.py
```

The script starts the math MCP server automatically, loads its tools into the LangChain agent, and prints the response to the sample question `What is 54 + 2 * 3?`.

## Run the Weather Server Example

The weather server is a separate MCP server that uses SSE transport. Start it with:

```bash
uv run python servers/weather-server.py
```

Its `get_weather` tool currently returns the fixed placeholder text `Hot as hell`. The main agent does not start or connect to this server.

## Configuration

The current model, math server path, and sample question are set in `main.py`:

- Change the model in `ChatOllama(model="gemma4:e4b")`.
- Change the request passed to `agent.ainvoke(...)` to ask a different question.
- Change `MATH_SERVER_PATH` (passed to `MCPAdapter`) to connect the main example to a different stdio MCP server script. `MCPAdapter` also accepts an `http(s)` URL string for a remote server.

## Development and Tests

Install the test and lint tools with:

```bash
uv sync --group test
```

Run the unit tests:

```bash
make test
```

The tests cover the formatting helpers and the math and weather server tools. They do not start the Ollama model. The Makefile also provides `make lint` and `make format`. Ruff ignores the `N999` (invalid module name) rule for `servers/*.py` because those filenames contain hyphens.

## Project Structure

```text
.
├── .gitignore                  # Ignores virtual environments, caches, and local files
├── README.md                   # Setup, usage, development, and project overview
├── Makefile                    # Test, lint, and formatting shortcuts
├── main.py                     # LangChain agent and MCP client (MCPAdapter)
├── pyproject.toml              # Project metadata, dependencies, and tool settings
├── uv.lock                     # Locked dependency versions
├── utils.py                    # Terminal formatting and elapsed-time helpers
├── servers/                    # MCP server examples
│   ├── __init__.py             # Marks servers as a Python package
│   ├── math-server.py          # MCP math tools used by the main agent
│   └── weather-server.py       # Standalone MCP weather example using SSE
└── tests/                      # Automated unit tests
    ├── test_math_server.py     # Tests addition and multiplication tools
    ├── test_utils.py           # Tests box formatting and elapsed-time helpers
    └── test_weather_server.py  # Tests the async weather tool response
```

### Project Notes

- `main.py` starts `servers/math-server.py` as a subprocess and loads its tools through `langchain.mcp.MCPAdapter`, which is built into `langchain` (installed with the `langchain[mcp]` extra).
- Both servers use `MCPServer` from `mcp.server.mcpserver`. In `mcp` 2.x, `FastMCP` was renamed to `MCPServer`, so `mcp.server.fastmcp` no longer exists.
- `langchain.mcp` is in beta and emits a `LangChainBetaWarning` on import; its API may change in future releases. The older `langchain-mcp-adapters` package is no longer used because it requires `mcp<2`.
- Tool calling depends heavily on the model. In informal testing, `gemma4:e4b` made correct, sequential tool calls. `llama3.1:8b` and `qwen2.5-coder:14b` often sent invalid arguments or printed the tool call as plain text instead of running it. `qwen3.5:9b` works but is a thinking model whose final reply is occasionally empty; use `ChatOllama(model="qwen3.5:9b", reasoning=False)` with it.
- `servers/weather-server.py` is independent of the main agent and returns a placeholder response.
- `uv.lock` records the resolved dependencies used by `uv sync`; project metadata and dependency groups are declared in `pyproject.toml`.
