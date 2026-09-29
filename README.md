# mcpdoc

An MCP server that makes documentation listed in [`llms.txt`](https://llmstxt.org/) available to AI assistants through explicit, inspectable tools.

## About

`llms.txt` files provide an index of documentation pages for language models. MCP clients such as Cursor, Windsurf, Claude Desktop, and Claude Code can use this server to read a configured set of `llms.txt` sources and fetch relevant pages on demand.

The server exposes two tools:

- `list_doc_sources` lists the configured documentation sources.
- `fetch_docs` reads a configured local source or fetches an allowed HTTP(S) URL and returns its content as Markdown.

The intended workflow is to list the sources, read the relevant `llms.txt`, then fetch the documentation pages it links to. Remote pages are fetched over HTTP; local `llms.txt` files can also be configured.

## Requirements and Setup

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Network access when fetching remote documentation

From the repository root, install the project and its runtime dependencies:

```bash
uv sync
```

This creates the project environment and installs the `mcpdoc` command. Check the available command-line options with:

```bash
uv run mcpdoc --help
```

To install the development and test tools as well, sync the `test` dependency group:

```bash
uv sync --group test
```

## Run the Server

The default transport is `stdio`, which is the usual choice when an MCP client launches the server as a subprocess. This starts the server with the sample LangGraph Python documentation source:

```bash
uv run mcpdoc --yaml sample_config.yaml
```

For a local HTTP server using SSE transport, specify the host and port:

```bash
uv run mcpdoc \
  --urls "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt" \
  --transport sse \
  --host 127.0.0.1 \
  --port 8000
```

The CLI defaults to host `127.0.0.1` and port `8000` for SSE. `--host` and `--port` only apply when `--transport sse` is selected.

## Configuration

Provide documentation sources through a YAML file, a JSON file, or `--urls`. These input methods can be combined; their source lists are merged. Each configured source must have an `llms_txt` value and may have a display `name`.

### YAML

The repository's [`sample_config.yaml`](sample_config.yaml) is:

```yaml
- name: LangGraph Python
  llms_txt: https://langchain-ai.github.io/langgraph/llms.txt
```

Start the server with it using `uv run mcpdoc --yaml sample_config.yaml`.

### JSON

The [`sample_config.json`](sample_config.json) file contains the equivalent configuration:

```json
[
  {
    "name": "LangGraph Python",
    "llms_txt": "https://langchain-ai.github.io/langgraph/llms.txt"
  }
]
```

Load it with `uv run mcpdoc --json sample_config.json`.

### Command-Line Sources

Pass one or more URL or path entries after `--urls`. Use `name:url` to assign a display name; a name is optional.

```bash
uv run mcpdoc --urls \
  "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt" \
  "LangChain:https://python.langchain.com/llms.txt"
```

YAML, JSON, and command-line sources can be combined:

```bash
uv run mcpdoc \
  --yaml sample_config.yaml \
  --json sample_config.json \
  --urls "LangChain:https://python.langchain.com/llms.txt"
```

Local sources can be provided as a filesystem path or a `file://` URL. Relative paths are resolved from the process's working directory. For example:

```bash
uv run mcpdoc --urls "LocalDocs:/absolute/path/to/llms.txt"
```

### Fetch Settings and Domain Access

The server restricts remote fetches by domain:

- The origin of each configured remote `llms.txt` URL is allowed automatically.
- Add other documentation domains with `--allowed-domains`. For example:

  ```bash
  uv run mcpdoc \
    --urls "LangGraph:https://langchain-ai.github.io/langgraph/llms.txt" \
    --allowed-domains https://docs.example.com/
  ```

- Use `--allowed-domains '*'` to allow remote URLs from any domain. Only use this when that broader access is intended.
- For local sources, only the specific local files configured as sources can be read. If a local `llms.txt` links to remote pages, allow those pages' domains with `--allowed-domains`.

Other fetch options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--timeout SECONDS` | `10.0` | HTTP request timeout. |
| `--follow-redirects` | Off | Follow HTTP redirects; the server also checks HTML meta-refresh redirects when enabled. |
| `--transport` | `stdio` | MCP transport: `stdio` or `sse`. |
| `--host` | `127.0.0.1` | Bind address for SSE transport. |
| `--port` | `8000` | Port for SSE transport. |

Example with a longer timeout and redirects enabled:

```bash
uv run mcpdoc \
  --yaml sample_config.yaml \
  --follow-redirects \
  --timeout 15
```

## MCP Client Configuration

For a client that launches the server over `stdio`, point its MCP server command at this checkout. Replace the directory below with the absolute path to your clone; run `uv sync` in that directory first.

```json
{
  "mcpServers": {
    "mcpdoc": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/LangChain-Research",
        "mcpdoc",
        "--yaml",
        "sample_config.yaml"
      ]
    }
  }
}
```

MCP clients store this configuration in different locations. Add the entry to the client's MCP server configuration, then restart or reload the client if required. You can change the source by editing the YAML file or replacing `--yaml sample_config.yaml` with `--json sample_config.json` or `--urls` arguments.

## Programmatic Usage

The package also exposes `create_server` for Python callers:

```python
from mcpdoc.main import create_server

server = create_server(
    [
        {
            "name": "LangGraph Python",
            "llms_txt": "https://langchain-ai.github.io/langgraph/llms.txt",
        }
    ],
    timeout=15.0,
    follow_redirects=True,
)
server.run(transport="stdio")
```

`create_server` also accepts `allowed_domains` for additional remote domains and `settings` for MCP server constructor settings.

## Development

Install the `test` dependency group with `uv sync --group test`. The test suite is under `tests/`; run it with:

```bash
uv run pytest --disable-socket --allow-unix-socket
```

The Makefile also provides `make test`, `make lint`, and `make format` targets.

## Project Structure

```text
.                                # Repository root; run the commands in this README from this directory
├── README.md                    # Project overview, setup, configuration, and usage
├── Makefile                     # Test, lint, and formatting shortcuts
├── pyproject.toml               # Package metadata, dependencies, CLI entry point, and tool config
├── uv.lock                      # Locked dependency versions
├── sample_config.yaml           # Example YAML documentation source list
├── sample_config.json           # Example JSON documentation source list
├── mcpdoc/                      # Installable Python package
│   ├── __init__.py              # Package exports and version
│   ├── _version.py              # Installed package version lookup
│   ├── cli.py                   # mcpdoc command-line interface
│   ├── main.py                  # MCP server, tools, fetching, and access controls
│   ├── langgraph.py             # Standalone LangGraph documentation server example
│   └── splash.py                # CLI banner for SSE startup
└── tests/                       # Automated test suite
    └── unit_tests/              # Unit tests for package imports and helpers
        ├── __init__.py          # Marks unit_tests as a Python package
        ├── test_imports.py      # Package import checks
        └── test_main.py         # Tests for main module helpers
```

### Project Notes

- The `mcpdoc` command is registered in `pyproject.toml` and starts at `mcpdoc.cli:main`; `mcpdoc/main.py` creates the server and defines its tools.
- `mcpdoc/langgraph.py` is a standalone LangGraph-specific example. The `mcpdoc` CLI does not use it.
- `sample_config.yaml` and `sample_config.json` are equivalent examples. Edit either file to select the documentation sources for that run.
- `uv.lock` records the resolved dependency versions used by `uv sync`; package metadata and dependency groups are declared in `pyproject.toml`.
