from pathlib import Path
from runpy import run_path

import pytest


@pytest.fixture(scope="module")
def math_server_tools() -> dict[str, object]:
    server_path = Path(__file__).resolve().parents[1] / "servers" / "math-server.py"
    return run_path(str(server_path))


@pytest.mark.parametrize(
    ("tool_name", "a", "b", "expected"),
    [
        ("add", 54, 6, 60),
        ("multiply", 7, 8, 56),
    ],
)
def test_math_server_tools(
    math_server_tools: dict[str, object], tool_name: str, a: int, b: int, expected: int
) -> None:
    tool = math_server_tools[tool_name]

    assert callable(tool)
    assert tool(a, b) == expected
