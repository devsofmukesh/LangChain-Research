from pathlib import Path
from runpy import run_path
from typing import Any

import pytest


@pytest.fixture(scope="module")
def weather_server_tools() -> dict[str, Any]:
    server_path = Path(__file__).resolve().parents[1] / "servers" / "weather-server.py"
    return run_path(str(server_path))


@pytest.mark.asyncio
async def test_get_weather_returns_weather_description(
    weather_server_tools: dict[str, Any],
) -> None:
    get_weather = weather_server_tools["get_weather"]

    assert await get_weather("London") == "Hot as hell"
