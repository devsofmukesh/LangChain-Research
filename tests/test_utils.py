from datetime import datetime, timedelta

import pytest

from utils import format_box, format_time


@pytest.mark.parametrize(
    ("text", "width", "expected"),
    [
        ("hello", 10, "+-------+\n| hello |\n+-------+"),
        ("🌍", 10, "+----+\n| 🌍 |\n+----+"),
    ],
)
def test_format_box(text: str, width: int, expected: str) -> None:
    assert format_box(text, width=width) == expected


def test_format_box_preserves_blank_lines() -> None:
    boxed = format_box("first\n\nsecond", width=6)

    lines = boxed.splitlines()
    assert lines[0] == lines[-1]
    assert len(lines) == 5
    assert lines[2] == "|        |"
    assert lines[1].startswith("| first")
    assert lines[3].startswith("| second")


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [
        (timedelta(0), "00:00:00 (HH:MM:SS)"),
        (timedelta(hours=1, minutes=2, seconds=3), "01:02:03 (HH:MM:SS)"),
        (timedelta(days=1, seconds=61), "24:01:01 (HH:MM:SS)"),
    ],
)
def test_format_time(elapsed: timedelta, expected: str) -> None:
    start = datetime(2024, 1, 1)

    assert format_time(start, start + elapsed) == expected
