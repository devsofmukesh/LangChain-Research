# =========================
# Load Dependencies
# =========================

import textwrap

# =========================
# Custom Utility Functions
# =========================

def format_box(text: str, width: int = 100) -> str:
    """
    Wrap text to `width` characters and render it inside a bordered box.
    Multi-line input is supported — each line is wrapped independently.
    """
    wrapped_lines = []
    for line in str(text).split("\n"):
        if line == "":
            wrapped_lines.append("")
        else:
            wrapped_lines.extend(textwrap.wrap(line, width=width) or [""])

    box_width = min(max(len(l) for l in wrapped_lines), width) if wrapped_lines else 0
    top = "+" + "-" * (box_width + 2) + "+"
    body = "\n".join(f"| {line.ljust(box_width)} |" for line in wrapped_lines)
    return f"{top}\n{body}\n{top}"

def format_time(start_time, end_time) -> str:
    """Calculate and format elapsed time between start_time and end_time as HH:MM:SS."""
    total_seconds = int((end_time - start_time).total_seconds())
    hh, remainder = divmod(total_seconds, 3600)
    mm, ss        = divmod(remainder, 60)
    time_taken    = f"{hh:02}:{mm:02}:{ss:02}"
    return f"{time_taken} (HH:MM:SS)"