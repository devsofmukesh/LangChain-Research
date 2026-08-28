# =========================
# Load Dependencies
# =========================

import textwrap
from wcwidth import wcswidth

# =========================
# Custom Utility Functions
# =========================

def format_box(text: str, width: int = 100) -> str:
    """
    Wrap text and render it inside a terminal-safe bordered box.
    Handles emojis and other Unicode characters correctly.
    """

    # Initialize a list to hold the wrapped lines of text
    wrapped_lines = []

    # Loop through each line in the input text
    for line in str(text).split("\n"):
        if not line:
            wrapped_lines.append("")
            continue

        # Wrap based on approximate character count
        wrapped_lines.extend(textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False) or [""])

    # Find actual terminal display width
    box_width = min(max(wcswidth(line) for line in wrapped_lines), width) if wrapped_lines else 0
    top = "+" + "-" * (box_width + 2) + "+"

    # Initialize a list to hold the body lines of the box
    body_lines = []

    # Loop through each wrapped line and pad it to fit the box width
    for line in wrapped_lines:

        # Calculate the display width of the line considering Unicode characters
        display_width = wcswidth(line)

        # Padding based on actual terminal width
        padding = box_width - display_width

        # Append the line with padding to the body lines
        body_lines.append(f"| {line}{' ' * padding} |")

    # Join the top, body, and bottom of the box
    body = "\n".join(body_lines)

    # Return the complete box as a string
    return f"{top}\n{body}\n{top}"

def format_time(start_time, end_time) -> str:
    """Calculate and format elapsed time between start_time and end_time as HH:MM:SS."""
    total_seconds = int((end_time - start_time).total_seconds())
    hh, remainder = divmod(total_seconds, 3600)
    mm, ss        = divmod(remainder, 60)
    time_taken    = f"{hh:02}:{mm:02}:{ss:02}"
    return f"{time_taken} (HH:MM:SS)"