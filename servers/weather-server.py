# =========================
# Load Dependencies
# =========================

from mcp.server.fastmcp import FastMCP

# =========================
# MCP Server Configuration
# =========================
mcp = FastMCP("Weather")

# ========================
# MCP Tool Definitions
# ========================
@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    return "Hot as hell"

# ========================
# Main Function
# ========================
if __name__ == "__main__":

    # Run the MCP server with SSE transport
    mcp.run(transport="sse")