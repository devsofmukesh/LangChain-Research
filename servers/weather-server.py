# =========================
# Load Dependencies
# =========================

from mcp.server.mcpserver import MCPServer

# =========================
# MCP Server Configuration
# =========================
mcp = MCPServer(
    name="Weather",
    description="A simple weather server that provides weather information for a given location.",
)


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
