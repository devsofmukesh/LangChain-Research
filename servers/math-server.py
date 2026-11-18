# =========================
# Load Dependencies
# =========================

from mcp.server.mcpserver import MCPServer

# =========================
# MCP Server Configuration
# =========================
mcp = MCPServer(
    name="Math-Server",
    description="A simple math server that provides basic arithmetic operations.",
)

# ========================
# MCP Tool Definitions
# ========================


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


# ========================
# Main Function
# ========================

if __name__ == "__main__":
    # Run the MCP server with stdio transport
    mcp.run(transport="stdio")
