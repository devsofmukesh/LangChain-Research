# =========================
# Load Dependencies
# =========================

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from utils import format_box

# =================================
# Environment & Model Configuration
# =================================

# Load environment variables from .env file
load_dotenv()

# Initialize the LLM model
llm = ChatOllama(model="gemma4:e4b")

# ========================
# MCP Server Configuration
# ========================

MATH_SERVER_PATH = Path(__file__).resolve().parent / "servers" / "math-server.py"

# ========================
# Main Function
# ========================


async def main():

    # Start the MCP client (launches the server over stdio) and discover its tools
    async with MCPAdapter(target=MATH_SERVER_PATH) as adapter:
        print(format_box("MCP Client Session Initialized. Loading Tools...", width=80))

        # Load tools from the MCP server as LangChain tools
        tools = await adapter.list_tools()

        print(format_box(f"Loaded {len(tools)} tools from MCP session.", width=80))

        # Create an agent with the loaded tools and the LLM model
        agent = create_agent(
            model=llm, tools=tools, system_prompt="You are a helpful assistant."
        )

        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="What is 54 + 2 * 3?")]}
        )

        # Print the result of the agent's response in a formatted box
        print(format_box(result["messages"][-1].content, width=80))


# ========================
# Entry Point
# ========================

if __name__ == "__main__":
    # Run the main function asynchronously
    asyncio.run(main())
