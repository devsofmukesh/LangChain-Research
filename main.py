# =========================
# Load Dependencies
# =========================

import os
import asyncio
from utils import format_box
from mcp import ClientSession
from dotenv import load_dotenv
from mcp import StdioServerParameters
from langchain_ollama import ChatOllama
from mcp.client.stdio import stdio_client
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools

# =================================
# Environment & Model Configuration
# =================================

# Load environment variables from .env file
load_dotenv()

# Initialize the LLM model
llm = ChatOllama(model="llama3.1:8b")

# ========================
# MCP Server Configuration
# ========================

stdio_server_params = StdioServerParameters(
    command="python", 
    args=[os.path.join(os.path.dirname(os.path.abspath(__file__)), "servers", "math-server.py")]
)

# ========================
# Main Function
# ========================

async def main():

    # Start the MCP client and connect to the server
    async with stdio_client(stdio_server_params) as (read, write):

        # Initialize the MCP client session
        async with ClientSession(read_stream=read, write_stream=write) as session:

            # Initialize the session and load tools
            await session.initialize()

            # Print a formatted message indicating that the MCP client session has been initialized
            print(format_box("MCP Client Session Initialized. Loading Tools...", width=80))

            # Load tools from the MCP session
            tools = await load_mcp_tools(session)

            # Print a formatted message indicating that the tools have been loaded
            print(format_box(f"Loaded {len(tools)} tools from MCP session.", width=80))

            # Create an agent with the loaded tools and the LLM model
            agent = create_agent(model=llm, tools=tools, system_prompt="You are a helpful assistant.")

            # Print a formatted message indicating that the agent has been created
            result = await agent.ainvoke({"messages": [HumanMessage(content="What is 54 + 2 * 3?")]})

            # Print the result of the agent's response in a formatted box
            print(format_box(result["messages"][-1].content, width=80))

# ========================
# Entry Point
# ========================

if __name__ == "__main__":

    # Run the main function asynchronously
    asyncio.run(main())
