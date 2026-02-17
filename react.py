# =========================
# Load Dependencies
# =========================
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_experimental.tools import PythonREPLTool

# =========================
# Environment Configuration
# =========================
load_dotenv()

# =========================
# Tool Definition and LLM Binding
# =========================
@tool
def triple(num: float) -> float:
    """Triples the input number."""
    return float(num) * 3

# Define tools to be used by the LLM
tools = [TavilySearch(max_results=1), triple]

# Bind the tools to the Ollama LLM
llm = ChatOllama(model="llama3.1:8b", temperature=0.0).bind_tools(tools=tools)