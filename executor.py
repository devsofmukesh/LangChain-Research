# =========================
# Load Dependencies
# =========================

from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage

# =========================
# Environment Setup
# =========================

load_dotenv()

# =========================
# Tavily Tool
# =========================

# Initialize the TavilySearch tool, which will be used to execute 
# search queries generated during the revision process.
tavily_tool = TavilySearch(max_results=5)

# =========================
# Search Executor
# =========================

def execute_tools(state: MessagesState):
    """
    Execute the search queries attached to the last AI message (via tool_calls)
    and return the results as ToolMessages to append to the graph state.
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    tool_messages = []
    for call in tool_calls:
        search_queries = call["args"].get("search_queries", [])
        if not search_queries:
            continue
        results = tavily_tool.batch([{"query": query} for query in search_queries])
        tool_messages.append(
            ToolMessage(
                content=str(results),
                name=call["name"],
                tool_call_id=call["id"],
            )
        )
    return {"messages": tool_messages}