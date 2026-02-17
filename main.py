# =========================
# Load Dependencies
# =========================
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END
from nodes import run_agent_reasoning, tool_node

# =========================
# Environment Configuration
# =========================
load_dotenv()

# Define Constants
AGENT_REASON="agent_reason"
ACT= "act"
LAST = -1

# Define the conditional function to determine the next step in the flow
def should_continue(state: MessagesState) -> str:
    """Determine whether to continue with the agent reasoning or to end the flow based on tool calls."""
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

# Define the Flow
flow = StateGraph(MessagesState)
# Add Nodes and Edges to the Flow
flow.add_node(AGENT_REASON, run_agent_reasoning)
# The tool_node is already defined in nodes.py and imported here, so we can directly use it in the flow.
flow.set_entry_point(AGENT_REASON)
# Add the tool node to the flow and connect it to the agent reasoning node
flow.add_node(ACT, tool_node)
# Add conditional edges based on the should_continue function
flow.add_conditional_edges(AGENT_REASON, should_continue, {END:END, ACT:ACT})
# Connect the ACT node back to the AGENT_REASON node to allow for iterative reasoning based on tool calls
flow.add_edge(ACT, AGENT_REASON)
# Compile the Flow into an Application
app = flow.compile()
# Draw the Flow as a Mermaid Diagram and save it as a PNG file
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("Hello ReAct LangGraph with Function Calling")
    res = app.invoke({"messages": [HumanMessage(content="What is the temperature in Tokyo? List it and then triple it")]})
    print(res["messages"][LAST].content)