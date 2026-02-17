from react import llm
from react import tools
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState

# =========================
# Environment Configuration
# =========================
load_dotenv()

# Agent Reasoning Node
SYSYEM_MESSAGE="""You are a helpful assistant that can use tools to answer questions."""

# Define the Agent Reasoning Node
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """Run the agent reasoning node."""
    response = llm.invoke([{"role": "system", "content": SYSYEM_MESSAGE}, *state["messages"]])
    return {"messages": [response]}

# Create the Tool Node
tool_node = ToolNode(tools)
