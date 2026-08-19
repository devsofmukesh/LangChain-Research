# =========================
# Load Dependencies
# =========================

import uuid
from typing import Literal
from executor import execute_tools
from utils import format_box, format_time
from chains import revisor, first_responder
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import START, END, StateGraph, MessagesState

# =========================
# Constants
# =========================

MAX_ITERATIONS = 2

# =========================
# Draft Node (Initial Response)
# =========================

def draft_node(state: MessagesState):
    """
    Generate an initial response to the user's question.
    """
    response = first_responder.invoke(input={"messages": state["messages"]})
    message = AIMessage(
        content=response.answer,
        tool_calls=[{"name": "AnswerQuestion", "args": response.model_dump(), "id": str(uuid.uuid4())}],
    )
    return {"messages": [message]}

# =========================
# Search Node
# =========================

def revise_node(state: MessagesState):
    """
    Generate a revised answer based on the critique and search results.
    """
    response = revisor.invoke(input={"messages": state["messages"]})
    message = AIMessage(
        content=response.answer,
        tool_calls=[{"name": "ReviseAnswer", "args": response.model_dump(), "id": str(uuid.uuid4())}],
    )
    return {"messages": [message]}

# =========================
# Conditional Edge Function
# =========================

def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """
    Determine whether to execute tools based on the number of iterations.
    If the number of iterations exceeds the maximum allowed, end the process.
    """
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state["messages"])
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


# =========================
# Graph Builder
# =========================

# Create the state graph
builder = StateGraph(MessagesState)

# Register graph nodes
builder.add_node(node="draft", action=draft_node)              # Generates the initial response
builder.add_node(node="execute_tools", action=execute_tools)   # Runs required tools/actions
builder.add_node(node="revise", action=revise_node)            # Refines the response using results

# Define graph flow
builder.add_edge(start_key=START, end_key="draft")             # Start → Draft
builder.add_edge(start_key="draft", end_key="execute_tools")   # Draft → Tool execution
builder.add_edge(start_key="execute_tools", end_key="revise")  # Tools → Revision

# Decide whether to continue looping or finish
builder.add_conditional_edges(
    source="revise",
    path=event_loop,
    path_map={
        "execute_tools": "execute_tools",  # Continue refinement loop
        END: END                           # Stop execution
    }
)

# Compile the graph to create an executable version of it.
graph = builder.compile()

# Draw the Flow as a Mermaid Diagram and save it as a PNG file
print(graph.get_graph().print_ascii())
print(graph.get_graph().draw_mermaid())
graph.get_graph().draw_mermaid_png(output_file_path="flow.png")

# =========================
# Run Example
# =========================

# Run the graph with an initial user message and print the final response.
response = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
            }
        ]
    }
)

# Extract the final answer from the last message with tool calls
last_message = response["messages"][-1]
if isinstance(last_message, AIMessage):
    print(format_box(last_message.content))