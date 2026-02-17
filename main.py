# =========================
# Load Dependencies
# =========================
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from chains import generate_chain, reflect_chain
from langgraph.graph import MessagesState, StateGraph, END

# =========================
# Environment Configuration
# =========================
load_dotenv()

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Define Constants
REFLECT = "reflect"
GENERATE = "generate"

def generation_node(state: MessageGraph):
    """Node function for generating a response based on the current messages."""
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}

def reflection_node(state: MessageGraph):
    """Node function for reflecting on the current messages and generating a new response."""
    response = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=response.content)]}

def should_continue(state: MessageGraph):
    """"Function to determine whether to continue the loop or end it based on the number of messages."""
    if len(state["messages"]) >= 6:
        return END
    return REFLECT

# Build the State Graph
builder = StateGraph(state_schema=MessageGraph)
# Add nodes and edges to the graph
builder.add_node(GENERATE, generation_node)
# The reflection node will take the output of the generation node as its input, creating a loop between the two nodes.
builder.add_node(REFLECT, reflection_node)
# Set the entry point of the graph to the generation node, which will start the process of generating a response based on the initial messages.
builder.set_entry_point(GENERATE)
# Add a conditional edge from the generation node to itself, which will allow the graph to continue generating responses until the should_continue function determines that it should end.
builder.add_conditional_edges(GENERATE, should_continue, path_map={REFLECT: REFLECT, END: END})
# Add an edge from the reflection node back to the generation node, which will allow the graph to continue reflecting and generating responses until the should_continue function determines that it should end.
builder.add_edge(REFLECT, GENERATE)
# Compile the graph to create an executable version of it.
graph = builder.compile()
# Draw the Flow as a Mermaid Diagram and save it as a PNG file
print(graph.get_graph().print_ascii())
print(graph.get_graph().draw_mermaid())
graph.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    # Invoke the graph with an initial set of messages to start the process and print the final response.
    print("Hello LangGraph")
    
    # The initial input message is a HumanMessage containing a tweet that we want to improve. 
    # The graph will process this message through the generation and reflection nodes, iteratively 
    # improving the response until the stopping condition is met (when there are 6 or more messages in the state).
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                    @LangChainAI
            — newly Tool Calling feature is seriously underrated.

            After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

            Made a video covering their newest blog post."""
            )
        ]
    }
    # Invoke the graph with the initial messages and print the final response after processing through the graph.
    response = graph.invoke(inputs)
    
    # The final response will be the result of the iterative process of generating and reflecting on the messages,
    # ultimately producing an improved version of the original tweet.
    print(response)