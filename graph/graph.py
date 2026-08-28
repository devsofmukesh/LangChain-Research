# =================
# Load Dependencies
# =================

from utils import format_box
from dotenv import load_dotenv
from langgraph.graph import END
from graph.state import GraphState
from langgraph.graph import StateGraph
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH

# ==========================
# Load Environment Variables
# ==========================

load_dotenv()

# ===============================
# Functions for Conditional Edges
# ===============================

def decide_to_generate(state: GraphState) -> str:
    """
    This function decides whether to generate a response or perform 
    a web search based on the relevance of the documents to the question.

    Args:
        state (GraphState): The current state of the graph, which includes 
        the question, documents, and other relevant information.

    Returns:
        str: A string indicating the next node in the graph. It returns
        'WEBSEARCH' if not all documents are relevant to the question,
        otherwise it returns 'GENERATE' to proceed with generating a response.
    """
    # Assess graded documents
    print(format_box("ASSESS GRADED DOCUMENTS"))

    # Check if all documents are relevant to the question
    if state["web_search"]:
        print(format_box("DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH"))
        return WEBSEARCH
    else:
        print(format_box("DECISION: GENERATE"))
        return GENERATE


# ===============================
# Build the Graph Workflow
# ===============================
workflow = StateGraph(state_schema=GraphState)

# Add nodes to the graph
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

# Add edges to the graph
workflow.set_entry_point(key=RETRIEVE)
workflow.add_edge(start_key=RETRIEVE, end_key=GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    source=GRADE_DOCUMENTS, path=decide_to_generate, 
    path_map={WEBSEARCH: WEBSEARCH, GENERATE: GENERATE}
)
workflow.add_edge(start_key=WEBSEARCH, end_key=GENERATE)
workflow.add_edge(start_key=GENERATE, end_key=END)

# Compile the graph into an executable workflow
rag_graph = workflow.compile()

# Draw the Flow as a Mermaid Diagram and save it as a PNG file
print(format_box(rag_graph.get_graph().print_ascii()))
print(format_box(rag_graph.get_graph().draw_mermaid()))
rag_graph.get_graph().draw_mermaid_png(output_file_path="flow.png")