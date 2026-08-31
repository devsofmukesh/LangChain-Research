# =================
# Load Dependencies
# =================

import io
import re
import base64
import logging
import requests
import contextlib
from utils import format_box
from dotenv import load_dotenv
from langgraph.graph import END
from graph.state import GraphState
from grandalf.utils import geometry
from langgraph.graph import StateGraph
from graph.chains.router import RouteQuery
from graph.chains.router import question_router
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH


# ==================
# Disable INFO Noise
# ==================

for noisy in ("httpx", "httpcore", "unstructured", "chromadb", "sentence_transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

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

    Arguments:
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

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    """
    This function evaluates the generated response in two stages:
    1. Checks whether the response is grounded in the retrieved documents.
    2. Checks whether the response properly answers the question.

    Arguments:
        state (GraphState): The current state of the graph, which includes
        the question, documents, and generated response.

    Returns:
        str: A string indicating the quality of the generated response.
        It returns:
        - 'useful' if the response is grounded and answers the question.
        - 'not useful' if the response is grounded but does not answer the question.
        - 'unsupported' if the response is not grounded in the documents.
    """

    # Check generated response and extract state information
    print(format_box("CHECK GENERATED RESPONSE"))
    question, documents, generation = state["question"], state["documents"], state["generation"]

    # Check for hallucinations
    print(format_box("CHECK HALLUCINATIONS"))
    hallucination_score = hallucination_grader.invoke(input={"documents": documents, "generation": generation})

    # Check if the generated response is grounded in the documents
    if hallucination_score.binary_score:
        print(format_box("DECISION: GENERATION IS GROUNDED IN DOCUMENTS"))

        # Check Answer Relevance
        print(format_box("CHECK GENERATION AGAINST QUESTION"))
        answer_score = answer_grader.invoke(input={"question": question, "generation": generation})

        # Check if the generated response answers the question
        if answer_score.binary_score:
            print(format_box("DECISION: GENERATION ADDRESSES QUESTION"))
            return "useful"

        else:
            print(format_box("DECISION: GENERATION DOES NOT ADDRESS QUESTION"))
            return "not useful"

    # Generation Is Unsupported
    else:
        print(format_box("DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY"))
        return "unsupported"

# ===============================
# Functions for Conditional Edges
# ===============================

def route_question(state: GraphState) -> str:
    """
    This function routes the user's question to the appropriate data source
    based on the decision made by the question router.

    Arguments:
        state (GraphState): The current state of the graph, which includes
        the user's question and other relevant information.

    Returns:
        str: A string indicating the next node in the graph.
        It returns:
        - 'WEBSEARCH' if the question requires web search.
        - 'RETRIEVE' if the question can be answered using the vector store.
    """

    # Extract User Question
    print(format_box("ROUTE QUESTION"))
    question = state["question"]

    # Use the question router to determine the appropriate data source
    source: RouteQuery = question_router.invoke(input={"question": question})

    # Check if the question requires information from the web
    if source.datasource == WEBSEARCH:
        print(format_box("DECISION: ROUTE QUESTION TO WEB SEARCH"))
        return WEBSEARCH

    # Check if the question can be answered using retrieved documents
    elif source.datasource == "vectorstore":
        print(format_box("DECISION: ROUTE QUESTION TO RAG"))
        return RETRIEVE


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
workflow.set_conditional_entry_point(path=route_question, path_map={WEBSEARCH: WEBSEARCH, RETRIEVE: RETRIEVE})
workflow.add_edge(start_key=RETRIEVE, end_key=GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    source=GRADE_DOCUMENTS, path=decide_to_generate, 
    path_map={WEBSEARCH: WEBSEARCH, GENERATE: GENERATE}
)
workflow.add_conditional_edges(
    source=GENERATE, path=grade_generation_grounded_in_documents_and_question,
    path_map={"unsupported": GENERATE, "useful": END, "not useful": WEBSEARCH}
)
workflow.add_edge(start_key=WEBSEARCH, end_key=GENERATE)
# workflow.add_edge(start_key=GENERATE, end_key=END)

# Compile the graph into an executable workflow
rag_graph = workflow.compile()

# ==============================================
# Visualize the LangGraph Workflow (ASCII + PNG)
# ==============================================

# Monkey-patch grandalf's intersectR to avoid crashing on self-loops (band-aid fix)
_original_intersectR = geometry.intersectR  # keep a reference to the original function

def _safe_intersectR(view, topt):
    try:
        # try the normal calculation first
        return _original_intersectR(view, topt) 
    
    except ValueError:
        # fallback: return node center instead of crashing
        return (view.xy[0], view.xy[1])  

# Replace the buggy function with the safe wrapper
geometry.intersectR = _safe_intersectR  

# Draw the Flow as a Mermaid Diagram and save it as a PNG file
try:
    # Create an in-memory buffer to capture printed output
    ascii_buffer = io.StringIO()

    # Redirect stdout into the buffer while print_ascii() runs
    with contextlib.redirect_stdout(ascii_buffer):
        rag_graph.get_graph().print_ascii()

    # Extract the captured text as a string
    ascii_output = ascii_buffer.getvalue()

    # Wrap the captured diagram text in format_box and print it
    print(format_box(ascii_output))

except Exception as e:
    print(format_box(f"ASCII render skipped (grandalf layout bug): {e}"))

# Save PNG (requires network access to mermaid.ink by default)
try:
    # Get raw Mermaid flowchart syntax (includes its own default front-matter block)
    mermaid_text = rag_graph.get_graph().draw_mermaid()

    # Strip out the auto-generated front-matter block so we don't end up with two
    mermaid_body = re.sub(r"^---.*?---\n", "", mermaid_text, flags=re.DOTALL)

    # Build a single YAML front-matter block to force ELK adaptive layout
    config_override = "---\nconfig:\n  layout: elk\n  flowchart:\n    curve: linear\n---\n"

    # Combine our config with just the graph body
    full_mermaid = config_override + mermaid_body

    # Convert string to bytes for encoding
    graph_bytes = full_mermaid.encode("utf8")

    # base64-encode bytes, then decode to a URL-safe string
    base64_string = base64.urlsafe_b64encode(graph_bytes).decode("ascii")

    # Build the mermaid.ink render URL
    url = f"https://mermaid.ink/img/{base64_string}"

    # Request the rendered PNG from mermaid.ink
    response = requests.get(url)

    # Raise an exception if the server returned an error instead of an image
    response.raise_for_status()

    # Open flow.png in binary write mode
    with open(file="flow.png", mode="wb") as mermaid_file:

        # write the PNG bytes to disk
        mermaid_file.write(response.content)

    # Confirm success to the console
    print(format_box("flow.png saved successfully"))

except Exception as e:
    # Report failure without crashing the script
    print(format_box(f"PNG render skipped — likely network/VPN issue: {e}"))