# =================
# Load Dependencies
# =================

from tqdm import tqdm
from typing import Any
from typing import Dict
from utils import format_box
from graph.state import GraphState
from graph.chains.retrieval_grader import retrieval_grader

# =========================
# Grade Retrieved Documents
# =========================

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Arguments:
        state (dict): The current graph state containing the question and retrieved documents.

    Returns:
        state (dict): Filtered out irrelevant documents and updated web search state. 
    """

    # Print the validate message in a formatted box
    print(format_box("VALIDATE DOCUMENT RELEVANCY TO QUESTION"))
    question, documents = state["question"], state["documents"]

    # Initialize an empty list to hold the filtered documents and a flag for web search
    filtered_documents = []
    web_search = False

    # Initialize counters for total, relevant, and irrelevant documents
    relevant_documents = 0
    irrelevant_documents = 0
    total_documents = len(documents)

    # Loop through each document and grade its relevancy to the question
    for document in tqdm(iterable=documents, desc="Grading Documents", unit="document"):
        score = retrieval_grader.invoke({"question": question, "document": document.page_content})
        if score.binary_score.lower() == "yes":
            filtered_documents.append(document)
            relevant_documents += 1
        else:
            irrelevant_documents += 1
            web_search = True

    # Print the grading summary in a formatted box
    print(format_box(
        f"DOCUMENT GRADING SUMMARY\n\n"
        f"Total Documents Graded   : {total_documents}\n"
        f"Relevant Documents       : {relevant_documents}\n"
        f"Irrelevant Documents     : {irrelevant_documents}\n"
        f"Web Search Required      : {'Yes' if web_search else 'No'}"
    ))

    # Return the updated state with filtered documents and web search flag
    return {"documents": filtered_documents, "question": question, "web_search": web_search}