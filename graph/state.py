# =================
# Load Dependencies
# =================

from typing import List
from typing import TypedDict

# ==========================
# Graph State Representation
# ==========================

class GraphState(TypedDict):
    """
    It represents the state of our graph.

    Attributes:
        question (string): The question to be answered.
        generation (string): LLM generated answer to the question.
        web_search (bool): Whether to add search results to the graph or not.
        documents (List[string]): A list of documents that are relevant to the question and can be used to answer it.
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]