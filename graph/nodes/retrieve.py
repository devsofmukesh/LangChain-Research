# =================
# Load Dependencies
# =================

from typing import Any
from typing import Dict
from utils import format_box
from ingestion import retriever
from graph.state import GraphState

# ================================
# Retrieve state from Vector Store
# ================================

def retrieve(state: GraphState) -> Dict[str, Any]:

    # Print the retrieve message in a formatted box
    print(format_box(f"RETRIEVE", width=150))

    # Retrieve the question from the state
    question = state["question"]

    # Use the retriever to get relevant documents from the vector store
    documents = retriever.invoke(question)

    # Return the retrieved documents and the question in a dictionary
    return {"documents": documents, "question": question}