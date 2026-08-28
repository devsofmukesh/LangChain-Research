# =================
# Load Dependencies
# =================

from typing import Any
from typing import Dict
from utils import format_box
from graph.state import GraphState
from graph.chains.generation import generation_chain


# ============================
# Generate Answer from Context
# ============================

def generate(state: GraphState) -> Dict[str, Any]:
    
    # Print the generate message in a formatted box
    print(format_box("GENERATE"))

    # Extract the question and documents from the state
    question, documents = state["question"], state["documents"]

    # Generate an answer using the generation chain
    generation = generation_chain.invoke(input={"context": documents, "question": question})

    # Update the state with the generated answer
    return {"documents": documents, "question": question, "generation": generation}