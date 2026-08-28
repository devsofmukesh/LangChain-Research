# =========================
# Load Dependencies
# =========================

from utils import format_box
from dotenv import load_dotenv
from graph.graph import rag_graph


# ==========================
# Load Environment Variables
# ==========================

load_dotenv()


# =========================
# Run the RAG Graph Workflow
# =========================

if __name__ == "__main__":

    # Execute the RAG Graph Workflow with a sample question
    result = rag_graph.invoke(input={"question": "What is agent memory?"})

    # Just show the generation cleanly:
    print(format_box(result["generation"]))

    # Or format documents readably:
    doc_summaries = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'N/A')}\n{doc.page_content[:300]}..."
        for doc in result["documents"]
    )
    print(format_box(doc_summaries))