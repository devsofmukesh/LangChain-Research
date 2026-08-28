# =================
# Load Dependencies
# =================

from typing import Any
from typing import Dict
from utils import format_box
from dotenv import load_dotenv
from graph.state import GraphState
from langchain_tavily import TavilySearch
from langchain_core.documents import Document


# ==========================
# Load Environment Variables
# ==========================

load_dotenv()


# ==============================
# Web Search Tool Initialization
# ==============================

web_search_tool = TavilySearch(max_results=3)


# ========================
# Web Search Node Function
# ========================

def web_search(state: GraphState) -> Dict[str, Any]:

    # Print the web search message in a formatted box
    print(format_box("WEB SEARCH FOR QUESTION"))

    # Extract the question and documents from the state
    question = state["question"]
    documents = state["documents"] if "documents" in state else None

    # Print the search message
    print(format_box(f"🔎 Searching Tavily for: {question}"))

    # Perform web search using TavilySearch
    tavily_results = web_search_tool.invoke(input={"query": question})["results"]

    # Confirm search worked
    print(format_box(f"✅ Web search completed for: {question}"))
    print(format_box(f"📄 Results found: {len(tavily_results)}"))

    # Print all result information in one box
    results_text = "\n\n".join(
        [
            f"--- Result {i} ---\n"
            f"Title: {result.get('title', 'N/A')}\n"
            f"URL:   {result.get('url', 'N/A')}"
            for i, result in enumerate(tavily_results, start=1)
        ]
    )

    print(format_box(results_text))

    # Join the content of all TavilySearch results into a single string
    joined_tavily_result = "\n".join([tavily_result["content"] for tavily_result in tavily_results])

    # Create a Document object for the web search results
    web_results = Document(page_content=joined_tavily_result)

    # Append the web search results to the documents list in the state
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]

    # Print the number of documents in the state
    print(format_box(f"📚 Documents in state: {len(documents)}"))
    print(format_box("✅ WEB SEARCH NODE COMPLETED"))

    # Return the updated state with the new documents
    return {"documents": documents, "question": question}


if __name__ == "__main__":

    # Test the web_search function with a sample state
    result = web_search(state={"question": "agent memory", "documents": None})
    print(format_box(f"FINAL STATE\nQuestion: {result['question']}\nDocuments: {len(result['documents'])}"))