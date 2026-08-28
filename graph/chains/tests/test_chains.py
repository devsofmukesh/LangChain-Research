# =================
# Load Dependencies
# =================

from pprint import pprint
from utils import format_box
from dotenv import load_dotenv
from ingestion import retriever
from graph.chains.generation import generation_chain
from graph.chains.retrieval_grader import GradeDocuments
from graph.chains.retrieval_grader import retrieval_grader


# ==========================
# Load Environment Variables
# ==========================

load_dotenv()


# =====================
# Test Retrieval Grader
# =====================

def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    documents = retriever.invoke(input=question)
    document_text = documents[1].page_content
    response: GradeDocuments = retrieval_grader.invoke(input={"question": question, "document": document_text})
    assert response.binary_score == "yes"

def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    documents = retriever.invoke(input=question)
    document_text = documents[1].page_content
    response: GradeDocuments = retrieval_grader.invoke(input={"question": "how to make pizaa", "document": document_text})
    assert response.binary_score == "no"

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(input=question)
    generation = generation_chain.invoke(input={"context": docs, "question": question})
    print("\n", format_box(generation, width=150))