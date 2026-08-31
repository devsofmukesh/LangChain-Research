# =================
# Load Dependencies
# =================

from utils import format_box
from dotenv import load_dotenv
from ingestion import retriever
from graph.chains.router import RouteQuery
from graph.chains.router import question_router
from graph.chains.generation import generation_chain
from graph.chains.retrieval_grader import GradeDocuments
from graph.chains.retrieval_grader import retrieval_grader
from graph.chains.hallucination_grader import GradeHallucinations
from graph.chains.hallucination_grader import hallucination_grader


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

def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    documents = retriever.invoke(input=question)
    generation = generation_chain.invoke(input={"context": documents, "question": question})
    response: GradeHallucinations = hallucination_grader.invoke(input={"documents": documents, "generation": generation})
    assert response.binary_score

def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    documents = retriever.invoke(input=question)
    response: GradeHallucinations = hallucination_grader.invoke(
        input={"documents": documents, "generation": "In order to make pizza we need to first start with the dough"}
    )
    assert not response.binary_score

def test_router_to_vectorstore() -> None:
    question = "agent memory"
    response: RouteQuery = question_router.invoke(input={"question": question})
    assert response.datasource == "vectorstore"

def test_router_to_websearch() -> None:
    question = "how to make pizza?"
    response: RouteQuery = question_router.invoke(input={"question": question})
    assert response.datasource == "websearch"