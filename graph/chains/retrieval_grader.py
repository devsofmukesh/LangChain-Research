# =================
# Load Dependencies
# =================

from pydantic import Field
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# ========================
# Structured Output Schema
# ========================

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


# ==============
# Grading Prompt
# ==============

system = """
You are a grader assessing the relevance of a retrieved document to a user question.
If the document contains keywords or semantic meaning related to the question, grade it as relevant.
Return only a binary score:
- "yes" if the document is relevant
- "no" if the document is not relevant
"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question:\n\n{question}"),
    ]
)

# ==========
# Ollama LLM
# ==========

llm = ChatOllama(model="qwen3.5:9b", temperature=0)

# =====================
# Structured LLM Grader
# =====================

structured_llm_grader = llm.with_structured_output(schema=GradeDocuments)


# ======================
# Retrieval Grader Chain
# ======================

retrieval_grader = grade_prompt | structured_llm_grader