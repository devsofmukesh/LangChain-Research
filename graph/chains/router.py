# =================
# Load Dependencies
# =================

from pydantic import Field
from typing import Literal
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# ========================
# Structured Output Schema
# ========================

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


# ==========
# Ollama LLM
# ==========

llm = ChatOllama(model="qwen3.5:9b", temperature=0)


# =====================
# Structured LLM Router
# =====================

# Creates a structured LLM router that returns output according to the RouteQuery schema.
structured_llm_router = llm.with_structured_output(RouteQuery)

# Defines the system instructions for evaluating whether an LLM response is supported by retrieved facts.
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""

# Creates a chat prompt template containing the system instructions and human-provided facts and generation.
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# Chains the router prompt with the structured LLM router to produce a graded result.
question_router = route_prompt | structured_llm_router