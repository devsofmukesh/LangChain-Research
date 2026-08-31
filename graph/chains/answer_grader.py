# =================
# Load Dependencies
# =================

from pydantic import Field
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# ========================
# Structured Output Schema
# ========================

class GradeAnswer(BaseModel):
    """Binary score indicating whether the generated answer addresses the question."""
    binary_score: bool = Field(description="Answer addresses the question, 'yes' or 'no'")


# ==========
# Ollama LLM
# ==========

llm = ChatOllama(model="qwen3.5:9b", temperature=0)


# ============================
# Structured LLM Answer Grader
# ============================

# Creates a structured LLM grader that returns results according to the GradeAnswer schema.
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Defines the system instructions for evaluating whether an answer resolves the user's question.
system = """
    You are a grader assessing whether an answer addresses / resolves a question \n 
    Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question.
"""

# Creates a chat prompt template containing the grading instructions and input variables.
answer_prompt = ChatPromptTemplate.from_messages(

    # Defines the system and human messages that will be sent to the LLM.
    messages=[

        # Passes the grading instructions as the system message.
        ("system", system),

        # Provides the user's question and the LLM-generated answer for evaluation.
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

# Chains the answer prompt with the structured grader to evaluate whether the answer resolves the question.
answer_grader: RunnableSequence = answer_prompt | structured_llm_grader