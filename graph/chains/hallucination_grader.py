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

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: bool = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

# ==========
# Ollama LLM
# ==========

llm = ChatOllama(model="qwen3.5:9b", temperature=0)


# ===================================
# Structured LLM Hallucination Grader
# ===================================

# Creates a structured LLM grader that returns output according to the GradeHallucinations schema.
structured_llm_grader = llm.with_structured_output(schema=GradeHallucinations)

# Defines the system instructions for evaluating whether an LLM response is supported by retrieved facts.
system = """
    You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
"""

# Creates a chat prompt template containing the system instructions and human-provided facts and generation.
hallucination_prompt = ChatPromptTemplate.from_messages(
    
    # Defines the messages that will be passed to the LLM.
    messages=[
        
        # Adds the grading instructions as the system message.
        ("system", system),
        
        # Provides the retrieved facts and LLM-generated answer as variables for evaluation.
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

# Chains the hallucination prompt with the structured LLM grader to produce a graded result.
hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader