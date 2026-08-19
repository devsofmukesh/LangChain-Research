# =========================
# Load Dependencies
# =========================

from typing import List
from pydantic import Field, BaseModel

# =========================
# Reflection Schema
# =========================

# Defines the structure for reflecting on and improving a previous answer.
class Reflection(BaseModel):
    """
    critique: A critique of the previous answer, highlighting its weaknesses and areas for improvement.
    recommendations: A list of specific recommendations to improve the answer, such as suggestions for length, style, content, etc.
    """
    critique: str = Field(..., description="A critique of the previous answer.")
    recommendations: List[str] = Field(..., description="A list of recommendations to improve the answer.")

# =========================
# Initial Answer Schema
# =========================

# Defines the expected output format containing the answer, reflection, and search queries.
class AnswerQuestion(BaseModel):
    """
    answer: The final answer to the user's question.
    reflection: The reflection on the initial answer, including critique and recommendations.
    search_queries: The best three search queries to research information and improve the answer.
    """
    answer: str = Field(..., description="A detailed 250 words answer to the user's question.")
    reflection: Reflection = Field(..., description="The reflection on the initial answer, including critique and recommendations.")
    search_queries: List[str] = Field(..., description="Best three search queries to research information and improve the answer.")

# =========================
# Revised Answer Schema
# =========================

# Defines the expected output format for a revised answer based on the critique and recommendations.
class ReviseAnswer(AnswerQuestion):
    """
    references: A list of references used to improve the answer based on the recommendations.
    """
    references: List[str] = Field(..., description="A list of references used to improve the answer based on the recommendations.")