# =========================
# Load Dependencies
# =========================

import datetime
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from schemas import ReviseAnswer, AnswerQuestion
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# =========================
# Environment Setup
# =========================

# Load environment variables from a .env file
load_dotenv()

# =========================
# LLM Configuration
# =========================

# Initialize the ChatOllama model with specific parameters for response generation.
llm = ChatOllama(model="qwen3.5:9b", temperature=0.2)

# =========================
# Base Prompt Template
# =========================

# Define a base prompt template for the actor, which includes instructions for both the initial response and revision.
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are an expert researcher.\n"
                "Current time: {time}\n\n"
                "1. {first_instruction}\n"
                "2. Reflect and critique your answer.\n"
                "3. Recommend search queries to improve the answer."
            ),
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
).partial(
    time=lambda: datetime.datetime.now().isoformat()
)

# =========================
# Draft Prompt
# =========================

# Instructions for the initial response, guiding the model to provide a detailed answer and reflect on it.
first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer."
)

# =========================
# Revision Prompt
# =========================

# Instructions for revision, guiding the model on how to improve the answer based on the critique and new information.
revision_instructions = (
    "Revise your previous answer using the new information.\n"
    "- Improve factual accuracy.\n"
    "- Improve clarity.\n"
    "- Use the critique to improve the answer.\n"
    "- Include references at the end.\n"
    "- Keep the answer under 250 words.\n"
)

# Create a revision prompt template by partially applying the actor prompt 
# template with specific instructions for revision.
revision_prompt_template = actor_prompt_template.partial(
    first_instruction=revision_instructions
)

# =========================
# Structured Output Chains
# =========================

# Create structured output chains for the first response and revision, using the defined schemas.
first_responder = (first_responder_prompt_template | llm.with_structured_output(AnswerQuestion))

# The revisor chain takes the previous messages, including the initial answer and tool results, 
# and produces a revised answer based on the critique and recommendations. It uses the ReviseAnswer 
# schema to ensure the output includes the revised answer, reflection, search queries, and references.
revisor = (revision_prompt_template | llm.with_structured_output(ReviseAnswer))