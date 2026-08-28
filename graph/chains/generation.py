# =================
# Load Dependencies
# =================

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==========
# Ollama LLM
# ==========

llm = ChatOllama(model="qwen3.5:9b", temperature=0)


# ===========================
# Generation Prompt and Chain
# ===========================

# hub.pull() depricated, use langsmith client.pull_prompt() instead
# Docs: https://reference.langchain.com/python/langsmith/async_client/AsyncClient/pull_prompt
# Here using custom prompt for generation.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {question} \nContext: {context} \nAnswer:",
        )
    ]
)


# ================
# Generation Chain
# ================
generation_chain = prompt | llm | StrOutputParser()