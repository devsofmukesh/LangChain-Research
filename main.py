# ==========================
# Load Dependencies
# ==========================

import os
from utils import format_box
from dotenv import load_dotenv
from operator import itemgetter
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import HumanMessage
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ==========================
# Load Environment Variables
# ==========================

# Read the Pinecone configuration from the local environment or .env file.
load_dotenv()


# ==========================
# Shared Helpers
# ==========================


def format_docs(docs):
    """Join retrieved document text into one context string."""
    return "\n\n".join(doc.page_content for doc in docs)


# ==========================
# Initialize Components
# ==========================


def initialize_components():
    """Create the Ollama model, Pinecone store, and document retriever."""

    # Use a chat model for answers and a dedicated model for vector embeddings.
    embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
    llm = ChatOllama(model="qwen3.5:9b")

    # Accept the documented index variable and the legacy name in existing .env files.
    index_name = os.getenv("PINECONE_INDEX_NAME") or os.getenv("INDEX_NAME")
    if not index_name:
        raise ValueError("Set PINECONE_INDEX_NAME in your .env file.")

    # Connect to the configured Pinecone index and retrieve its nearest chunks.
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    # Create a retriever that returns the top 3 most relevant document chunks.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Return the model and retriever for use in the main execution.
    return llm, retriever


# ==========================
# Retrieval Without LCEL
# ==========================


def retrieval_chain_without_lcel(query: str, retriever, llm, prompt_template):
    """Retrieve documents and generate an answer through explicit steps."""

    # Retrieve matching document chunks from Pinecone.
    docs = retriever.invoke(query)

    # Add the retrieved text and question to the prompt.
    context = format_docs(docs)

    # Format the prompt messages for the model to answer the question.
    messages = prompt_template.format_messages(context=context, question=query)

    # Ask Ollama to answer using the formatted prompt messages.
    response = llm.invoke(messages)

    # Return the model's answer content to the caller.
    return response.content


# ==========================
# Retrieval With LCEL
# ==========================


def create_retrieval_chain_with_lcel(retriever, llm, prompt_template):
    """Build a reusable retrieval chain with LangChain Expression Language."""

    # Compose retrieval, prompt formatting, model invocation, and parsing.
    return (
        RunnablePassthrough.assign(context=itemgetter("question") | retriever | format_docs)
        | prompt_template
        | llm
        | StrOutputParser()
    )


# ==========================
# Main Execution
# ==========================


def main():
    """Run direct-model and retrieval-augmented question answering examples."""

    # Show a boxed startup message before connecting to external services.
    print(format_box("Initializing components...", width=96))
    llm, retriever = initialize_components()

    # Build the prompt that grounds generated answers in retrieved context.
    prompt_template = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
    )

    # Set the sample question used by all three answer approaches.
    query = "What is Pinecone in Machine Learning?"
    print(format_box(f"Retrieving for question: {query}", width=96))

    # Ask the model directly without retrieving supporting documents.
    result_raw = llm.invoke([HumanMessage(content=query)])
    print(
        format_box(
            f"Implementation 0: Raw LLM Invocation (No RAG)\n\n"
            f"Answer:\n{result_raw.content}",
            width=96,
        )
    )

    # Run retrieval with explicit Python steps and box its answer.
    result_without_lcel = retrieval_chain_without_lcel(
        query,
        retriever,
        llm,
        prompt_template,
    )
    print(
        format_box(
            f"Implementation 1: Retrieval Without LCEL\n\n"
            f"Answer:\n{result_without_lcel}",
            width=96,
        )
    )

    # Summarize the composition features demonstrated by the LCEL approach.
    print(
        format_box(
            "Why use LCEL?\n"
            "- More concise and declarative\n"
            "- Supports streaming with chain.stream()\n"
            "- Supports async calls with chain.ainvoke()\n"
            "- Easy to compose with other chains\n"
            "- Reusable for additional inputs",
            width=96,
        )
    )

    # Run the composed LCEL retrieval chain and box the final answer.
    chain_with_lcel = create_retrieval_chain_with_lcel(
        retriever,
        llm,
        prompt_template,
    )
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print(
        format_box(
            f"Implementation 2: Retrieval With LCEL\n\n" f"Answer:\n{result_with_lcel}",
            width=96,
        )
    )


if __name__ == "__main__":
    # Execute the demo only when this file is run directly.
    main()
