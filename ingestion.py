# ==========================
# Load Dependencies
# ==========================

import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

from utils import format_box

# ==========================
# Load Environment Variables
# ==========================

# Read Pinecone credentials and configuration from the local environment.
load_dotenv()


# ==========================
# Main Execution
# ==========================


def main():
    """Load the source text, create chunks and embeddings, and index them."""

    # Show a boxed status message before starting the ingestion steps.
    print(format_box("Starting document ingestion...", width=96))

    # Load the source article from the repository's current working directory.
    document_path = os.path.join(os.getcwd(), "mediumblog1.txt")
    loader = TextLoader(file_path=document_path, encoding="utf-8")
    documents = loader.load()

    # Split the source documents into chunks for embedding and retrieval.
    print(format_box("Splitting document into chunks...", width=96))
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents=documents)
    print(format_box(f"Number of chunks created: {len(texts)}", width=96))

    # Use a dedicated embedding model, matching the one used by main.py.
    embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    embedding = OllamaEmbeddings(model=embedding_model)

    # Store the chunk vectors in the configured Pinecone index.
    print(format_box("Creating vector store and adding documents...", width=96))
    index_name = os.getenv("PINECONE_INDEX_NAME") or os.getenv("INDEX_NAME")
    if not index_name:
        raise ValueError("Set PINECONE_INDEX_NAME in your .env file.")

    PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embedding,
        index_name=index_name,
    )

    # Confirm that ingestion completed successfully.
    print(format_box("Ingestion complete!", width=96))


if __name__ == "__main__":
    # Run ingestion only when this file is executed as a script.
    main()
