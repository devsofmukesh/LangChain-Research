import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()


if __name__ == "__main__":
    # Display startup message
    print("Ingesting!...")
    
    # Load text document
    loader = TextLoader(file_path=os.path.join(os.getcwd(), "mediumblog1.txt"), encoding="utf-8")
    documents = loader.load()

    # Display message about splitting document
    print("Splitting document into chunks...")

    # Split document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents=documents)
    print(f"Number of chunks created: {len(texts)}")

    # Create embedding using Ollama
    embedding = OllamaEmbeddings(model="llama3.1:8b")

    # Create Pinecone vector store and add documents
    print("Creating vector store and adding documents...")

    # Initialize Pinecone vector store from documents
    PineconeVectorStore.from_documents(documents=texts, embedding=embedding, index_name=os.getenv("PINECONE_INDEX_NAME"))

    # Display completion message
    print("Ingestion complete!")