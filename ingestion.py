# =========================
# Load Dependencies
# =========================

import os
import ssl
import hashlib
import asyncio
import certifi
from dotenv import load_dotenv
from typing import Any, Dict, List, Generator
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

# =========================
# Environment & SSL Setup
# =========================

# Load environment variables from .env file
load_dotenv()

# Create a default SSL context using certifi CA bundle
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Set SSL certificate path for Python SSL
os.environ["SSL_CERT_FILE"] = certifi.where()

# Set CA bundle path for requests-based libraries
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# =========================
# Embeddings & Vector Store
# =========================

# Initialize Ollama embeddings model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Initialize Pinecone vector store with embedding function
vectorstore = PineconeVectorStore(index_name="documentation-assistant-index", embedding=embeddings)


# =========================
# Tavily Tool Configuration
# =========================

# Configure Tavily site mapper with crawl limits
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)

# Initialize Tavily crawler
tavily_crawl = TavilyCrawl()

# Initialize Tavily extractor
tavily_extract = TavilyExtract()


# =========================
# Utility Functions
# =========================

# Define a function to chunk a list of URLs into smaller batches
def chunk_urls(urls: List[str], chunk_size: int = 20) -> Generator[List[str], None, None]:
    """Chunk a list of URLs into smaller lists of a given size."""

    # Iterate over URLs in steps of chunk_size
    for i in range(0, len(urls), chunk_size):

        # Yield a sublist of URLs
        yield urls[i:i + chunk_size]


# =========================
# Asynchronous Extraction
# =========================

# Define an async function to extract documents from a batch of URLs
async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract documents from a batch of URLs."""
    try:
        # Log the start of batch processing
        log_info(f"TavilyExtract: Processing batch {batch_num} with {len(urls)}", Colors.BLUE)

        # Asynchronously extract documents from URLs
        documents = await tavily_extract.ainvoke(input={"urls": urls})

        # Log successful batch completion
        log_success(f"TavilyExtract: Completed batch {batch_num} with {len(documents.get('results', []))} documents extracted.")

        # Return extracted documents
        return documents

    # Handle extraction errors
    except Exception as e:
        log_error(f"TavilyExtract: Error in batch {batch_num} - {str(e)}")


# =========================
# Batch Orchestration
# =========================

# Define an async function to process multiple URL batches concurrently
async def asynch_extract_batches(url_batches: List[List[str]]) -> List[Dict[str, Any]]:
    """Asynchronously extract documents from multiple batches of URLs."""

    # Log extraction start header
    log_header("STARTING ASYNCHRONOUS EXTRACTION OF URL BATCHES")

    # Log total number of batches
    log_info(f"Total batches to process: {len(url_batches)}", Colors.DARKCYAN)

    # Create async tasks for each URL batch
    tasks = [extract_batch(batch, i + 1) for i, batch in enumerate(url_batches)]

    # Run all extraction tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Initialize list to store extracted documents
    all_pages = list()

    # Initialize failed batch counter
    failed_batches = 0

    # Process results from all tasks
    for result in results:

        # Check if the task resulted in an exception
        if isinstance(result, Exception):
            failed_batches += 1
            log_error(f"TavilyExtract: Batch failed with exception - {str(result)}")

        # Handle successful extraction results
        else:
            for extracted_page in result.get("results", []):

                # Create a LangChain Document from extracted content
                document = Document(page_content=extracted_page.get("raw_content", ""), metadata={"source": extracted_page.get("url", "")})

                # Append document to collection
                all_pages.append(document)

    # Log total extracted pages
    log_success(f"TavilyExtract: Extraction completed with {len(all_pages)} pages extracted.")

    # Warn if any batches failed
    if failed_batches > 0:
        log_warning(f"TavilyExtract: {failed_batches} batches failed during extraction.", Colors.YELLOW)

    # Return all extracted documents
    return all_pages

def deterministic_id(document: Document) -> str:
    """Generate a deterministic ID for a document based on its content and source."""

    # Use source and content hash to create a unique ID
    source = document.metadata.get("source", "unknown")

    # Generate a SHA-256 hash of the document content
    content = document.page_content.strip()
    
    # Generate SHA-256 hash of the content
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # Combine source and content hash to form the ID
    return f"{source}:{content_hash}"


async def index_documents(documents: List[Document], batch_size: int = 100):
    """Index documents into the Pinecone vector store."""

    # Log the start of the indexing phase
    log_header("VECTOR STORAGE PHASE")

    # Log details about the number of documents to be indexed
    log_info(f"Vecto Store Indexing: Preparing to add {len(documents)} documents into Pinecone vector store.", Colors.DARKCYAN)

    # Create batches of documents for indexing
    batches = list(documents[i:i + batch_size] for i in range(0, len(documents), batch_size))

    # Log batching information
    log_info(f"Vector Store Indexing: Split into {len(batches)} batches for indexing with batch size {batch_size}.")

    # ============================
    # Vector Store Batch Ingestion
    # ============================

    # Define an async function to add a batch of documents to the vector store
    async def add_batch(batch: List[Document], batch_num: int):
        """Add a batch of documents to the vector store."""

        # Begin protected execution block
        try:
            # Asynchronously add the current batch of documents to the vector store
            await vectorstore.aadd_documents(documents=batch, ids=[deterministic_id(document) for document in batch])

            # Log successful batch ingestion with progress details
            log_success(f"Vector Store Indexing: Successfully added batch {batch_num} / {len(batches)} ({len(batch)} documents).")

        # Handle vector store ingestion errors
        except Exception as e:
            # Log batch ingestion failure
            log_error(f"Vector Store Indexing: Failed to add batch {batch_num} - {str(e)}")

            # Return failure status
            return False

        # Return success status
        return True

    # --------The below code is creating pressure on Ollama locally; commenting it out for now.--------
    # # Create async tasks for each batch of documents
    # tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]

    # # Run all batch ingestion tasks concurrently
    # results = await asyncio.gather(*tasks, return_exceptions=True)

    # # Count the number of successful batch ingestions
    # successful = sum(1 for result in results if result is True)

    # Initialize successful batch counter
    successful = 0

    # Sequentially add each batch with a delay to reduce pressure on Ollama
    for i, batch in enumerate(batches):
        result = await add_batch(batch, i + 1)
        if result:
            successful += 1
        await asyncio.sleep(0.5)

    # Log overall indexing results
    if successful == len(batches):
        log_success(f"Vector Store Indexing: All {len(documents)} documents successfully indexed into Pinecone vector store.")
    else:
        log_warning(f"Vector Store Indexing: Only {successful} out of {len(batches)} batches were successfully indexed.")


# =========================
# Main Application Logic
# =========================

# Define the main async ingestion function
async def main():
    """Main function to run the ingestion process."""

    # Log ingestion start header
    log_header("DOCUMENT INGESTION STARTED")

    # Log crawl start message
    log_info("Starting crawl for document ingestion using Tavily...", Colors.PURPLE)

    # Generate site map from documentation URL
    site_map = tavily_map.invoke(input={"url": "https://docs.langchain.com/oss/python/langchain/overview"})

    # Log mapping completion
    log_success(f"Mapping completed. {len(site_map['results'])} urls found.")

    # Split URLs into batches
    url_batches = list(chunk_urls(site_map["results"], chunk_size=20))

    # Log batching information
    log_info(f"URL Processing: Split {len(site_map['results'])} URLs into {len(url_batches)} batches", Colors.BLUE)

    # Asynchronously extract documents from all URL batches
    all_documents = await asynch_extract_batches(url_batches=url_batches)

    # Log the start of the document chunking phase
    log_header("DOCUMENT CHUNKING PHASE")

    # Log details about chunk size and overlap configuration
    log_info(f"Text Splitter: Processing {len(all_documents)} documents with 4000 chunk size and 200 overlap", Colors.YELLOW)

    # Initialize the recursive character text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)

    # Split the documents into overlapping chunks
    splitted_documents = text_splitter.split_documents(all_documents)

    # Enforce a hard max-length guard for Ollama embeddings
    splitted_documents = [doc for doc in splitted_documents if len(doc.page_content) <= 800]

    # Log the total number of chunks created
    log_success(f"Text Splitter: Created {len(splitted_documents)} chunks from {len(all_documents)} documents.")

    # Process the splitted documents in batches and index them
    await index_documents(documents=splitted_documents, batch_size=100)

    # Log ingestion completion header
    log_header("DOCUMENT INGESTION PIPELINE COMPLETED SUCCESSFULLY")
    log_success("All documents have been ingested, processed, and indexed into the Pinecone vector store.")
    log_info("Summary:", Colors.BOLD)
    log_info(f"Total URLs Mapped: {len(site_map['results'])}")
    log_info(f"Total Documents Extracted: {len(all_documents)}")
    log_info(f"Total Document Chunks Created: {len(splitted_documents)}")





# =========================
# Script Entry Point
# =========================

# Check if script is executed directly
if __name__ == "__main__":

    # Run the async main function
    asyncio.run(main())