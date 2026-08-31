# =================
# Load Dependencies
# =================

from utils import format_box
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================
# Load Environment Variables
# ==========================

load_dotenv()

# ========================
# Initialize website links
# ========================
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# ========================
# Load and Split Documents
# =========================

# Get the document content from the urls
url_contents = [UnstructuredLoader(web_url=url, chunking_strategy="basic", max_characters=1000000).load() for url in urls]
print(format_box(f"URL Results [Count: {len(url_contents)}]:\n\n {url_contents}", width=150))

# Flatten the list of lists into a single list of documents
documents = [document for url_content in url_contents for document in url_content]
print(format_box(f"Flattened Documents [Count: {len(documents)}]:\n\n {documents}", width=150))

# Split the documents into chunks of 250 characters with no overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
document_splits = text_splitter.split_documents(documents=documents)
print(format_box(f"Document Splits [Count: {len(document_splits)}]:\n\n {document_splits}", width=150))


# # ===================
# # Create Vector Store
# # ===================
# # TODO: After first run comment out the following lines to avoid re-creating 
# # the vector store and re-embedding the documents. The vector store will 
# # be persisted in the .chroma directory.
# vectorstore = Chroma.from_documents(
#     documents=document_splits,
#     collection_name="rag-chroma",
#     embedding=OllamaEmbeddings(model="qwen3-embedding:8b"),
#     persist_directory="./.chroma",
# )
# print(format_box(f"Vector Store Created [Count: {vectorstore._collection.count()}]:\n\n{vectorstore}", width=150))

# ====================================
# Retrieve Documents from Vector Store
# ====================================
retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory="./.chroma",
    embedding_function=OllamaEmbeddings(model="qwen3-embedding:8b"),
).as_retriever(search_kwargs={"k": 3})
print(format_box(f"Retriever Created:\n\n{retriever}", width=150))