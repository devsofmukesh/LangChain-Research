import os
from dotenv import load_dotenv
from langchain_classic import hub
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Execute this block only if the script is run directly
if __name__ == "__main__":
    # Print a simple greeting
    print("Hi there!")
    # Initialize the PDF loader with the given file path
    loader = PyPDFLoader(file_path=os.path.join(os.getcwd(), "ReAct.pdf"))
    # Load the PDF into document objects
    documents = loader.load()
    # Initialize text splitter to chunk the document text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # Split the documents into smaller chunks
    dcoument_chunks = text_splitter.split_documents(documents=documents)
    # Create an embeddings model for vectorization
    embeddings = OllamaEmbeddings(model="llama3.1:8b")
    # Build a FAISS vector store from the document chunks
    vectorstore = FAISS.from_documents(documents=dcoument_chunks, embedding=embeddings)
    # Save the FAISS index locally
    vectorstore.save_local(folder_path="FAISS_INDEX_REACT")
    # Load the saved FAISS index with permission for unsafe deserialization
    new_vectorstore = FAISS.load_local(folder_path="FAISS_INDEX_REACT", embeddings=embeddings, allow_dangerous_deserialization=True)
    # Pull a retrieval QA chat prompt template from LangChain hub
    retrieval_qa_chat_prompt = hub.pull(owner_repo_commit="langchain-ai/retrieval-qa-chat")
    # Create a chain that formats and processes retrieved documents
    combine_docs_chain = create_stuff_documents_chain(llm=ChatOllama(model="llama3.1:8b"), prompt=retrieval_qa_chat_prompt)
    # Build the retrieval chain using the FAISS retriever and document-combination chain
    retrieval_chain = create_retrieval_chain(retriever=new_vectorstore.as_retriever(search_kwargs={"k": 8}), combine_docs_chain=combine_docs_chain)
    # Run the retrieval chain with a query asking for a gist of ReAct
    result = retrieval_chain.invoke(input={"input": "Give me the gist of ReAct in 3 sentences."})
    # Print the final answer returned by the chain
    print(result["answer"])
