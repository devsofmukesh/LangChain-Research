# =========================
# Load Dependencies
# =========================

from typing import Any
from typing import Dict
from typing import List
from backend.consts import INDEX_NAME
from dotenv import load_dotenv
from langchain_classic import hub
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever

# =========================
# Environment & SSL Setup
# =========================

load_dotenv()

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    llm = ChatOllama(model="llama3.1:8b", verbose=True, temperature=0)
    rephrase_prompt = hub.pull(owner_repo_commit="langchain-ai/chat-langchain-rephrase")
    print("Rephrase Prompt:", rephrase_prompt)
    retrieval_qa_chat_prompt = hub.pull(owner_repo_commit="langchain-ai/retrieval-qa-chat")
    print("\nRetrieval QA Chat Prompt:", retrieval_qa_chat_prompt)
    document_prompt = PromptTemplate.from_template("{page_content}")
    stuff_documents_chain = create_stuff_documents_chain(llm=llm, prompt=retrieval_qa_chat_prompt, document_prompt=document_prompt, document_separator="\n\n",)
    history_aware_retriever = create_history_aware_retriever(llm=llm, retriever=docsearch.as_retriever(), prompt=rephrase_prompt)
    qa = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain)
    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    rephrased_result = {
        "query": result.get("input"),
        "result": result.get("answer"),
        "source_documents": result.get("context"),
    }
    return rephrased_result

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def run_llm2(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    llm = ChatOllama(model="llama3.1:8b", verbose=True, temperature=0)
    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    rag_chain = (
        {"context": docsearch.as_retriever() | format_docs, "input": RunnablePassthrough()}
        | retrieval_qa_chat_prompt
        | llm
        | StrOutputParser()
    )

    retrieve_docs_chain = (lambda x: x["input"]) | docsearch.as_retriever()
    chain = RunnablePassthrough.assign(context=retrieve_docs_chain).assign(answer=rag_chain)
    result = chain.invoke({"input": query, "chat_history": chat_history})
    return result

if __name__ == "__main__":
    chat_history = []
    response = run_llm(query="What is LangChain chain?", chat_history=chat_history)
    print(response.get("result"))