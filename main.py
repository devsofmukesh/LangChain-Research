import os
from dotenv import load_dotenv
from langchain_classic import hub
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

def format_docs(docs):
    """Formats documents by extracting their page content and joining them with double newlines."""
    return "\n\n".join([doc.page_content for doc in docs])


if __name__ == "__main__":
    # Display startup message
    print("Retrieving!...")

    embeddings = OllamaEmbeddings(model="llama3.1:8b")
    llm = ChatOllama(model="llama3.1:8b")

    query = "What is Pinecone in Machine Learning?"
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})
    print("Response (1):", result.content)

    #----------------------------------------------------------------------

    vectorstore = PineconeVectorStore(embedding=embeddings, index_name=os.getenv("PINECONE_INDEX_NAME"))

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=retrieval_qa_chat_prompt)

    retrieval_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)

    result = retrieval_chain.invoke(input={"input": query})
    print("Response (2):", result)

    #----------------------------------------------------------------------

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer concise and to the point.
    Always say "thanks for asking!" at the end of the answer.
    
    {context}
    
    Question: {question}
    
    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template=template)

    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    result = rag_chain.invoke(input=query)
    print("Response (3):", result)