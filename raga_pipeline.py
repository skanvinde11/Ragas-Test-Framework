# rag_pipeline.py

# --- Core LangChain and Ollama Imports ---
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_ollama import \
    OllamaEmbeddings
# --- B. RAG Setup Functions ---
# Note: Other imports (Chroma, Loaders, Splitter) are defined inside the function
# where they are used, or kept global if convenient.

# Define the constants globally for easy modification
FINANCE_DOCS_PATH = "C:/Users/nasak/PycharmProjects/pythonProject1/finance_docs"
LLM = ChatOllama(model="mistral")
EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")


def setup_rag_system():
    """Initializes and returns the vectorstore retriever from documents."""

    # Imports needed ONLY for data loading and splitting
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    import os

    print(f"--- 1. Loading documents from: {FINANCE_DOCS_PATH} ---")

    loader = DirectoryLoader(
        path=FINANCE_DOCS_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    docs = loader.load()

    print(f"--- 2. Splitting {len(docs)} documents into chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    print(f"--- 3. Creating Chroma Vector Store and Retriever ---")
    #  uses the global embedding model
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=EMBEDDING_MODEL,
        persist_directory="./chroma_db"
    )

    # Define the RETRIEVER
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return retriever


def create_rag_chain(retriever):
    """Defines and returns the LCEL RAG chain and the retrieval-only chain."""

    def format_docs(docs):
        """Formats the retrieved document chunks into a single string, including source/page."""
        formatted_texts = []
        for doc in docs:
            # handle the case where metadata might be missing source/page
            source = doc.metadata.get('source', 'Unknown Source').split('/')[-1].split('\\')[-1]  # Clean up path
            page = doc.metadata.get('page', 'N/A')
            formatted_texts.append(f"Source: {source} (Page {page})\nContent: {doc.page_content}")
        return "\n\n".join(formatted_texts)

    # Modify the prompt to guide the LLM as a financial expert
    rag_prompt = ChatPromptTemplate.from_template(
        """You are an expert financial research assistant. 
        Use the following pieces of retrieved context to answer the question accurately and professionally. 
        Always cite the source document and page number (e.g., Source: report.pdf (Page 5)) for every fact you state. 
        If you don't know the answer or the information is not in the context, state: 
        'The specific information requested is not available in the provided financial documents.'

        CONTEXT:
        {context}

        QUESTION: {question}
        """
    )

    # The main RAG chain (Retrieval -> Formatting -> Prompt -> LLM -> Output)
    rag_chain = (
            RunnableParallel({
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            })
            | rag_prompt
            | LLM
            | StrOutputParser()
    )

    # A helper chain for RAGAS: returns the raw documents (contexts) and the question
    retrieval_chain = RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough(),
    })

    return rag_chain, retrieval_chain


# -----------------------------------------------------
# --- C. Main Execution (for Demo/Testing) ---
# -----------------------------------------------------

if __name__ == "__main__":
    # This section runs only when rag_pipeline.py is executed directly
    retriever = setup_rag_system()
    rag_chain, _ = create_rag_chain(retriever)

    # 6. Invoke the Chain with a relevant financial question
    financial_question = "What was the underlying loss before tax for the Non-core and Legacy segment in Q2 2025?"
    response = rag_chain.invoke(financial_question)
    print(f"\n--- Running Demo Question ---\nQUESTION: {financial_question}")
    print(f"--- ANSWER ---\n{response}")