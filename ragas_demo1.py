from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- A. Setup LLM (Using the recommended, fixed import) ---
# NOTE: You MUST run 'pip install -U langchain-ollama' first!
from langchain_ollama import ChatOllama
llm = ChatOllama(model="mistral")

# --- B. RAG Setup (FIX for 'retriever' not defined) ---

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# 1. Load Data (Replace 'my_document.txt' with your file path)
# Assuming you have a file to load for RAG
loader = TextLoader("./my_document.txt") # <-- Change this path!
docs = loader.load()

# 2. Split Documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 3. Create Embeddings and Vector Store
# Use OllamaEmbeddings for consistency and local execution
embedding_model = OllamaEmbeddings(model="nomic-embed-text") # <-- RECOMMENDED CHANGE

print(f"Type of splits: {type(splits)}")
print(f"Number of documents: {len(splits)}")
print(f"First document content (start): {splits[0].page_content[:50]}...")

# This line now uses the dedicated embedding model
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)

# 4. Define the RETRIEVER
retriever = vectorstore.as_retriever()

# --- C. LCEL Pipeline Definition ---

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know.

    CONTEXT:
    {context}

    QUESTION: {question}
    """
)

# 5. Build the LCEL RAG Chain
rag_chain = (
    RunnableParallel({
        "context": retriever | format_docs, # <-- Now this works!
        "question": RunnablePassthrough(),
    })
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 6. Invoke the Chain
response = rag_chain.invoke("What is the main topic of the document?")
print(response)