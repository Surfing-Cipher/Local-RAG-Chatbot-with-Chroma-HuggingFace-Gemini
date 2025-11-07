#query_data.py


import os
import argparse
from dotenv import load_dotenv

# --- Modern LangChain Core Imports ---
from langchain_core.prompts import ChatPromptTemplate
# Explicitly use the full path for both functions
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain 
from langchain_core.documents import Document

# --- Local Embedding Imports (Modernized) ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Gemini LLM Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI


# Configuration
CHROMA_PATH = "chroma"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
LLM_MODEL = "gemini-2.5-flash"
load_dotenv() # Load key from .env file

# RAG Prompt Template
RAG_PROMPT_TEMPLATE = """
You are an intelligent assistant for question-answering tasks. 
Use ONLY the following context to answer the user's question. If the answer is not
found in the context, politely state that you cannot find the information in the provided text.

Context: {context}

Question: {input}
"""


def main():
    """
    Main function to parse arguments and initiate the RAG query process.
    """
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Query your RAG Database using Gemini.")
    parser.add_argument("query_text", type=str, nargs='?', default=None, help="The text query to search for.")
    args = parser.parse_args()
    
    # 2. Check API Key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if args.query_text is None:
        print("\n🚫 Error: Please provide a query. Example usage:")
        print("python query_data.py \"What did Alice say about the Cheshire Cat?\"")
        return

    # 3. Execute Query
    query_data(args.query_text, api_key)


def display_sources(query: str, chunks: list[Document]):
    """Displays the retrieved chunks in the console."""
    if query:
        print(f"\n❓ Query: {query}")
        print("\n------------------\n")
        print("✨ RETRIEVAL-ONLY MODE: No API key found. Displaying retrieved chunks instead of a generated answer.")
    
    print("\n📚 Sources Used (Top Chunks):\n")
    
    if not chunks:
        print("No context was retrieved.")
        return
        
    for i, doc in enumerate(chunks):
        if not isinstance(doc, Document):
            doc = Document(page_content=doc.get('page_content', 'N/A'), metadata=doc.get('metadata', {}))

        print(f"--- Document {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'N/A')}")
        print(f"Content Start: {doc.metadata.get('start_index', 'N/A')}")
        print(f"Content Preview: {doc.page_content[:150]}...")
        print("-" * 20)


def query_data(query: str, api_key: str):
    """
    Loads the Chroma DB, initializes the LLM (if key is present), and performs the RAG query 
    or falls back to retrieval-only mode.
    """
    print(f"Loading database with model: {EMBEDDING_MODEL}...")

    # 1. Load Embeddings and Database
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )
    
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embedding_function
        )
    except Exception as e:
        print(f"❌ ERROR: Could not load Chroma database from {CHROMA_PATH}.")
        print("Please ensure you ran 'python create_database.py' successfully.")
        print(f"Details: {e}")
        return
    
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # --- LOCAL FALLBACK CHECK (Retrieval-Only Mode) ---
    if not api_key:
        chunks = retriever.invoke(query)
        display_sources(query, chunks)
        return

    # --- FULL RAG MODE (API KEY PRESENT) ---
    print(f"--- Running full RAG pipeline with Gemini ({LLM_MODEL})... ---")
    
    # 2. Initialize the Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL, 
        google_api_key=api_key,
        temperature=0.1
    )

    # 3. Define RAG Chain Components (LCEL)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # 4. Execute the Query
    print(f"Executing query...")
    result = retrieval_chain.invoke({"input": query})

    # 5. Output Result
    print(f"\n❓ Query: {query}")
    print("\n------------------\n")
    print(f"🤖 Answer (Powered by Gemini): {result['answer']}")
    
    print("\n📚 Sources Used (Context):")
    display_sources("", result['context'])


if __name__ == "__main__":
    main()

