import os
from dotenv import load_dotenv
import streamlit as st
import numpy as np # <-- NEW: Needed for vector math
from typing import List, Tuple

# --- LangChain Core/Utility Imports ---
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document

# --- Local Embedding & Vector Store Imports ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Gemini LLM Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI


# --- Configuration ---
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

# --- NEW: Cosine Similarity Function for Scoring ---
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calculates the cosine similarity between two vectors. Higher score = more relevant."""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)
# --- END NEW FUNCTION ---

@st.cache_resource
def get_retriever() -> Tuple[Chroma, HuggingFaceEmbeddings]:
    """
    Initializes the embeddings and loads the Chroma DB, caching the result.
    Returns both the retriever and the embedding_function (needed for scoring).
    """
    try:
        # 1. Load Embeddings
        embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        
        # 2. Load Database
        db = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embedding_function
        )
        
        # 3. Define the Retriever
        # Search for the top 3 chunks (k=3)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        
        # Return both the retriever and the function for scoring
        return retriever, embedding_function
        
    except Exception as e:
        st.error(f"❌ Error loading RAG database: {e}")
        st.info("Please ensure you ran 'python create_database.py' successfully.")
        # Return None for both if there's an error
        return None, None


def run_rag_query(query: str, retriever):
    """
    Executes the RAG query using the loaded retriever and Gemini.
    Returns the answer, the source documents (context), and the original query.
    """
    # Check for API Key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        # Fallback to retrieval-only mode if no key is found
        chunks = retriever.invoke(query)
        # We return the query as the third element for scoring later
        return (
            "⚠️ Retrieval Only Mode: No API key found. Showing context only.", 
            chunks,
            query
        )

    # --- FULL RAG MODE (API KEY PRESENT) ---
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL, 
        google_api_key=api_key,
        temperature=0.1,
        # Forces synchronous transport, resolving "no current event loop" issue with Streamlit
        transport="rest" 
    )

    # Define RAG Chain Components
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Execute the Query
    result = retrieval_chain.invoke({"input": query})
    
    # Return answer, context, and the original query
    return result['answer'], result['context'], query


def display_sources(query: str, chunks: List[Document], embeddings: HuggingFaceEmbeddings):
    """
    Displays retrieved chunks, calculates cosine similarity score against the query, 
    and displays the chunks sorted by score.
    """
    
    with st.expander("📚 **Sources Retrieved & Relevance Scores**", expanded=True):
        if not chunks:
            st.warning("No context was retrieved for this query.")
            return
        
        # 1. Get embedding for the user's query
        query_vector = np.array(embeddings.embed_query(query))
        
        scored_chunks = []
        # 2. Score each retrieved chunk
        for doc in chunks:
            try:
                # Get embedding for the chunk content
                chunk_vector = np.array(embeddings.embed_query(doc.page_content))
                # Calculate the score (cosine similarity)
                score = cosine_similarity(query_vector, chunk_vector)
            except Exception:
                # Fallback if embedding fails
                score = -1.0 

            scored_chunks.append((score, doc))
            
        # 3. Sort by score descending (most relevant first)
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        st.markdown(
            "*(Relevance Score reflects the cosine similarity between the query and the chunk. Higher score = more relevant.)*"
        )
        
        # 4. Display sorted, scored results
        for i, (score, doc) in enumerate(scored_chunks):
            # Extract basic metadata
            source = doc.metadata.get('source', 'N/A').split('\\')[-1]
            start_index = doc.metadata.get('start_index', 'N/A')

            st.markdown(f"**{i+1}. Chunk (Score: `{score:.4f}`):** `Source: {source} (Index: {start_index})`")
            st.code(doc.page_content)


def main_streamlit_app():
    st.set_page_config(page_title="RAG Chatbot (Gemini + Chroma)", layout="wide")
    st.title("📚 Local RAG Chatbot with Scoring (Alice in Wonderland)")
    st.caption(f"Powered by **Gemini-{LLM_MODEL}** and **{EMBEDDING_MODEL}** embeddings.")
    
    # Load the retriever and the embedding function (cached)
    retriever, embedding_function = get_retriever()
    if not retriever:
        return

    # User Input
    query = st.text_input(
        "Ask a question about the book:",
        placeholder="e.g., What did the Cheshire Cat say to Alice?"
    )

    if st.button("Generate Answer", use_container_width=True) and query:
        # Display a spinner while processing
        with st.spinner(f"Searching database and consulting Gemini..."):
            answer, context, original_query = run_rag_query(query, retriever)
        
        # Display the Results
        st.subheader("Generated Answer")
        st.info(answer)
        
        # Display the Sources with Scoring
        display_sources(original_query, context, embedding_function)
    
    st.sidebar.header("Configuration")
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        st.sidebar.error("⚠️ Gemini API Key Missing.")
        st.sidebar.info("The app is currently running in retrieval-only mode.")
    else:
        st.sidebar.success("✅ Gemini API Key Loaded.")


if __name__ == "__main__":
    main_streamlit_app()