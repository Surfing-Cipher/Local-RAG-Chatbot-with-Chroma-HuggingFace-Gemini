#create_database.py

import os
import shutil
from dotenv import load_dotenv

# --- Local Embedding Model Imports ---
# **THIS IS THE FIX for the first warning**
from langchain_huggingface import HuggingFaceEmbeddings

# --- LangChain Core/Community Imports ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

# Configuration and Paths
CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

# Using the local embedding model to avoid API quota issues
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

load_dotenv()


def main():
    """
    Main function to load documents, split them into chunks, and save
    them to a Chroma vector store using a local (HuggingFace) Embeddings model.
    """
    GEMINI_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not GEMINI_KEY:
        print("💡 NOTE: GOOGLE_API_KEY not found. This key is ONLY required for a final query step (not in this script).")
        print("   The database creation is 100% local and will succeed.\n")

    generate_data_store()


def generate_data_store():
    """
    Orchestrates the data loading, splitting, and saving process.
    """
    documents = load_documents()
    if not documents:
        print("No documents found. Exiting.")
        return
    
    chunks = split_text(documents)
    if not chunks:
        print("No chunks were created. Exiting.")
        return
        
    save_to_chroma(chunks)


def load_documents() -> list[Document]:
    """
    Loads documents from the specified DATA_PATH, using TextLoader and
    explicitly setting the encoding to 'utf-8' to prevent decoding errors.
    """
    print(f"Loading documents from {DATA_PATH}...")

    loader = DirectoryLoader(
        DATA_PATH,
        glob="*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True
    )
    
    try:
        documents = loader.load()
        if not documents:
            print(f"⚠️ No documents found in '{DATA_PATH}' matching '*.md'.")
            return []
        print(f"Loaded {len(documents)} documents.")
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []


def split_text(documents: list[Document]) -> list[Document]:
    """
    Splits documents into smaller, overlapping chunks suitable for embedding.
    """
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    if chunks:
        print("\n--- Sample Chunk Content ---")
        print(chunks[0].page_content)
        print(f"Metadata: {chunks[0].metadata}")
        print("----------------------------\n")

    return chunks


def save_to_chroma(chunks: list[Document]):
    """
    Initializes the local HuggingFace Embeddings model and saves document chunks
    to the Chroma vector store.
    """
    if os.path.exists(CHROMA_PATH):
        print(f"Removing existing database at {CHROMA_PATH}...")
        shutil.rmtree(CHROMA_PATH)

    # Using the new HuggingFaceEmbeddings import
    print(f"Initializing local embedding model: {EMBEDDING_MODEL}...")
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'} # Force CPU for compatibility
    )

    print(f"Creating new Chroma DB at {CHROMA_PATH}...")

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks,
        embedding_function,
        persist_directory=CHROMA_PATH
    )
    
    print(f"✅ Successfully saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()

