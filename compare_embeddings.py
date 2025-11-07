#compare_embeddings.py


import os
from dotenv import load_dotenv
import numpy as np

# --- Local Embedding Model Imports (FIXED) ---
# FIX 1: Import HuggingFaceEmbeddings from the dedicated new package
from langchain_huggingface import HuggingFaceEmbeddings

# --- LangChain Core Imports (FIXED) ---
# FIX 2: load_evaluator is now imported from langchain_core
from langchain.evaluation import load_evaluator 
from langchain_core.embeddings import Embeddings # Good practice to import base class

# Load environment variables
load_dotenv()

# --- Configuration ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
# --- End Configuration ---

def main():
    """
    Demonstrates using the local HuggingFace Embeddings model to:
    1. Generate an embedding vector for a single word.
    2. Compare the semantic distance between two words using LangChain's evaluator.
    
    NOTE: This script requires 'numpy' to be installed: pip install numpy
    """
    
    print(f"Initializing local Embedding Model: {EMBEDDING_MODEL}...")

    # Initialize HuggingFace Embeddings
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        # Explicitly set device to CPU for broad compatibility
        model_kwargs={'device': 'cpu'}
    )

    # 1. Get embedding for a word.
    print("\n--- Testing Single Embedding ---")
    vector = embedding_function.embed_query("apple")
    
    # Convert to NumPy array for cleaner print formatting
    try:
        vector_np = np.array(vector)
    except NameError:
        vector_np = vector 

    print(f"Vector for 'apple' starts with: {vector_np[:5]}...")
    print(f"Vector length: {len(vector_np)}")
    print(f"Expected length for this model is 384.")

    # 2. Compare vector of two words
    # The 'pairwise_embedding_distance' evaluator works with any valid embedding function.
    print("\n--- Initializing Distance Evaluator ---")
    evaluator = load_evaluator(
        "pairwise_embedding_distance", 
        embeddings=embedding_function
    )
    
    # Test case 1: Semantically related words (should have a lower score)
    words_related = ("apple", "iphone")
    print(f"\n--- Comparing Semantically Related Words ({words_related[0]}, {words_related[1]}) ---")
    x_related = evaluator.evaluate_string_pairs(
        prediction=words_related[0], 
        prediction_b=words_related[1]
    )
    print(f"Embedding Distance Score: {x_related['score']:.4f}")
    print("Interpretation: Lower scores indicate greater semantic similarity.")
    
    # Test case 2: Semantically unrelated words (should have a higher score)
    words_unrelated = ("apple", "truck")
    print(f"\n--- Comparing Semantically Unrelated Words ({words_unrelated[0]}, {words_unrelated[1]}) ---")
    x_unrelated = evaluator.evaluate_string_pairs(
        prediction=words_unrelated[0], 
        prediction_b=words_unrelated[1]
    )
    print(f"Embedding Distance Score: {x_unrelated['score']:.4f}")
    print("Interpretation: Higher scores indicate less semantic similarity.")


if __name__ == "__main__":
    # Ensure numpy is installed, as it's a dependency for this script's display features.
    if 'numpy' not in locals():
        print("⚠️ Warning: Numpy is not available. Install with 'pip install numpy' for optimal display.")
        
    main()