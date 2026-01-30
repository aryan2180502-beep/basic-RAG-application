"""
Load Knowledge Base into ChromaDB with LangChain Text Splitting
This script:
1. Reads the knowledge base from knowledge_base.txt
2. Splits it into chunks using LangChain's TextSplitter
3. Creates embeddings using a default embedding model
4. Stores embeddings in ChromaDB
"""

import os
import sys
from pathlib import Path
from typing import List

# Import LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
KNOWLEDGE_BASE_PATH = "knowledge_base.txt"
CHROMA_DB_PATH = "./chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight, efficient embedding model


def load_knowledge_base(file_path: str) -> str:
    """Load knowledge base from text file."""
    print(f"Loading knowledge base from {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Knowledge base file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"✓ Knowledge base loaded successfully ({len(content)} characters)")
    return content


def split_knowledge_base(content: str, chunk_size: int = CHUNK_SIZE, 
                        chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split knowledge base into chunks using RecursiveCharacterTextSplitter."""
    print(f"\nSplitting knowledge base into chunks...")
    print(f"  - Chunk size: {chunk_size}")
    print(f"  - Chunk overlap: {chunk_overlap}")
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]  # Split by paragraphs first, then sentences
    )
    
    # Split the content
    chunks = text_splitter.split_text(content)
    
    print(f"✓ Created {len(chunks)} chunks from knowledge base")
    print(f"  - Average chunk size: {sum(len(c) for c in chunks) / len(chunks):.0f} characters")
    
    return chunks


def create_embeddings_and_store_in_chroma(chunks: List[str], 
                                         db_path: str = CHROMA_DB_PATH,
                                         embedding_model: str = EMBEDDING_MODEL) -> Chroma:
    """Create embeddings and store in ChromaDB."""
    print(f"\nCreating embeddings and storing in ChromaDB...")
    print(f"  - Embedding model: {embedding_model}")
    print(f"  - Database path: {db_path}")
    
    # Initialize embeddings
    print("  - Initializing embedding model (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # Create or load ChromaDB vector store
    print("  - Creating/updating ChromaDB vector store...")
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=db_path,
        collection_name="techgear_products"
    )
    
    # Persist the database
    vector_store.persist()
    
    print(f"✓ Successfully stored {len(chunks)} chunks in ChromaDB")
    print(f"  - Collection name: techgear_products")
    
    return vector_store


def verify_database(vector_store: Chroma) -> None:
    """Verify the database by performing a test query."""
    print(f"\nVerifying database with test query...")
    
    test_query = "What is the price of SmartWatch Pro X?"
    results = vector_store.similarity_search(test_query, k=3)
    
    print(f"✓ Test query successful! Retrieved {len(results)} relevant chunks:")
    for i, result in enumerate(results, 1):
        print(f"\n  Chunk {i}:")
        print(f"  {result.page_content[:150]}...")


def main():
    """Main function to load and process the knowledge base."""
    print("=" * 70)
    print("TechGear Electronics - Knowledge Base Loader")
    print("=" * 70)
    
    try:
        # Step 1: Load knowledge base
        content = load_knowledge_base(KNOWLEDGE_BASE_PATH)
        
        # Step 2: Split into chunks
        chunks = split_knowledge_base(content)
        
        # Step 3: Create embeddings and store in ChromaDB
        vector_store = create_embeddings_and_store_in_chroma(chunks)
        
        # Step 4: Verify the database
        verify_database(vector_store)
        
        print("\n" + "=" * 70)
        print("✓ Knowledge base successfully loaded into ChromaDB!")
        print("=" * 70)
        
        return vector_store
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    vector_store = main()
