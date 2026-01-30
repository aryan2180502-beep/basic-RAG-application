"""
Verify ChromaDB Embeddings Storage
This script checks if embeddings are properly stored in ChromaDB
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "techgear_products"


def verify_chroma_db():
    """Verify if ChromaDB exists and contains embeddings."""
    print("=" * 70)
    print("ChromaDB Verification")
    print("=" * 70)
    
    # Check if directory exists
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"\n✗ ChromaDB directory not found at: {CHROMA_DB_PATH}")
        print("\n📋 To create the database, run:")
        print("   python load_knowledge_base.py")
        return False
    
    print(f"\n✓ ChromaDB directory exists at: {CHROMA_DB_PATH}")
    
    try:
        # Initialize embeddings
        print(f"\nInitializing embeddings model: {EMBEDDING_MODEL}")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print("✓ Embeddings model loaded")
        
        # Load the vector store
        print(f"\nLoading ChromaDB vector store...")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME
        )
        print("✓ Vector store loaded successfully")
        
        # Get collection info
        collection = vector_store._collection
        count = collection.count()
        
        print(f"\n📊 Database Statistics:")
        print(f"  - Collection name: {COLLECTION_NAME}")
        print(f"  - Total documents: {count}")
        
        if count == 0:
            print("\n⚠ Warning: Database exists but contains no documents!")
            print("   Run: python load_knowledge_base.py")
            return False
        
        # Test retrieval
        print(f"\n🔍 Testing retrieval with sample query...")
        test_query = "SmartWatch Pro X"
        results = vector_store.similarity_search(test_query, k=3)
        
        print(f"✓ Successfully retrieved {len(results)} documents")
        print(f"\nSample retrieved content:")
        for i, doc in enumerate(results[:2], 1):
            print(f"\n  Document {i}:")
            print(f"  {doc.page_content[:150]}...")
        
        # Test with another query
        print(f"\n🔍 Testing with product query...")
        test_query2 = "What are the prices of wireless earbuds?"
        results2 = vector_store.similarity_search(test_query2, k=2)
        
        print(f"✓ Retrieved {len(results2)} relevant documents")
        print(f"\nRelevant content:")
        for i, doc in enumerate(results2, 1):
            print(f"\n  Document {i}:")
            print(f"  {doc.page_content[:200]}...")
        
        print("\n" + "=" * 70)
        print("✓ ChromaDB verification completed successfully!")
        print("  Embeddings are properly stored and retrievable.")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error verifying ChromaDB: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    verify_chroma_db()
