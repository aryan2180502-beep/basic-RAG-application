"""
RAG Chain Implementation for TechGear Electronics Customer Support
This script:
1. Creates a retriever from ChromaDB
2. Builds a RAG chain that retrieves context and generates answers
3. Uses Google's Gemini model via LangChain
"""

import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document

# Load environment variables from .env file
load_dotenv()

# Configuration
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash-lite"
COLLECTION_NAME = "techgear_products"
TOP_K_RESULTS = 4  # Number of relevant chunks to retrieve


class TechGearRAGChain:
    """RAG Chain for TechGear Electronics Customer Support."""
    
    def __init__(self, google_api_key: str = None):
        """
        Initialize the RAG chain.
        
        Args:
            google_api_key: Google API key for Gemini. If None, reads from GOOGLE_API_KEY env variable.
        """
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError(
                "Google API key not found. Set GOOGLE_API_KEY environment variable "
                "or pass google_api_key parameter."
            )
        
        self.embeddings = None
        self.vector_store = None
        self.retriever = None
        self.llm = None
        self.rag_chain = None
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_vector_store()
        self._initialize_retriever()
        self._initialize_llm()
        self._initialize_rag_chain()
    
    def _initialize_embeddings(self):
        """Initialize the embedding model."""
        print("Initializing embeddings...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print("✓ Embeddings initialized")
    
    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store."""
        print("Loading ChromaDB vector store...")
        
        if not os.path.exists(CHROMA_DB_PATH):
            raise FileNotFoundError(
                f"ChromaDB not found at {CHROMA_DB_PATH}. "
                "Please run load_knowledge_base.py first."
            )
        
        self.vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME
        )
        print("✓ Vector store loaded")
    
    def _initialize_retriever(self):
        """Create a retriever from ChromaDB."""
        print("Creating retriever from ChromaDB...")
        
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_RESULTS}
        )
        print(f"✓ Retriever created (retrieving top {TOP_K_RESULTS} results)")
    
    def _initialize_llm(self):
        """Initialize Google Gemini LLM."""
        print("Initializing Google Gemini LLM...")
        
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=self.google_api_key,
            temperature=0.3,  # Lower temperature for more consistent answers
            convert_system_message_to_human=True
        )
        print(f"✓ Gemini LLM initialized ({GEMINI_MODEL})")
    
    def _initialize_rag_chain(self):
        """Build the RAG chain."""
        print("Building RAG chain...")
        
        # Define the prompt template
        prompt_template = """You are a helpful customer support assistant for TechGear Electronics.
Use the following context from our product knowledge base to answer the customer's question accurately and professionally.

If the answer is not found in the context, politely say you don't have that information and suggest contacting support.

Context:
{context}

Customer Question: {question}

Assistant Response:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create the RAG chain
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Stuffs all retrieved documents into the prompt
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        print("✓ RAG chain built successfully")
    
    def query(self, question: str, verbose: bool = False) -> Dict:
        """
        Query the RAG chain with a customer question.
        
        Args:
            question: Customer question
            verbose: If True, print retrieved context
        
        Returns:
            Dict with 'result' (answer) and 'source_documents' (retrieved chunks)
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Question: {question}")
            print(f"{'='*70}")
        
        # Get response from RAG chain
        response = self.rag_chain({"query": question})
        
        if verbose:
            print(f"\n📚 Retrieved Context ({len(response['source_documents'])} chunks):")
            for i, doc in enumerate(response['source_documents'], 1):
                print(f"\nChunk {i}:")
                print(f"{doc.page_content[:200]}...")
            
            print(f"\n💬 Assistant Response:")
            print(f"{response['result']}")
            print(f"{'='*70}\n")
        
        return response
    
    def get_answer(self, question: str) -> str:
        """
        Get just the answer without source documents.
        
        Args:
            question: Customer question
        
        Returns:
            Answer string
        """
        response = self.query(question, verbose=False)
        return response['result']


def main():
    """Demo function to test the RAG chain."""
    print("=" * 70)
    print("TechGear Electronics - RAG Chain Demo")
    print("=" * 70)
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n⚠ WARNING: GOOGLE_API_KEY environment variable not set!")
        print("Please set your Google API key:")
        print("  export GOOGLE_API_KEY='your-api-key-here'")
        print("\nOr get one from: https://makersuite.google.com/app/apikey")
        return
    
    try:
        # Initialize RAG chain
        print("\nInitializing RAG chain...")
        rag = TechGearRAGChain()
        
        print("\n✓ RAG chain ready!")
        print("=" * 70)
        
        # Test queries
        test_questions = [
            "What is the price of the SmartWatch Pro X?",
            "Tell me about the warranty on wireless earbuds",
            "Do you have any gaming laptops? What are the specs?",
            "What is your return policy?",
            "How can I contact customer support?"
        ]
        
        print("\n🧪 Testing RAG chain with sample questions...\n")
        
        for question in test_questions:
            response = rag.query(question, verbose=True)
        
        print("=" * 70)
        print("✓ RAG chain demo completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
