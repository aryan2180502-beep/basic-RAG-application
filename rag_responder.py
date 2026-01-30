"""
RAG Responder Agent - Node 2
Uses RAG to generate responses based on retrieved context
"""

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()


class RAGResponder:
    """Agent to generate responses using RAG."""
    
    def __init__(self, api_key: str = None, chroma_path: str = "./chroma_db"):
        """Initialize the RAG responder."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.chroma_path = chroma_path
        
        # Initialize embeddings
        print("Loading embeddings...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load vector store
        print("Loading ChromaDB...")
        self.vector_store = Chroma(
            persist_directory=chroma_path,
            embedding_function=self.embeddings,
            collection_name="techgear_products"
        )
        
        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=self.api_key,
            temperature=0.3
        )
        
        # Define RAG prompt templates for different categories
        self.prompts = {
            "products": ChatPromptTemplate.from_messages([
                ("system", """You are a helpful customer support assistant for TechGear Electronics.
Use the following product information to answer the customer's question about our products.

Product Information:
{context}

Provide a clear, accurate response. If the information isn't in the context, say so politely."""),
                ("human", "{query}")
            ]),
            
            "returns": ChatPromptTemplate.from_messages([
                ("system", """You are a helpful customer support assistant for TechGear Electronics.
Use the following information to answer the customer's question about returns, refunds, or exchanges.

Information:
{context}

Provide a clear, professional response about our return policies."""),
                ("human", "{query}")
            ]),
            
            "general": ChatPromptTemplate.from_messages([
                ("system", """You are a helpful customer support assistant for TechGear Electronics.
Use the following information to answer the customer's general question.

Information:
{context}

Provide a helpful, professional response. Include contact information if relevant."""),
                ("human", "{query}")
            ])
        }
    
    def _format_docs(self, docs):
        """Format retrieved documents into a context string."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def respond(self, query: str, category: str = "general") -> dict:
        """
        Generate a response using RAG.
        
        Args:
            query: The customer query
            category: The query category (products, returns, general)
            
        Returns:
            Dictionary with response and retrieved documents
        """
        try:
            # Get appropriate prompt
            prompt = self.prompts.get(category, self.prompts["general"])
            
            # Create RAG chain
            rag_chain = (
                {
                    "context": self.retriever | self._format_docs,
                    "query": RunnablePassthrough()
                }
                | prompt
                | self.llm
                | StrOutputParser()
            )
            
            # Retrieve documents
            docs = self.retriever.invoke(query)
            
            # Generate response
            response = rag_chain.invoke(query)
            
            return {
                "response": response,
                "retrieved_docs": [doc.page_content[:200] + "..." for doc in docs],
                "num_docs": len(docs),
                "success": True
            }
            
        except Exception as e:
            return {
                "response": f"I apologize, but I encountered an error processing your request. Please contact support@techgear.com",
                "retrieved_docs": [],
                "num_docs": 0,
                "success": False,
                "error": str(e)
            }


def test_rag_responder():
    """Test the RAG responder with sample queries."""
    print("=" * 70)
    print("Testing RAG Responder Agent")
    print("=" * 70)
    
    responder = RAGResponder()
    
    test_cases = [
        ("What is the price of the SmartWatch Pro X?", "products"),
        ("What is your return policy?", "returns"),
        ("How can I contact customer support?", "general"),
    ]
    
    for query, category in test_cases:
        print(f"\n📝 Query: {query}")
        print(f"   Category: {category}")
        print(f"\n   Generating response...")
        
        result = responder.respond(query, category)
        
        print(f"\n   ✓ Response: {result['response']}")
        print(f"   Retrieved {result['num_docs']} documents")


if __name__ == "__main__":
    test_rag_responder()
