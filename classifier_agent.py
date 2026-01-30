"""
Classifier Agent - Node 1
Categorizes customer queries into: products, returns, general, or unhandled
"""

from typing import Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()


class QueryClassification(BaseModel):
    """Classification result for a customer query."""
    category: Literal["products", "returns", "general", "unhandled"] = Field(
        description="The category of the query"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation for the classification"
    )


class ClassifierAgent:
    """Agent to classify customer queries."""
    
    def __init__(self, api_key: str = None):
        """Initialize the classifier agent."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        # Initialize LLM with structured output
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=self.api_key,
            temperature=0.1  # Low temperature for consistent classification
        )
        
        # Create structured output LLM
        self.structured_llm = self.llm.with_structured_output(QueryClassification)
        
        # Define classification prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a customer support query classifier for TechGear Electronics.

Your job is to categorize customer queries into one of these categories:

1. **products** - Questions about:
   - Product specifications, features, prices
   - Product comparisons
   - Product availability
   - What products are offered

2. **returns** - Questions about:
   - Return policy
   - Refund process
   - Exchange procedures
   - How to return items

3. **general** - Questions about:
   - Warranty information
   - Customer support contact
   - Shipping information
   - Store hours/locations
   - General company information

4. **unhandled** - Queries that are:
   - Inappropriate or offensive
   - Completely unrelated to TechGear Electronics
   - Too vague or unclear
   - Requests for illegal activities
   - Personal complaints or rants

Provide a confidence score (0.0 to 1.0) and brief reasoning for your classification."""),
            ("human", "Classify this customer query: {query}")
        ])
        
        # Create the chain
        self.chain = self.prompt | self.structured_llm
    
    def classify(self, query: str) -> dict:
        """
        Classify a customer query.
        
        Args:
            query: The customer query string
            
        Returns:
            Dictionary with category, confidence, and reasoning
        """
        try:
            result = self.chain.invoke({"query": query})
            
            return {
                "category": result.category,
                "confidence": result.confidence,
                "reasoning": result.reasoning
            }
        except Exception as e:
            # Fallback to unhandled on error
            return {
                "category": "unhandled",
                "confidence": 0.0,
                "reasoning": f"Classification error: {str(e)}"
            }


def test_classifier():
    """Test the classifier with sample queries."""
    print("=" * 70)
    print("Testing Classifier Agent")
    print("=" * 70)
    
    classifier = ClassifierAgent()
    
    test_queries = [
        "What is the price of the SmartWatch Pro X?",
        "How do I return a product?",
        "What are your customer support hours?",
        "Can you help me hack something?",
        "Tell me about your gaming laptops",
        "What's your refund policy?",
        "How long is the warranty on earbuds?",
        "The weather is nice today",
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        result = classifier.classify(query)
        print(f"   Category: {result['category']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Reasoning: {result['reasoning']}")


if __name__ == "__main__":
    test_classifier()
