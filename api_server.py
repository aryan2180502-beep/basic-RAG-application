"""
FastAPI Server for TechGear Electronics Customer Support Chatbot
Provides REST API endpoint with Swagger documentation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uvicorn
from dotenv import load_dotenv

from langgraph_workflow import CustomerSupportWorkflow

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="TechGear Electronics Customer Support API",
    description="""
    ## Customer Support Chatbot API
    
    This API provides an intelligent customer support chatbot powered by:
    - **LangGraph** for multi-agent orchestration
    - **Google Gemini** for natural language understanding
    - **ChromaDB** for RAG-based responses
    
    ### Features:
    - Automatic query classification (products, returns, general, unhandled)
    - Context-aware responses using RAG
    - Intelligent escalation for complex queries
    - Confidence scoring and routing
    
    ### Workflow:
    1. **Classifier Agent** - Categorizes the query
    2. **Orchestrator** - Routes to appropriate handler
    3. **RAG Responder** - Generates response using knowledge base
    4. **Escalation Agent** - Handles unprocessable queries
    """,
    version="1.0.0",
    contact={
        "name": "TechGear Electronics Support",
        "email": "support@techgear.com"
    },
    license_info={
        "name": "MIT License"
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the workflow
print("Initializing Customer Support Workflow...")
workflow = CustomerSupportWorkflow()
print("✓ Workflow ready!")


# Request/Response Models
class ChatRequest(BaseModel):
    """Request model for chatbot query."""
    query: str = Field(
        ...,
        description="Customer query or question",
        example="What is the price of the SmartWatch Pro X?"
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional session ID for conversation tracking"
    )
    
    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "query": "What is the price of the SmartWatch Pro X?",
                    "session_id": "user_123"
                },
                {
                    "query": "What is your return policy?"
                },
                {
                    "query": "Tell me about gaming laptops"
                }
            ]
        }


class ChatResponse(BaseModel):
    """Response model for chatbot."""
    
    # Natural language response
    response: str = Field(
        ...,
        description="Natural language response from the chatbot"
    )
    
    # Metadata
    category: str = Field(
        ...,
        description="Query category (products, returns, general, unhandled)"
    )
    confidence: float = Field(
        ...,
        description="Classification confidence score (0.0 to 1.0)"
    )
    reasoning: str = Field(
        ...,
        description="Reasoning for the classification"
    )
    node_executed: str = Field(
        ...,
        description="Which agent handled the query (classifier, rag_responder, escalation)"
    )
    requires_escalation: bool = Field(
        ...,
        description="Whether the query was escalated to human support"
    )
    
    # Additional info
    retrieved_docs: Optional[List[str]] = Field(
        None,
        description="Retrieved document snippets (for RAG responses)"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Response timestamp"
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID if provided in request"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "The SmartWatch Pro X is priced at ₹15,999.",
                "category": "products",
                "confidence": 0.95,
                "reasoning": "Query is asking about product price",
                "node_executed": "rag_responder",
                "requires_escalation": False,
                "retrieved_docs": ["Product: SmartWatch Pro X..."],
                "timestamp": "2026-01-30T10:30:00",
                "session_id": "user_123"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str
    timestamp: str


# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API information.
    """
    return {
        "api": "TechGear Electronics Customer Support Chatbot",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "chat": "/api/chat",
            "health": "/health"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the operational status of the API.
    """
    return {
        "status": "healthy",
        "message": "Customer Support Chatbot is operational",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Process a customer support query.
    
    ## Description
    This endpoint processes customer queries through a multi-agent workflow:
    
    1. **Classification** - Query is categorized and confidence is calculated
    2. **Routing** - Query is routed to RAG or Escalation based on category and confidence
    3. **Response Generation** - Appropriate agent generates the response
    
    ## Request Body
    - **query** (required): The customer's question or request
    - **session_id** (optional): Session identifier for conversation tracking
    
    ## Response
    Returns both:
    - **Natural language response** - Human-readable answer
    - **JSON metadata** - Classification, confidence, routing info, etc.
    
    ## Examples
    
    ### Product Query
    ```json
    {
        "query": "What is the price of the SmartWatch Pro X?"
    }
    ```
    
    ### Returns Query
    ```json
    {
        "query": "What is your return policy?"
    }
    ```
    
    ### General Query
    ```json
    {
        "query": "How can I contact support?"
    }
    ```
    """
    try:
        # Process the query through the workflow
        result = workflow.process_query(request.query)
        
        # Build response
        response = ChatResponse(
            response=result["response"],
            category=result["category"],
            confidence=result["confidence"],
            reasoning=result["reasoning"],
            node_executed=result["node_executed"],
            requires_escalation=result["requires_escalation"],
            retrieved_docs=result.get("retrieved_docs", None),
            session_id=request.session_id
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/api/categories", tags=["Information"])
async def get_categories():
    """
    Get available query categories.
    
    Returns information about the query categories the chatbot can handle.
    """
    return {
        "categories": [
            {
                "name": "products",
                "description": "Questions about product specifications, prices, features, and availability",
                "examples": [
                    "What is the price of SmartWatch Pro X?",
                    "Tell me about gaming laptops",
                    "What features does the wireless earbuds have?"
                ]
            },
            {
                "name": "returns",
                "description": "Questions about return policy, refunds, and exchanges",
                "examples": [
                    "What is your return policy?",
                    "How do I return a product?",
                    "Can I get a refund?"
                ]
            },
            {
                "name": "general",
                "description": "General inquiries about warranty, support, shipping, and company info",
                "examples": [
                    "What are your customer support hours?",
                    "How long is the warranty?",
                    "How can I contact support?"
                ]
            },
            {
                "name": "unhandled",
                "description": "Queries that are inappropriate, unclear, or outside the scope",
                "examples": [
                    "Too vague or unclear queries",
                    "Inappropriate or offensive content",
                    "Requests outside TechGear Electronics scope"
                ],
                "action": "Routes to escalation agent"
            }
        ]
    }


@app.post("/api/chat/simple", tags=["Chat"])
async def chat_simple(request: ChatRequest):
    """
    Simplified chat endpoint - returns only the response text.
    
    Use this endpoint if you only need the natural language response
    without metadata.
    
    ## Response
    Returns a simple JSON with just the response text:
    ```json
    {
        "response": "The SmartWatch Pro X is priced at ₹15,999."
    }
    ```
    """
    try:
        result = workflow.process_query(request.query)
        return {"response": result["response"]}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


# Run the server
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
