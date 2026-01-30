# TechGear Electronics Customer Support API Documentation

## Overview
REST API for the TechGear Electronics Customer Support Chatbot powered by RAG (Retrieval Augmented Generation), ChromaDB vector database, LangChain, and LangGraph multi-agent workflow.

## Base URL
```
http://localhost:8000
```

## Interactive API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Endpoints

### 1. Health Check
**GET** `/health`

Check if the API server is operational.

**Response**:
```json
{
  "status": "healthy",
  "message": "Customer Support Chatbot is operational",
  "timestamp": "2026-01-30T09:42:41.794520"
}
```

---

### 2. Get Categories
**GET** `/api/categories`

Get information about all supported query categories.

**Response**:
```json
{
  "categories": [
    {
      "name": "products",
      "description": "Questions about product specifications, prices, features, and availability",
      "examples": [
        "What is the price of SmartWatch Pro X?",
        "Tell me about gaming laptops"
      ]
    },
    {
      "name": "returns",
      "description": "Questions about return policy, refunds, and exchanges",
      "examples": ["What is your return policy?"]
    },
    {
      "name": "general",
      "description": "General inquiries about warranty, support, shipping",
      "examples": ["What are your customer support hours?"]
    },
    {
      "name": "unhandled",
      "description": "Queries that are inappropriate, unclear, or outside scope",
      "action": "Routes to escalation agent"
    }
  ]
}
```

---

### 3. Chat Endpoint (Full Response)
**POST** `/api/chat`

Send a query and get a detailed response with full metadata.

**Request Body**:
```json
{
  "query": "What is the price of the SmartWatch Pro X?",
  "session_id": "optional-session-id"
}
```

**Response**:
```json
{
  "response": "The SmartWatch Pro X is priced at ₹15,999.",
  "category": "products",
  "confidence": 0.95,
  "reasoning": "The query is asking about the price of a specific product.",
  "node_executed": "rag_responder",
  "requires_escalation": false,
  "retrieved_docs": [
    "Product: SmartWatch Pro X\nPrice: ₹15,999...",
    "..."
  ],
  "timestamp": "2026-01-30T09:42:15.713505",
  "session_id": "optional-session-id"
}
```

**Fields**:
- `response` (string): Natural language answer from the chatbot
- `category` (string): Query category (products/returns/general/unhandled)
- `confidence` (float): Classification confidence score (0.0 - 1.0)
- `reasoning` (string): Why this category was chosen
- `node_executed` (string): Which agent processed the query (rag_responder/escalation)
- `requires_escalation` (boolean): Whether human intervention is needed
- `retrieved_docs` (array): Relevant documents retrieved from ChromaDB
- `timestamp` (string): ISO 8601 formatted timestamp
- `session_id` (string|null): Optional session identifier

---

### 4. Simple Chat Endpoint
**POST** `/api/chat/simple`

Send a query and get only the text response (no metadata).

**Request Body**:
```json
{
  "query": "Tell me about gaming laptops",
  "session_id": "optional-session-id"
}
```

**Response**:
```json
{
  "response": "TechGear Electronics offers the Gaming Laptop Beast 15 for ₹89,999..."
}
```

---

## Query Categories & Routing

### Category Classification
The chatbot uses an intelligent classifier agent to categorize queries:

1. **Products** (confidence ≥ 0.7)
   - Product specifications, prices, features, availability
   - Routes to: RAG Responder Agent
   - Example: "What is the price of SmartWatch Pro X?"

2. **Returns** (confidence ≥ 0.7)
   - Return policies, refunds, exchanges
   - Routes to: RAG Responder Agent
   - Example: "What is your return policy?"

3. **General** (confidence ≥ 0.7)
   - Warranty, support, shipping, company information
   - Routes to: RAG Responder Agent
   - Example: "What are your customer support hours?"

4. **Unhandled** (confidence < 0.7 OR inappropriate)
   - Vague, unclear, inappropriate, or out-of-scope queries
   - Routes to: Escalation Agent
   - Example: "The thing is broken" (too vague)

### Multi-Agent Workflow

```
User Query
    ↓
┌─────────────────────┐
│ Classifier Agent    │
│ (Gemini 2.5 Flash)  │
└─────────────────────┘
    ↓
[Category + Confidence]
    ↓
    ├─→ confidence ≥ 0.7 AND category != "unhandled"
    │       ↓
    │   ┌─────────────────────┐
    │   │ RAG Responder Agent │
    │   │ (ChromaDB + Gemini) │
    │   └─────────────────────┘
    │       ↓
    │   Natural Language Answer
    │
    └─→ confidence < 0.7 OR category == "unhandled"
            ↓
        ┌────────────────────┐
        │ Escalation Agent   │
        │ (Professional Msg) │
        └────────────────────┘
            ↓
        Escalation Message
```

---

## Example Use Cases

### 1. Product Information Query
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What features does the wireless earbuds have?"
  }'
```

**Response**:
- Category: `products`
- Confidence: `0.95`
- Node: `rag_responder`
- Retrieves relevant product information from ChromaDB
- Generates natural language response using Gemini

### 2. Return Policy Query
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How can I return my laptop?"
  }'
```

**Response**:
- Category: `returns`
- Confidence: `1.0`
- Node: `rag_responder`
- Retrieves return policy information
- Provides specific return windows and procedures

### 3. Escalation (Vague Query)
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "It doesnt work"
  }'
```

**Response**:
- Category: `unhandled`
- Confidence: `0.9`
- Node: `escalation`
- `requires_escalation`: `true`
- Professional escalation message with contact information

---

## Error Handling

### HTTP Status Codes
- `200 OK`: Successful request
- `422 Unprocessable Entity`: Invalid request body
- `500 Internal Server Error`: Server error during processing

### Error Response Format
```json
{
  "detail": "Error message describing what went wrong"
}
```

---

## Technical Stack

### Core Technologies
- **FastAPI** 0.128.0: Modern Python web framework
- **Uvicorn** 0.40.0: ASGI server
- **LangChain** 1.2.7: LLM application framework
- **LangGraph** 1.0.7: Multi-agent orchestration
- **ChromaDB** 1.4.1: Vector database for embeddings
- **Google Gemini** (gemini-2.5-flash-lite): LLM for responses

### Embeddings & Retrieval
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: ChromaDB (./chroma_db)
- **Collection**: techgear_products
- **Chunks**: 22 product information chunks
- **Retrieval**: Top-4 similar documents per query

### Agents
1. **Classifier Agent**
   - Model: Google Gemini 2.5 Flash Lite
   - Input: User query
   - Output: Category, confidence, reasoning (Pydantic structured output)

2. **RAG Responder Agent**
   - Retrieval: ChromaDB vector similarity search
   - Generation: Google Gemini with category-specific prompts
   - Context: Up to 4 relevant product chunks

3. **Escalation Agent**
   - Handles: Low-confidence and inappropriate queries
   - Provides: Professional escalation messages
   - Contact: support@techgear.com, Mon-Sat 9AM-6PM IST

---

## Performance Notes

### Initial Startup Time
- **First Load**: 30-60 seconds
  - Loading embeddings model
  - Initializing ChromaDB connection
  - Loading LLM components

### Response Times (After Initialization)
- Health check: < 10ms
- Categories endpoint: < 10ms
- Chat queries: 1-3 seconds
  - Classification: ~500ms
  - RAG retrieval: ~200ms
  - LLM generation: 1-2 seconds

### Resource Requirements
- **Memory**: ~2-3 GB (embeddings + models)
- **Disk**: ~500 MB (models + ChromaDB)
- **CPU**: Moderate (faster with GPU for embeddings)

---

## Development & Testing

### Starting the Server
```bash
# Standard mode
python api_server.py

# Background mode with logging
python api_server.py > server.log 2>&1 &

# Using uvicorn directly
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### Running Tests
```bash
# Run all automated tests
python test_api.py

# Interactive testing mode
python test_api.py interactive
```

### Accessing Swagger UI
Open your browser to: `http://localhost:8000/docs`
- Interactive API documentation
- Test endpoints directly
- View request/response schemas

---

## Contact & Support

For issues, questions, or feature requests:
- **Email**: support@techgear.com
- **Hours**: Monday to Saturday, 9 AM to 6 PM IST
- **Response Time**: Within 24 hours

---

## Version Information
- **API Version**: 1.0.0
- **Last Updated**: January 30, 2026
- **Status**: Production Ready ✅

---

## License
© 2026 TechGear Electronics. All rights reserved.
