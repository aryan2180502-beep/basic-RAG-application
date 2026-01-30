"""
Test script for the FastAPI chatbot server
"""

import requests
import json
from typing import Dict

# API endpoint
BASE_URL = "http://localhost:8000"


def print_response(response: requests.Response, title: str):
    """Pretty print API response."""
    print("\n" + "=" * 70)
    print(f"TEST: {title}")
    print("=" * 70)
    
    if response.status_code == 200:
        data = response.json()
        
        # Print natural language response
        print(f"\n🤖 RESPONSE:")
        print(f"{data.get('response', 'N/A')}")
        
        # Print JSON metadata
        print(f"\n📊 METADATA (JSON):")
        print(f"  Category: {data.get('category', 'N/A')}")
        print(f"  Confidence: {data.get('confidence', 0):.2f}")
        print(f"  Reasoning: {data.get('reasoning', 'N/A')}")
        print(f"  Node Executed: {data.get('node_executed', 'N/A')}")
        print(f"  Requires Escalation: {data.get('requires_escalation', False)}")
        print(f"  Timestamp: {data.get('timestamp', 'N/A')}")
        
        if data.get('retrieved_docs'):
            print(f"  Retrieved Docs: {len(data['retrieved_docs'])} documents")
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(f"   {response.text}")


def test_health():
    """Test health endpoint."""
    print("\n" + "=" * 70)
    print("TESTING HEALTH ENDPOINT")
    print("=" * 70)
    
    response = requests.get(f"{BASE_URL}/health")
    data = response.json()
    
    print(f"\nStatus: {data['status']}")
    print(f"Message: {data['message']}")
    print(f"Timestamp: {data['timestamp']}")


def test_categories():
    """Test categories endpoint."""
    print("\n" + "=" * 70)
    print("AVAILABLE CATEGORIES")
    print("=" * 70)
    
    response = requests.get(f"{BASE_URL}/api/categories")
    data = response.json()
    
    for category in data['categories']:
        print(f"\n📂 {category['name'].upper()}")
        print(f"   Description: {category['description']}")
        print(f"   Examples:")
        for example in category['examples']:
            print(f"     - {example}")


def test_chat(query: str, session_id: str = None):
    """Test chat endpoint."""
    payload = {"query": query}
    if session_id:
        payload["session_id"] = session_id
    
    response = requests.post(
        f"{BASE_URL}/api/chat",
        json=payload
    )
    
    return response


def test_simple_chat(query: str):
    """Test simple chat endpoint."""
    response = requests.post(
        f"{BASE_URL}/api/chat/simple",
        json={"query": query}
    )
    
    print("\n" + "=" * 70)
    print(f"SIMPLE ENDPOINT TEST: {query}")
    print("=" * 70)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n🤖 Response: {data['response']}")
    else:
        print(f"\n❌ Error: {response.status_code}")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "#" * 70)
    print("FASTAPI CHATBOT SERVER - API TESTS")
    print("#" * 70)
    
    # Test 1: Health check
    test_health()
    
    # Test 2: Categories
    test_categories()
    
    # Test 3: Product query
    response = test_chat("What is the price of the SmartWatch Pro X?", "test_session_1")
    print_response(response, "Product Query")
    
    # Test 4: Returns query
    response = test_chat("What is your return policy?")
    print_response(response, "Returns Query")
    
    # Test 5: General query
    response = test_chat("How can I contact customer support?")
    print_response(response, "General Query")
    
    # Test 6: Gaming laptop query
    response = test_chat("Tell me about gaming laptops")
    print_response(response, "Gaming Laptop Query")
    
    # Test 7: Unhandled query
    response = test_chat("Can you help me with something illegal?")
    print_response(response, "Unhandled/Inappropriate Query")
    
    # Test 8: Vague query (escalation)
    response = test_chat("The thing is broken")
    print_response(response, "Vague Query (Escalation)")
    
    # Test 9: Simple endpoint
    test_simple_chat("What are the features of wireless earbuds?")
    
    print("\n" + "#" * 70)
    print("ALL TESTS COMPLETED")
    print("#" * 70 + "\n")


def interactive_test():
    """Interactive testing mode."""
    print("\n" + "=" * 70)
    print("INTERACTIVE API TEST MODE")
    print("=" * 70)
    print("Enter queries to test the API (type 'quit' to exit)\n")
    
    session_id = "interactive_session"
    
    while True:
        try:
            query = input("\n💬 Your query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            response = test_chat(query, session_id)
            
            if response.status_code == 200:
                data = response.json()
                print(f"\n🤖 Assistant: {data['response']}")
                print(f"\n📊 [Category: {data['category']} | Confidence: {data['confidence']:.2f}]")
            else:
                print(f"\n❌ Error: {response.status_code}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_test()
    else:
        run_all_tests()
