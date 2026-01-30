"""Test script to validate Google Gemini API key"""
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')
print(f"Testing API key: {api_key[:10]}...")

try:
    # Test with gemini-2.5-flash-lite
    print("\nTesting with gemini-2.5-flash-lite model...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=api_key
    )
    
    response = llm.invoke("Say 'API key is valid!'")
    print("✓ API Key is VALID!")
    print(f"✓ Model: gemini-2.5-flash-lite")
    print(f"Response: {response.content}")
    
except Exception as e:
    print(f"✗ API Key validation FAILED!")
    print(f"Error: {str(e)}")
