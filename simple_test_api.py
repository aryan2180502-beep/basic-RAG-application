"""Simple test script to validate Google Gemini API key"""
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')
print(f"Testing API key: {api_key[:20]}...")

try:
    import google.genai as genai
    
    # Configure the client
    client = genai.Client(api_key=api_key)
    
    # Test with gemini-2.5-flash-lite
    print("\nTesting with model: gemini-2.5-flash-lite")
    
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents='Say "API key is valid!"'
    )
    
    print("✓ API Key is VALID!")
    print(f"✓ Model: gemini-2.5-flash-lite works!")
    print(f"Response: {response.text}")
    
except Exception as e:
    print(f"✗ API Key validation FAILED!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error: {str(e)}")
