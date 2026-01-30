#!/usr/bin/env python
"""Quick demo script for the customer support chatbot"""

from langgraph_workflow import CustomerSupportWorkflow

def demo():
    """Run a quick demo of the chatbot."""
    print("\n" + "="*70)
    print("🤖 TECHGEAR ELECTRONICS - CUSTOMER SUPPORT CHATBOT DEMO")
    print("="*70 + "\n")
    
    # Initialize
    workflow = CustomerSupportWorkflow()
    
    # Demo queries
    demo_queries = [
        "What gaming laptops do you have?",
        "How much is the SmartWatch Pro X?",
        "Can I return a product after 10 days?",
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{'─'*70}")
        print(f"Demo Query {i}: {query}")
        print('─'*70)
        
        result = workflow.process_query(query)
        
        print(f"\n✅ Response:")
        print(result['response'])
        print()

if __name__ == "__main__":
    demo()
