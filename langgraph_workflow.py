"""
LangGraph Workflow - Complete Multi-Agent Orchestration
Integrates Classifier, RAG Responder, and Escalation agents
"""

from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os

from classifier_agent import ClassifierAgent
from rag_responder import RAGResponder
from escalation_agent import EscalationAgent

load_dotenv()


# Define the state schema
class WorkflowState(TypedDict):
    """State that flows through the workflow."""
    # Input
    query: str
    
    # Classification results
    category: Literal["products", "returns", "general", "unhandled"]
    confidence: float
    reasoning: str
    
    # RAG results
    retrieved_docs: list[str]
    context: str
    
    # Output
    response: str
    node_executed: str
    requires_escalation: bool
    escalation_log: dict


class CustomerSupportWorkflow:
    """LangGraph workflow for customer support."""
    
    def __init__(self, api_key: str = None):
        """Initialize the workflow with all agents."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        print("Initializing Customer Support Workflow...")
        print("=" * 70)
        
        # Initialize agents
        print("Loading Classifier Agent...")
        self.classifier = ClassifierAgent(api_key=self.api_key)
        
        print("Loading RAG Responder Agent...")
        self.rag_responder = RAGResponder(api_key=self.api_key)
        
        print("Loading Escalation Agent...")
        self.escalation_agent = EscalationAgent()
        
        # Build the graph
        self.graph = self._build_graph()
        
        print("✓ Workflow initialized successfully!")
        print("=" * 70)
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create the graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("classifier", self._classifier_node)
        workflow.add_node("rag_responder", self._rag_responder_node)
        workflow.add_node("escalation", self._escalation_node)
        
        # Set entry point
        workflow.set_entry_point("classifier")
        
        # Add conditional routing from classifier
        workflow.add_conditional_edges(
            "classifier",
            self._route_query,
            {
                "rag": "rag_responder",
                "escalate": "escalation"
            }
        )
        
        # Add edges to END
        workflow.add_edge("rag_responder", END)
        workflow.add_edge("escalation", END)
        
        return workflow.compile()
    
    def _classifier_node(self, state: WorkflowState) -> WorkflowState:
        """Node 1: Classify the query."""
        print(f"\n🔍 CLASSIFIER NODE: Analyzing query...")
        
        result = self.classifier.classify(state["query"])
        
        print(f"   Category: {result['category']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Reasoning: {result['reasoning']}")
        
        return {
            **state,
            "category": result["category"],
            "confidence": result["confidence"],
            "reasoning": result["reasoning"],
            "node_executed": "classifier"
        }
    
    def _rag_responder_node(self, state: WorkflowState) -> WorkflowState:
        """Node 2: Generate RAG response."""
        print(f"\n💬 RAG RESPONDER NODE: Generating response...")
        
        result = self.rag_responder.respond(
            query=state["query"],
            category=state["category"]
        )
        
        print(f"   Retrieved {result['num_docs']} relevant documents")
        print(f"   Response generated successfully: {result['success']}")
        
        return {
            **state,
            "response": result["response"],
            "retrieved_docs": result.get("retrieved_docs", []),
            "node_executed": "rag_responder",
            "requires_escalation": False
        }
    
    def _escalation_node(self, state: WorkflowState) -> WorkflowState:
        """Node 3: Handle escalation."""
        print(f"\n🚨 ESCALATION NODE: Escalating query...")
        
        # Check if clarification is needed (low confidence on valid category)
        if state["confidence"] < 0.5 and state["category"] != "unhandled":
            result = self.escalation_agent.generate_clarification_request(state["query"])
        else:
            result = self.escalation_agent.escalate(
                query=state["query"],
                category=state["category"],
                confidence=state["confidence"],
                reasoning=state["reasoning"]
            )
        
        print(f"   Escalation handled")
        
        return {
            **state,
            "response": result["response"],
            "node_executed": "escalation",
            "requires_escalation": result.get("requires_escalation", True),
            "escalation_log": result.get("escalation_log", {})
        }
    
    def _route_query(self, state: WorkflowState) -> str:
        """
        Orchestrator: Route query based on classification.
        
        Returns:
            "rag" or "escalate"
        """
        category = state["category"]
        confidence = state["confidence"]
        
        print(f"\n🎯 ORCHESTRATOR: Routing decision...")
        
        # Route to escalation if unhandled or low confidence
        if category == "unhandled":
            print(f"   → Routing to ESCALATION (category: unhandled)")
            return "escalate"
        
        if confidence < 0.7:
            print(f"   → Routing to ESCALATION (low confidence: {confidence:.2f})")
            return "escalate"
        
        # Route to RAG for valid categories with good confidence
        print(f"   → Routing to RAG RESPONDER (category: {category}, confidence: {confidence:.2f})")
        return "rag"
    
    def process_query(self, query: str) -> dict:
        """
        Process a customer query through the workflow.
        
        Args:
            query: The customer query string
            
        Returns:
            Dictionary with response and metadata
        """
        print("\n" + "=" * 70)
        print(f"PROCESSING QUERY: {query}")
        print("=" * 70)
        
        # Initialize state
        initial_state = {
            "query": query,
            "category": "general",
            "confidence": 0.0,
            "reasoning": "",
            "retrieved_docs": [],
            "context": "",
            "response": "",
            "node_executed": "",
            "requires_escalation": False,
            "escalation_log": {}
        }
        
        # Run the workflow
        result = self.graph.invoke(initial_state)
        
        print("\n" + "=" * 70)
        print("WORKFLOW COMPLETE")
        print("=" * 70)
        print(f"Node Executed: {result['node_executed']}")
        print(f"Category: {result['category']}")
        print(f"Requires Escalation: {result['requires_escalation']}")
        
        return result


def test_workflow():
    """Test the complete workflow with various queries."""
    print("\n" + "=" * 70)
    print("TESTING CUSTOMER SUPPORT WORKFLOW")
    print("=" * 70)
    
    # Initialize workflow
    workflow = CustomerSupportWorkflow()
    
    # Test queries
    test_queries = [
        "What is the price of the SmartWatch Pro X?",
        "What is your return policy?",
        "How can I contact customer support?",
        "Can you help me with something illegal?",
        "The thing is broken",
        "Tell me about gaming laptops",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'#' * 70}")
        print(f"TEST CASE {i}/{len(test_queries)}")
        print(f"{'#' * 70}")
        
        result = workflow.process_query(query)
        
        print(f"\n📤 FINAL RESPONSE:")
        print(f"{result['response']}")
        print("\n" + "-" * 70)


def interactive_mode():
    """Run the workflow in interactive mode."""
    print("\n" + "=" * 70)
    print("CUSTOMER SUPPORT CHATBOT - Interactive Mode")
    print("=" * 70)
    print("Type your questions or 'quit' to exit\n")
    
    workflow = CustomerSupportWorkflow()
    
    while True:
        try:
            query = input("\n💬 You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using TechGear Support! Goodbye!")
                break
            
            result = workflow.process_query(query)
            
            print(f"\n🤖 Assistant: {result['response']}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        test_workflow()
