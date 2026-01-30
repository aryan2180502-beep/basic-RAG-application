"""
Escalation Agent - Node 3
Handles queries that cannot be processed automatically
"""

from datetime import datetime


class EscalationAgent:
    """Agent to handle queries that need human escalation."""
    
    def __init__(self):
        """Initialize the escalation agent."""
        self.support_email = "support@techgear.com"
        self.support_hours = "Mon-Sat, 9AM-6PM IST"
        self.response_time = "24 hours"
    
    def escalate(self, query: str, category: str = "unhandled", 
                 confidence: float = 0.0, reasoning: str = "") -> dict:
        """
        Generate an escalation message for unhandled queries.
        
        Args:
            query: The customer query
            category: The classified category
            confidence: Confidence score
            reasoning: Reason for escalation
            
        Returns:
            Dictionary with escalation response and metadata
        """
        
        # Determine escalation reason
        if category == "unhandled":
            escalation_reason = "This query requires human assistance"
        elif confidence < 0.7:
            escalation_reason = "The query needs clarification"
        else:
            escalation_reason = "This request needs specialized support"
        
        # Build escalation message
        response = f"""I apologize, but I need to connect you with a human support agent for this request.

**Your Query:** "{query}"

**Reason:** {escalation_reason}

Our support team is here to help you with:
• Complex technical issues
• Account-specific inquiries  
• Special requests and customizations
• Detailed product consultations

**Contact Information:**
📧 **Email:** {self.support_email}
⏰ **Hours:** {self.support_hours}
⏱️ **Response Time:** Within {self.response_time}

A support agent will review your request and respond as soon as possible.

Thank you for your patience!"""
        
        # Log escalation (in production, this would go to a ticketing system)
        escalation_log = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "category": category,
            "confidence": confidence,
            "reasoning": reasoning,
            "status": "escalated"
        }
        
        return {
            "response": response,
            "requires_escalation": True,
            "escalation_log": escalation_log,
            "contact_info": {
                "email": self.support_email,
                "hours": self.support_hours,
                "response_time": self.response_time
            }
        }
    
    def generate_clarification_request(self, query: str) -> dict:
        """
        Generate a message requesting clarification for vague queries.
        
        Args:
            query: The vague customer query
            
        Returns:
            Dictionary with clarification request
        """
        response = f"""Thank you for contacting TechGear Electronics!

I'd be happy to help, but I need a bit more information to provide you with the best answer.

**Your Query:** "{query}"

Could you please provide more details about:
• What product or service you're asking about?
• What specific information do you need?
• Is this related to a purchase, return, or general inquiry?

Alternatively, you can:
📧 **Email us:** {self.support_email} with detailed information
⏰ **Available:** {self.support_hours}

Thank you for your understanding!"""
        
        return {
            "response": response,
            "requires_escalation": False,
            "needs_clarification": True
        }


def test_escalation_agent():
    """Test the escalation agent with sample scenarios."""
    print("=" * 70)
    print("Testing Escalation Agent")
    print("=" * 70)
    
    agent = EscalationAgent()
    
    # Test Case 1: Unhandled query
    print("\n📝 Test Case 1: Unhandled/Inappropriate Query")
    print("   Query: 'Can you help me hack something?'")
    result = agent.escalate(
        query="Can you help me hack something?",
        category="unhandled",
        confidence=0.95,
        reasoning="Inappropriate request"
    )
    print(f"\n{result['response']}")
    
    # Test Case 2: Low confidence
    print("\n" + "=" * 70)
    print("\n📝 Test Case 2: Low Confidence Query")
    print("   Query: 'The thing doesn't work'")
    result = agent.escalate(
        query="The thing doesn't work",
        category="general",
        confidence=0.3,
        reasoning="Too vague to classify accurately"
    )
    print(f"\n{result['response']}")
    
    # Test Case 3: Clarification needed
    print("\n" + "=" * 70)
    print("\n📝 Test Case 3: Clarification Request")
    print("   Query: 'How much?'")
    result = agent.generate_clarification_request("How much?")
    print(f"\n{result['response']}")


if __name__ == "__main__":
    test_escalation_agent()
