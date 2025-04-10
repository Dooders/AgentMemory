"""
Conversation Agent Memory Simulation.
Tests memory system performance in a realistic chat scenario.
"""

import random
import time
from typing import Dict, List, Optional

from memory import AgentMemorySystem, MemoryConfig
from memory.config import RedisSTMConfig, RedisIMConfig, SQLiteLTMConfig

# Simulated conversation topics and responses
TOPICS = [
    "product_info",
    "technical_support",
    "billing",
    "general_inquiry",
    "complaint"
]

CUSTOMER_SENTIMENTS = ["positive", "neutral", "negative"]

class ConversationAgent:
    def __init__(self, agent_id: str, memory_system: AgentMemorySystem):
        self.agent_id = agent_id
        self.memory_system = memory_system
        self.conversation_count = 0
        self.current_conversation: Optional[Dict] = None
    
    def start_conversation(self, customer_id: str, topic: str) -> None:
        """Start a new conversation with a customer."""
        self.conversation_count += 1
        self.current_conversation = {
            "customer_id": customer_id,
            "topic": topic,
            "start_time": time.time(),
            "messages": [],
            "sentiment": "neutral"
        }
        
        # Store conversation start in memory
        self.memory_system.store_agent_state(
            self.agent_id,
            {
                "type": "conversation_start",
                "customer_id": customer_id,
                "topic": topic,
                "conversation_id": f"conv_{self.conversation_count}"
            },
            self.conversation_count,
            0.7
        )
    
    def process_message(self, message: str, sentiment: str) -> str:
        """Process an incoming message and generate a response."""
        if not self.current_conversation:
            return "No active conversation"
        
        # Update conversation state
        self.current_conversation["messages"].append({
            "content": message,
            "sentiment": sentiment,
            "timestamp": time.time()
        })
        
        # Update overall sentiment
        if sentiment == "negative":
            self.current_conversation["sentiment"] = "negative"
        elif sentiment == "positive" and self.current_conversation["sentiment"] != "negative":
            self.current_conversation["sentiment"] = "positive"
        
        # Store message in memory
        self.memory_system.store_agent_interaction(
            self.agent_id,
            {
                "type": "customer_message",
                "content": message,
                "sentiment": sentiment,
                "customer_id": self.current_conversation["customer_id"],
                "conversation_id": f"conv_{self.conversation_count}"
            },
            self.conversation_count,
            0.6 if sentiment == "neutral" else 0.8
        )
        
        # Generate response based on similar past interactions
        response = self._generate_response(message, sentiment)
        return response
    
    def _generate_response(self, message: str, sentiment: str) -> str:
        """Generate a response based on memory of similar interactions."""
        # Search for similar past interactions
        similar_interactions = self.memory_system.retrieve_similar_states(
            self.agent_id,
            {
                "type": "customer_message",
                "sentiment": sentiment,
                "topic": self.current_conversation["topic"]
            },
            k=3
        )
        
        # Simulate response generation (in real world, this would use the similar interactions)
        response = f"Response to: {message} (based on {len(similar_interactions)} similar interactions)"
        
        # Store response in memory
        self.memory_system.store_agent_action(
            self.agent_id,
            {
                "type": "agent_response",
                "content": response,
                "based_on": [m.get("id") for m in similar_interactions],
                "conversation_id": f"conv_{self.conversation_count}"
            },
            self.conversation_count,
            0.6
        )
        
        return response
    
    def end_conversation(self) -> Dict:
        """End the current conversation and store summary."""
        if not self.current_conversation:
            return {}
        
        # Calculate conversation metrics
        duration = time.time() - self.current_conversation["start_time"]
        message_count = len(self.current_conversation["messages"])
        final_sentiment = self.current_conversation["sentiment"]
        
        # Store conversation summary
        summary = {
            "type": "conversation_summary",
            "customer_id": self.current_conversation["customer_id"],
            "topic": self.current_conversation["topic"],
            "duration": duration,
            "message_count": message_count,
            "final_sentiment": final_sentiment,
            "conversation_id": f"conv_{self.conversation_count}"
        }
        
        # Store with high priority if negative sentiment
        priority = 0.9 if final_sentiment == "negative" else 0.7
        self.memory_system.store_agent_state(
            self.agent_id,
            summary,
            self.conversation_count,
            priority
        )
        
        result = self.current_conversation
        self.current_conversation = None
        return result

def run_conversation_simulation(
    num_conversations: int,
    messages_per_conversation: int
) -> Dict:
    """Run a simulation of customer service conversations."""
    # Initialize memory system
    config = MemoryConfig(
        stm_config=RedisSTMConfig(
            host="localhost",
            port=6379,
            ttl=3600,
            memory_limit=10000
        ),
        im_config=RedisIMConfig(
            ttl=7200,
            memory_limit=20000
        ),
        ltm_config=SQLiteLTMConfig(
            db_path="conversation_ltm.db"
        ),
        cleanup_interval=100,
        enable_compression=True
    )
    
    memory_system = AgentMemorySystem.get_instance(config)
    agent = ConversationAgent("customer_service_1", memory_system)
    
    metrics = {
        "total_conversations": 0,
        "total_messages": 0,
        "sentiment_distribution": {s: 0 for s in CUSTOMER_SENTIMENTS},
        "avg_response_time": 0,
        "total_response_time": 0
    }
    
    print(f"Starting simulation with {num_conversations} conversations")
    
    for conv_idx in range(num_conversations):
        # Start new conversation
        customer_id = f"customer_{conv_idx}"
        topic = random.choice(TOPICS)
        agent.start_conversation(customer_id, topic)
        
        # Simulate message exchange
        for msg_idx in range(messages_per_conversation):
            # Generate random customer message
            message = f"Customer message {msg_idx} about {topic}"
            sentiment = random.choice(CUSTOMER_SENTIMENTS)
            
            # Measure response time
            start_time = time.time()
            response = agent.process_message(message, sentiment)
            response_time = time.time() - start_time
            
            # Update metrics
            metrics["total_messages"] += 1
            metrics["sentiment_distribution"][sentiment] += 1
            metrics["total_response_time"] += response_time
            
            # Small delay between messages
            time.sleep(0.1)
        
        # End conversation
        agent.end_conversation()
        metrics["total_conversations"] += 1
        
        if conv_idx % 10 == 0:
            print(f"Completed {conv_idx}/{num_conversations} conversations")
    
    # Calculate final metrics
    metrics["avg_response_time"] = (
        metrics["total_response_time"] / metrics["total_messages"]
        if metrics["total_messages"] > 0 else 0
    )
    
    return metrics

def main():
    # Run simulation
    try:
        metrics = run_conversation_simulation(
            num_conversations=50,
            messages_per_conversation=5
        )
        
        # Print results
        print("\nSimulation Results:")
        print(f"Total Conversations: {metrics['total_conversations']}")
        print(f"Total Messages: {metrics['total_messages']}")
        print(f"Average Response Time: {metrics['avg_response_time']:.3f} seconds")
        print("\nSentiment Distribution:")
        for sentiment, count in metrics['sentiment_distribution'].items():
            percentage = (count / metrics['total_messages']) * 100
            print(f"  {sentiment}: {percentage:.1f}%")
            
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 