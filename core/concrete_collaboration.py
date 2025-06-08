"""
Concrete implementation of agent collaboration framework for WitsV3
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json

from .agent_collaboration import CollaborationProtocol, CollaborationMessage
from .schemas import StreamData


class MessageType(Enum):
    """Types of collaboration messages"""
    TASK_REQUEST = "task_request"
    TASK_OFFER = "task_offer"
    TASK_ACCEPT = "task_accept"
    TASK_REJECT = "task_reject"
    STATUS_UPDATE = "status_update"
    RESULT_SHARE = "result_share"
    HELP_REQUEST = "help_request"
    CONSENSUS_REQUEST = "consensus_request"
    CONSENSUS_VOTE = "consensus_vote"
    KNOWLEDGE_SHARE = "knowledge_share"


@dataclass
class AgentCapability:
    """Represents an agent's capability"""
    name: str
    proficiency: float  # 0.0 to 1.0
    experience_count: int = 0
    success_rate: float = 1.0
    
    def update_experience(self, success: bool):
        """Update capability based on experience"""
        self.experience_count += 1
        # Update success rate with weighted average
        weight = 0.1  # Weight for new experience
        self.success_rate = (1 - weight) * self.success_rate + weight * (1.0 if success else 0.0)
        
        # Increase proficiency with experience
        if success and self.proficiency < 1.0:
            self.proficiency = min(1.0, self.proficiency + 0.01)


@dataclass 
class CollaborationTask:
    """Represents a collaborative task"""
    task_id: str
    description: str
    required_capabilities: List[str]
    priority: float = 0.5
    deadline: Optional[datetime] = None
    assigned_agents: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, assigned, in_progress, completed, failed
    results: Optional[Any] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class AdvancedCollaborationProtocol(CollaborationProtocol):
    """
    Advanced collaboration protocol with negotiation, consensus, and learning
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{agent_id}")
        
        # Agent registry and capabilities
        self.known_agents: Dict[str, Dict[str, Any]] = {}
        self.capabilities: Dict[str, AgentCapability] = {}
        
        # Collaboration state
        self.active_collaborations: Dict[str, CollaborationTask] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.pending_negotiations: Dict[str, Dict[str, Any]] = {}
        
        # Consensus mechanisms
        self.consensus_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Knowledge sharing
        self.shared_knowledge: Dict[str, Any] = {}
        self.knowledge_subscribers: Dict[str, Set[str]] = {}  # topic -> agents
        
        # Performance tracking
        self.collaboration_history: List[Dict[str, Any]] = []
        
    async def register_capability(self, capability: str, proficiency: float = 0.5):
        """Register an agent capability"""
        if capability not in self.capabilities:
            self.capabilities[capability] = AgentCapability(
                name=capability,
                proficiency=proficiency
            )
            self.logger.info(f"Registered capability: {capability} (proficiency: {proficiency})")
    
    async def discover_agents(self) -> List[str]:
        """
        Discover other agents in the system
        """
        # In a real implementation, this would query a registry or use discovery protocol
        # For now, return known agents
        return list(self.known_agents.keys())
    
    async def send_message(
        self, 
        recipient: str, 
        message_type: str, 
        content: Any
    ) -> bool:
        """
        Send a message to another agent
        """
        message = CollaborationMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=content,
            timestamp=datetime.now().isoformat()
        )
        
        # In real implementation, this would use actual messaging
        # For now, we'll simulate by adding to recipient's queue
        self.logger.info(f"Sending {message_type} to {recipient}")
        
        # Store in history
        self.collaboration_history.append({
            "type": "message_sent",
            "message": message.__dict__,
            "timestamp": datetime.now().isoformat()
        })
        
        return True
    
    async def receive_messages(self) -> List[CollaborationMessage]:
        """
        Receive pending messages
        """
        messages = []
        
        # Get all messages from queue (non-blocking)
        while not self.message_queue.empty():
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=0.1
                )
                messages.append(message)
            except asyncio.TimeoutError:
                break
        
        return messages
    
    async def negotiate_task(
        self, 
        task_description: str, 
        required_capabilities: List[str]
    ) -> Optional[List[str]]:
        """
        Negotiate task assignment with other agents
        """
        self.logger.info(f"Negotiating task: {task_description}")
        
        # Create task
        task = CollaborationTask(
            task_id=f"task_{datetime.now().timestamp()}",
            description=task_description,
            required_capabilities=required_capabilities
        )
        
        # Find capable agents
        capable_agents = await self._find_capable_agents(required_capabilities)
        
        if not capable_agents:
            self.logger.warning("No capable agents found")
            return None
        
        # Send task offers
        negotiation_id = f"neg_{task.task_id}"
        self.pending_negotiations[negotiation_id] = {
            "task": task,
            "offers_sent": capable_agents,
            "responses": {},
            "deadline": datetime.now().timestamp() + 30  # 30 second timeout
        }
        
        for agent_id in capable_agents:
            await self.send_message(
                agent_id,
                MessageType.TASK_OFFER.value,
                {
                    "negotiation_id": negotiation_id,
                    "task": task.__dict__,
                    "expected_reward": self._calculate_reward(task)
                }
            )
        
        # Wait for responses
        assigned_agents = await self._wait_for_negotiation_responses(negotiation_id)
        
        if assigned_agents:
            task.assigned_agents = assigned_agents
            task.status = "assigned"
            self.active_collaborations[task.task_id] = task
            
            self.logger.info(f"Task assigned to: {assigned_agents}")
            return assigned_agents
        
        return None
    
    async def request_consensus(
        self, 
        topic: str, 
        options: List[Any], 
        participants: List[str]
    ) -> Optional[Any]:
        """
        Request consensus from a group of agents
        """
        session_id = f"consensus_{datetime.now().timestamp()}"
        
        self.consensus_sessions[session_id] = {
            "topic": topic,
            "options": options,
            "participants": participants,
            "votes": {},
            "deadline": datetime.now().timestamp() + 60  # 60 second timeout
        }
        
        # Send consensus requests
        for participant in participants:
            await self.send_message(
                participant,
                MessageType.CONSENSUS_REQUEST.value,
                {
                    "session_id": session_id,
                    "topic": topic,
                    "options": options,
                    "voting_method": "ranked_choice"
                }
            )
        
        # Wait for votes
        consensus_result = await self._wait_for_consensus(session_id)
        
        return consensus_result
    
    async def share_results(
        self, 
        task_id: str, 
        results: Any, 
        recipients: List[str]
    ):
        """
        Share task results with other agents
        """
        task = self.active_collaborations.get(task_id)
        if not task:
            self.logger.error(f"Task {task_id} not found")
            return
        
        # Update task
        task.results = results
        task.status = "completed"
        task.completed_at = datetime.now()
        
        # Share with recipients
        for recipient in recipients:
            await self.send_message(
                recipient,
                MessageType.RESULT_SHARE.value,
                {
                    "task_id": task_id,
                    "results": results,
                    "task_description": task.description
                }
            )
        
        # Update capabilities based on success
        for capability in task.required_capabilities:
            if capability in self.capabilities:
                self.capabilities[capability].update_experience(success=True)
    
    async def subscribe_to_knowledge(self, topic: str):
        """Subscribe to knowledge updates on a topic"""
        if topic not in self.knowledge_subscribers:
            self.knowledge_subscribers[topic] = set()
        
        self.knowledge_subscribers[topic].add(self.agent_id)
        self.logger.info(f"Subscribed to knowledge topic: {topic}")
    
    async def publish_knowledge(self, topic: str, knowledge: Any):
        """Publish knowledge for other agents"""
        # Store knowledge
        if topic not in self.shared_knowledge:
            self.shared_knowledge[topic] = []
        
        knowledge_entry = {
            "content": knowledge,
            "publisher": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
        
        self.shared_knowledge[topic].append(knowledge_entry)
        
        # Notify subscribers
        if topic in self.knowledge_subscribers:
            for subscriber in self.knowledge_subscribers[topic]:
                if subscriber != self.agent_id:
                    await self.send_message(
                        subscriber,
                        MessageType.KNOWLEDGE_SHARE.value,
                        {
                            "topic": topic,
                            "knowledge": knowledge_entry
                        }
                    )
    
    async def handle_incoming_message(self, message: CollaborationMessage):
        """Handle an incoming collaboration message"""
        self.logger.debug(f"Handling message: {message.message_type} from {message.sender}")
        
        if message.message_type == MessageType.TASK_OFFER.value:
            await self._handle_task_offer(message)
        elif message.message_type == MessageType.TASK_ACCEPT.value:
            await self._handle_task_accept(message)
        elif message.message_type == MessageType.CONSENSUS_REQUEST.value:
            await self._handle_consensus_request(message)
        elif message.message_type == MessageType.KNOWLEDGE_SHARE.value:
            await self._handle_knowledge_share(message)
        elif message.message_type == MessageType.HELP_REQUEST.value:
            await self._handle_help_request(message)
        
    # Helper methods
    
    async def _find_capable_agents(
        self, 
        required_capabilities: List[str]
    ) -> List[str]:
        """Find agents with required capabilities"""
        capable_agents = []
        
        for agent_id, agent_info in self.known_agents.items():
            agent_capabilities = agent_info.get("capabilities", [])
            
            # Check if agent has all required capabilities
            if all(cap in agent_capabilities for cap in required_capabilities):
                # Calculate match score
                match_score = sum(
                    agent_info.get("capability_scores", {}).get(cap, 0.5)
                    for cap in required_capabilities
                ) / len(required_capabilities)
                
                if match_score > 0.3:  # Minimum threshold
                    capable_agents.append((agent_id, match_score))
        
        # Sort by match score
        capable_agents.sort(key=lambda x: x[1], reverse=True)
        
        return [agent_id for agent_id, _ in capable_agents[:5]]  # Top 5
    
    def _calculate_reward(self, task: CollaborationTask) -> float:
        """Calculate reward for a task"""
        base_reward = 1.0
        
        # Adjust for priority
        priority_multiplier = 1 + task.priority
        
        # Adjust for complexity (based on required capabilities)
        complexity_multiplier = 1 + (len(task.required_capabilities) * 0.2)
        
        return base_reward * priority_multiplier * complexity_multiplier
    
    async def _wait_for_negotiation_responses(
        self, 
        negotiation_id: str
    ) -> List[str]:
        """Wait for negotiation responses and select agents"""
        negotiation = self.pending_negotiations.get(negotiation_id)
        if not negotiation:
            return []
        
        # Wait for responses with timeout
        start_time = datetime.now().timestamp()
        deadline = negotiation["deadline"]
        
        while datetime.now().timestamp() < deadline:
            # Check for responses in messages
            messages = await self.receive_messages()
            
            for message in messages:
                if (message.message_type == MessageType.TASK_ACCEPT.value and
                    message.content.get("negotiation_id") == negotiation_id):
                    negotiation["responses"][message.sender] = "accept"
                elif (message.message_type == MessageType.TASK_REJECT.value and
                      message.content.get("negotiation_id") == negotiation_id):
                    negotiation["responses"][message.sender] = "reject"
            
            # Check if we have enough acceptances
            acceptances = [
                agent_id for agent_id, response in negotiation["responses"].items()
                if response == "accept"
            ]
            
            task = negotiation["task"]
            if len(acceptances) >= len(task.required_capabilities):
                # We have enough agents
                return acceptances[:len(task.required_capabilities)]
            
            await asyncio.sleep(0.5)
        
        # Timeout - return any acceptances we got
        return [
            agent_id for agent_id, response in negotiation["responses"].items()
            if response == "accept"
        ]
    
    async def _wait_for_consensus(self, session_id: str) -> Optional[Any]:
        """Wait for consensus votes and calculate result"""
        session = self.consensus_sessions.get(session_id)
        if not session:
            return None
        
        deadline = session["deadline"]
        
        while datetime.now().timestamp() < deadline:
            # Check for votes in messages
            messages = await self.receive_messages()
            
            for message in messages:
                if (message.message_type == MessageType.CONSENSUS_VOTE.value and
                    message.content.get("session_id") == session_id):
                    session["votes"][message.sender] = message.content.get("vote")
            
            # Check if we have all votes
            if len(session["votes"]) >= len(session["participants"]):
                break
            
            await asyncio.sleep(0.5)
        
        # Calculate consensus
        return self._calculate_consensus(session["votes"], session["options"])
    
    def _calculate_consensus(
        self, 
        votes: Dict[str, Any], 
        options: List[Any]
    ) -> Optional[Any]:
        """Calculate consensus from votes using ranked choice"""
        if not votes:
            return None
        
        # Simple majority for now
        vote_counts = {}
        for vote in votes.values():
            if isinstance(vote, list) and vote:  # Ranked choice
                first_choice = vote[0]
                vote_counts[str(first_choice)] = vote_counts.get(str(first_choice), 0) + 1
            else:  # Single choice
                vote_counts[str(vote)] = vote_counts.get(str(vote), 0) + 1
        
        # Find option with most votes
        if vote_counts:
            winner = max(vote_counts, key=vote_counts.get)
            # Convert back to original type
            for option in options:
                if str(option) == winner:
                    return option
        
        return options[0] if options else None  # Default to first option
    
    async def _handle_task_offer(self, message: CollaborationMessage):
        """Handle incoming task offer"""
        offer = message.content
        task_data = offer.get("task", {})
        
        # Evaluate if we can handle the task
        can_handle = await self._evaluate_task_capability(
            task_data.get("required_capabilities", [])
        )
        
        response_type = MessageType.TASK_ACCEPT if can_handle else MessageType.TASK_REJECT
        
        await self.send_message(
            message.sender,
            response_type.value,
            {
                "negotiation_id": offer.get("negotiation_id"),
                "agent_capabilities": list(self.capabilities.keys()),
                "availability": self._check_availability()
            }
        )
    
    async def _handle_task_accept(self, message: CollaborationMessage):
        """Handle task acceptance"""
        # This would be processed by the negotiation waiting logic
        pass
    
    async def _handle_consensus_request(self, message: CollaborationMessage):
        """Handle consensus request"""
        request = message.content
        session_id = request.get("session_id")
        options = request.get("options", [])
        
        # Make decision (simplified - could use more complex logic)
        vote = await self._make_consensus_decision(
            request.get("topic"),
            options
        )
        
        await self.send_message(
            message.sender,
            MessageType.CONSENSUS_VOTE.value,
            {
                "session_id": session_id,
                "vote": vote
            }
        )
    
    async def _handle_knowledge_share(self, message: CollaborationMessage):
        """Handle incoming knowledge share"""
        knowledge_data = message.content
        topic = knowledge_data.get("topic")
        
        # Store in our knowledge base
        if topic not in self.shared_knowledge:
            self.shared_knowledge[topic] = []
        
        self.shared_knowledge[topic].append(knowledge_data.get("knowledge"))
        
        self.logger.info(f"Received knowledge on topic: {topic}")
    
    async def _handle_help_request(self, message: CollaborationMessage):
        """Handle help request from another agent"""
        request = message.content
        problem = request.get("problem")
        
        # Check if we can help
        if await self._can_help_with(problem):
            # Offer assistance
            await self.send_message(
                message.sender,
                MessageType.TASK_OFFER.value,
                {
                    "help_type": "assistance",
                    "capabilities": list(self.capabilities.keys()),
                    "estimated_time": 60  # seconds
                }
            )
    
    async def _evaluate_task_capability(
        self, 
        required_capabilities: List[str]
    ) -> bool:
        """Evaluate if we can handle required capabilities"""
        for required in required_capabilities:
            if required not in self.capabilities:
                return False
            
            # Check proficiency
            if self.capabilities[required].proficiency < 0.3:
                return False
        
        return True
    
    def _check_availability(self) -> float:
        """Check agent availability (0.0 to 1.0)"""
        active_tasks = sum(
            1 for task in self.active_collaborations.values()
            if task.status == "in_progress"
        )
        
        # Simple availability calculation
        max_concurrent_tasks = 5
        availability = max(0, 1 - (active_tasks / max_concurrent_tasks))
        
        return availability
    
    async def _make_consensus_decision(
        self, 
        topic: str, 
        options: List[Any]
    ) -> Any:
        """Make a decision for consensus voting"""
        # Simple heuristic - could be much more sophisticated
        
        # Check if we have relevant knowledge
        relevant_knowledge = self.shared_knowledge.get(topic, [])
        
        if relevant_knowledge:
            # Use knowledge to inform decision
            # For now, just pick first option mentioned in knowledge
            for knowledge in relevant_knowledge:
                for i, option in enumerate(options):
                    if str(option) in str(knowledge):
                        return option
        
        # Default to first option
        return options[0] if options else None
    
    async def _can_help_with(self, problem: str) -> bool:
        """Determine if we can help with a problem"""
        # Simple keyword matching
        problem_lower = problem.lower()
        
        for capability in self.capabilities:
            if capability.lower() in problem_lower:
                return True
        
        return False
    
    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get collaboration statistics"""
        completed_tasks = [
            task for task in self.active_collaborations.values()
            if task.status == "completed"
        ]
        
        return {
            "total_collaborations": len(self.active_collaborations),
            "completed_tasks": len(completed_tasks),
            "success_rate": len(completed_tasks) / max(1, len(self.active_collaborations)),
            "capabilities": {
                cap.name: {
                    "proficiency": cap.proficiency,
                    "experience": cap.experience_count,
                    "success_rate": cap.success_rate
                }
                for cap in self.capabilities.values()
            },
            "known_agents": len(self.known_agents),
            "shared_knowledge_topics": len(self.shared_knowledge)
        }


# Example usage and testing
async def test_advanced_collaboration():
    """Test the advanced collaboration protocol"""
    
    # Create agents
    agent1 = AdvancedCollaborationProtocol("agent_1")
    agent2 = AdvancedCollaborationProtocol("agent_2")
    agent3 = AdvancedCollaborationProtocol("agent_3")
    
    # Register capabilities
    await agent1.register_capability("data_analysis", 0.8)
    await agent1.register_capability("visualization", 0.6)
    
    await agent2.register_capability("data_analysis", 0.7)
    await agent2.register_capability("machine_learning", 0.9)
    
    await agent3.register_capability("visualization", 0.9)
    await agent3.register_capability("reporting", 0.8)
    
    # Simulate agent discovery
    agent1.known_agents = {
        "agent_2": {"capabilities": ["data_analysis", "machine_learning"]},
        "agent_3": {"capabilities": ["visualization", "reporting"]}
    }
    
    # Test task negotiation
    print("Testing task negotiation...")
    assigned = await agent1.negotiate_task(
        "Analyze sales data and create visualizations",
        ["data_analysis", "visualization"]
    )
    print(f"Task assigned to: {assigned}")
    
    # Test consensus
    print("\nTesting consensus mechanism...")
    consensus = await agent1.request_consensus(
        "Choose analysis method",
        ["regression", "clustering", "classification"],
        ["agent_2", "agent_3"]
    )
    print(f"Consensus reached: {consensus}")
    
    # Test knowledge sharing
    print("\nTesting knowledge sharing...")
    await agent2.subscribe_to_knowledge("ml_insights")
    await agent1.publish_knowledge(
        "ml_insights",
        {"finding": "Feature X strongly correlates with outcome", "confidence": 0.85}
    )
    
    # Get statistics
    print("\nCollaboration statistics:")
    stats = agent1.get_collaboration_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    asyncio.run(test_advanced_collaboration())