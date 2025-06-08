"""
Agent Collaboration Framework for WitsV3

This module implements protocols for multi-agent collaboration, including
task negotiation, context sharing, and consensus decision-making.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Set, AsyncGenerator
from uuid import uuid4

from .schemas import StreamData
from .config import WitsV3Config

logger = logging.getLogger(__name__)


class CollaborationType(Enum):
    """Types of agent collaboration"""
    PEER_TO_PEER = "peer_to_peer"        # Equal agents collaborating
    HIERARCHICAL = "hierarchical"         # Leader-follower structure
    SWARM = "swarm"                      # Emergent collective behavior
    DELEGATION = "delegation"            # Task hand-off
    CONSENSUS = "consensus"              # Group decision making


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class CollaborationMessage:
    """Message exchanged between collaborating agents"""
    message_id: str = field(default_factory=lambda: str(uuid4()))
    sender_id: str = ""
    recipient_ids: List[str] = field(default_factory=list)  # Empty = broadcast
    message_type: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    requires_response: bool = False
    correlation_id: Optional[str] = None  # For request-response correlation
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SharedContext:
    """Shared context between collaborating agents"""
    context_id: str = field(default_factory=lambda: str(uuid4()))
    owner_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    access_control: Dict[str, Set[str]] = field(default_factory=dict)  # agent_id -> permissions
    version: int = 1
    last_modified: datetime = field(default_factory=datetime.now)
    
    def update(self, agent_id: str, updates: Dict[str, Any]) -> bool:
        """Update context if agent has permission"""
        if agent_id not in self.access_control or "write" not in self.access_control[agent_id]:
            return False
        
        self.data.update(updates)
        self.version += 1
        self.last_modified = datetime.now()
        return True
    
    def read(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Read context if agent has permission"""
        if agent_id not in self.access_control or "read" not in self.access_control[agent_id]:
            return None
        return self.data.copy()


@dataclass
class Task:
    """Task that can be distributed among agents"""
    task_id: str = field(default_factory=lambda: str(uuid4()))
    description: str = ""
    required_capabilities: Set[str] = field(default_factory=set)
    priority: int = 5  # 1-10, 1 is highest
    estimated_effort: float = 1.0  # Relative effort units
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)  # Other task IDs
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskAssignment:
    """Assignment of tasks to agents"""
    assignment_id: str = field(default_factory=lambda: str(uuid4()))
    task_id: str = ""
    agent_id: str = ""
    accepted: bool = False
    completion_estimate: Optional[float] = None
    confidence: float = 0.0
    assigned_at: datetime = field(default_factory=datetime.now)


@dataclass
class Decision:
    """Result of a consensus decision"""
    decision_id: str = field(default_factory=lambda: str(uuid4()))
    question: str = ""
    options: List[Any] = field(default_factory=list)
    votes: Dict[str, Any] = field(default_factory=dict)  # agent_id -> chosen option
    weights: Dict[str, float] = field(default_factory=dict)  # agent_id -> vote weight
    result: Optional[Any] = None
    confidence: float = 0.0
    decided_at: Optional[datetime] = None


class CollaborationProtocol(ABC):
    """
    Abstract base class for agent collaboration protocols.
    
    Defines how agents communicate, share context, and coordinate.
    """
    
    def __init__(self, config: WitsV3Config):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.message_queue: Dict[str, List[CollaborationMessage]] = {}
        self.shared_contexts: Dict[str, SharedContext] = {}
        self.active_agents: Set[str] = set()
    
    @abstractmethod
    async def register_agent(self, agent_id: str, capabilities: Set[str]) -> bool:
        """Register an agent with the collaboration protocol"""
        pass
    
    @abstractmethod
    async def send_message(self, message: CollaborationMessage) -> bool:
        """Send a message through the collaboration protocol"""
        pass
    
    @abstractmethod
    async def receive_messages(self, agent_id: str) -> List[CollaborationMessage]:
        """Receive messages for a specific agent"""
        pass
    
    @abstractmethod
    async def negotiate_task_distribution(
        self,
        tasks: List[Task],
        agent_capabilities: Dict[str, Set[str]]
    ) -> List[TaskAssignment]:
        """Negotiate optimal task distribution among agents"""
        pass
    
    @abstractmethod
    async def share_context(
        self,
        context: SharedContext,
        from_agent: str,
        to_agents: List[str]
    ) -> bool:
        """Share context between agents"""
        pass
    
    @abstractmethod
    async def consensus_decision(
        self,
        participants: List[str],
        question: str,
        options: List[Any],
        timeout: float = 30.0
    ) -> Decision:
        """Facilitate consensus decision among agents"""
        pass
    
    async def broadcast_message(
        self,
        sender_id: str,
        message_type: str,
        content: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> bool:
        """Broadcast a message to all active agents"""
        message = CollaborationMessage(
            sender_id=sender_id,
            recipient_ids=[],  # Empty means broadcast
            message_type=message_type,
            content=content,
            priority=priority
        )
        return await self.send_message(message)
    
    async def request_response(
        self,
        sender_id: str,
        recipient_id: str,
        request_type: str,
        request_data: Dict[str, Any],
        timeout: float = 10.0
    ) -> Optional[CollaborationMessage]:
        """Send a request and wait for response"""
        request = CollaborationMessage(
            sender_id=sender_id,
            recipient_ids=[recipient_id],
            message_type=request_type,
            content=request_data,
            requires_response=True
        )
        
        # Send request
        if not await self.send_message(request):
            return None
        
        # Wait for response with matching correlation_id
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            messages = await self.receive_messages(sender_id)
            for msg in messages:
                if msg.correlation_id == request.message_id:
                    return msg
            await asyncio.sleep(0.1)
        
        return None
    
    def create_shared_context(
        self,
        owner_id: str,
        initial_data: Dict[str, Any],
        allowed_agents: Dict[str, Set[str]]
    ) -> SharedContext:
        """Create a new shared context"""
        context = SharedContext(
            owner_id=owner_id,
            data=initial_data.copy(),
            access_control=allowed_agents.copy()
        )
        self.shared_contexts[context.context_id] = context
        return context
    
    async def stream_collaboration_event(
        self,
        event_type: str,
        description: str,
        participants: List[str]
    ) -> StreamData:
        """Stream a collaboration event"""
        return StreamData(
            type="action",
            content=f"Collaboration: {description}",
            source="CollaborationProtocol",
            metadata={
                "event_type": event_type,
                "participants": participants,
                "protocol": self.__class__.__name__
            }
        )


class SimpleCollaborationProtocol(CollaborationProtocol):
    """Simple implementation of collaboration protocol for testing"""
    
    def __init__(self, config: WitsV3Config):
        super().__init__(config)
        self.agent_capabilities: Dict[str, Set[str]] = {}
    
    async def register_agent(self, agent_id: str, capabilities: Set[str]) -> bool:
        """Register an agent"""
        self.active_agents.add(agent_id)
        self.agent_capabilities[agent_id] = capabilities
        self.message_queue[agent_id] = []
        self.logger.info(f"Registered agent {agent_id} with capabilities: {capabilities}")
        return True
    
    async def send_message(self, message: CollaborationMessage) -> bool:
        """Send a message"""
        if message.recipient_ids:
            # Direct message(s)
            for recipient in message.recipient_ids:
                if recipient in self.message_queue:
                    self.message_queue[recipient].append(message)
        else:
            # Broadcast
            for agent_id in self.active_agents:
                if agent_id != message.sender_id:
                    self.message_queue[agent_id].append(message)
        
        return True
    
    async def receive_messages(self, agent_id: str) -> List[CollaborationMessage]:
        """Receive messages"""
        if agent_id not in self.message_queue:
            return []
        
        messages = self.message_queue[agent_id]
        self.message_queue[agent_id] = []
        return messages
    
    async def negotiate_task_distribution(
        self,
        tasks: List[Task],
        agent_capabilities: Dict[str, Set[str]]
    ) -> List[TaskAssignment]:
        """Simple task distribution based on capabilities"""
        assignments = []
        
        for task in tasks:
            best_agent = None
            best_score = 0.0
            
            for agent_id, capabilities in agent_capabilities.items():
                # Score based on capability match
                matching_caps = task.required_capabilities.intersection(capabilities)
                score = len(matching_caps) / max(1, len(task.required_capabilities))
                
                if score > best_score:
                    best_score = score
                    best_agent = agent_id
            
            if best_agent:
                assignment = TaskAssignment(
                    task_id=task.task_id,
                    agent_id=best_agent,
                    accepted=True,
                    confidence=best_score
                )
                assignments.append(assignment)
        
        return assignments
    
    async def share_context(
        self,
        context: SharedContext,
        from_agent: str,
        to_agents: List[str]
    ) -> bool:
        """Share context with agents"""
        # Add read permissions for target agents
        for agent_id in to_agents:
            if agent_id not in context.access_control:
                context.access_control[agent_id] = set()
            context.access_control[agent_id].add("read")
        
        # Notify agents about shared context
        await self.broadcast_message(
            from_agent,
            "context_shared",
            {"context_id": context.context_id},
            MessagePriority.HIGH
        )
        
        return True
    
    async def consensus_decision(
        self,
        participants: List[str],
        question: str,
        options: List[Any],
        timeout: float = 30.0
    ) -> Decision:
        """Simple majority voting"""
        decision = Decision(
            question=question,
            options=options
        )
        
        # Request votes from participants
        for participant in participants:
            await self.send_message(CollaborationMessage(
                sender_id="consensus_protocol",
                recipient_ids=[participant],
                message_type="vote_request",
                content={"question": question, "options": options},
                requires_response=True
            ))
        
        # Collect votes (simplified - in real implementation would be async)
        # For now, simulate random voting
        import random
        for participant in participants:
            decision.votes[participant] = random.choice(options)
            decision.weights[participant] = 1.0
        
        # Determine result by majority
        vote_counts = {}
        for option in options:
            vote_counts[option] = sum(
                1 for vote in decision.votes.values() 
                if vote == option
            )
        
        decision.result = max(vote_counts, key=vote_counts.get)
        decision.confidence = vote_counts[decision.result] / len(participants)
        decision.decided_at = datetime.now()
        
        return decision


# Test function
async def test_agent_collaboration():
    """Test the agent collaboration framework"""
    from .config import load_config
    
    print("Testing Agent Collaboration Framework...")
    
    config = load_config("config.yaml")
    protocol = SimpleCollaborationProtocol(config)
    
    # Register agents
    await protocol.register_agent("agent1", {"coding", "testing"})
    await protocol.register_agent("agent2", {"research", "writing"})
    await protocol.register_agent("agent3", {"coding", "research"})
    
    print("✓ Registered 3 agents")
    
    # Test messaging
    await protocol.send_message(CollaborationMessage(
        sender_id="agent1",
        recipient_ids=["agent2"],
        message_type="greeting",
        content={"text": "Hello agent2!"}
    ))
    
    messages = await protocol.receive_messages("agent2")
    print(f"✓ Agent2 received {len(messages)} messages")
    
    # Test task distribution
    tasks = [
        Task(description="Write code", required_capabilities={"coding"}),
        Task(description="Research topic", required_capabilities={"research"}),
        Task(description="Write tests", required_capabilities={"coding", "testing"})
    ]
    
    assignments = await protocol.negotiate_task_distribution(
        tasks, 
        protocol.agent_capabilities
    )
    
    print(f"✓ Distributed {len(assignments)} tasks")
    for assignment in assignments:
        print(f"  Task {assignment.task_id} -> Agent {assignment.agent_id} (confidence: {assignment.confidence:.2f})")
    
    # Test consensus decision
    decision = await protocol.consensus_decision(
        ["agent1", "agent2", "agent3"],
        "Which framework to use?",
        ["FastAPI", "Django", "Flask"]
    )
    
    print(f"✓ Consensus decision: {decision.result} (confidence: {decision.confidence:.2f})")
    
    print("\nAgent Collaboration Framework tests completed! 🎉")


if __name__ == "__main__":
    asyncio.run(test_agent_collaboration())