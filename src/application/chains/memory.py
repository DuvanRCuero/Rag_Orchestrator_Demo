import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_classic.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.core.config import settings
from src.core.schemas import Conversation, ConversationTurn


@dataclass
class ConversationContext:
    """Enhanced conversation context with metadata."""

    session_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_turn(self, role: str, content: str, **kwargs):
        """Add a conversation turn."""
        turn = ConversationTurn(
            role=role, content=content, timestamp=datetime.utcnow(), **kwargs
        )
        self.turns.append(turn)
        self.updated_at = datetime.utcnow()

    def get_recent_turns(self, n: int = None) -> List[ConversationTurn]:
        """Get recent conversation turns."""
        n = n or settings.MEMORY_WINDOW_SIZE
        return self.turns[-n:]

    def get_formatted_history(self, token_limit: int = 2000) -> str:
        """Format conversation history for context."""
        if not self.turns:
            return ""

        # Start from most recent and work backwards until token limit
        history_parts = []
        total_tokens = 0

        for turn in reversed(self.turns):
            turn_text = f"{turn.role.upper()}: {turn.content}\n"
            turn_tokens = len(turn_text.split())

            if total_tokens + turn_tokens > token_limit:
                break

            history_parts.append(turn_text)
            total_tokens += turn_tokens

        # Reverse to maintain chronological order
        return "\n".join(reversed(history_parts))

    def to_langchain_messages(self) -> List[BaseMessage]:
        """Convert to LangChain message format."""
        messages = []
        for turn in self.turns:
            if turn.role == "user":
                messages.append(HumanMessage(content=turn.content))
            elif turn.role == "assistant":
                messages.append(AIMessage(content=turn.content))
        return messages


class EnhancedConversationMemory:
    """Production-grade conversation memory with multiple strategies."""

    def __init__(self, strategy: str = None):
        self.strategy = strategy or settings.MEMORY_TYPE
        self.sessions: Dict[str, ConversationContext] = {}

        # Initialize LangChain memories for different strategies
        if self.strategy == "buffer":
            self._memory_class = ConversationBufferWindowMemory
        elif self.strategy == "summary":
            self._memory_class = ConversationSummaryMemory
        else:
            self._memory_class = ConversationBufferWindowMemory

    def get_session(self, session_id: str) -> ConversationContext:
        """Get or create a conversation session."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationContext(
                session_id=session_id,
                metadata={
                    "strategy": self.strategy,
                    "created": datetime.utcnow().isoformat(),
                },
            )
        return self.sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str, **kwargs):
        """Add a message to conversation memory."""
        session = self.get_session(session_id)
        session.add_turn(role, content, **kwargs)

        # Trim old conversations if needed
        self._trim_session(session)

    def _trim_session(self, session: ConversationContext):
        """Trim session to prevent unbounded growth."""
        max_turns = 100  # Hard limit
        if len(session.turns) > max_turns:
            # Keep recent turns and summary of older ones
            keep_recent = 20
            older_turns = session.turns[:-keep_recent]
            recent_turns = session.turns[-keep_recent:]

            # Create summary of older turns
            summary = self._summarize_conversation(older_turns)

            # Replace older turns with summary
            summary_turn = ConversationTurn(
                role="system",
                content=f"Summary of previous conversation: {summary}",
                timestamp=older_turns[-1].timestamp
                if older_turns
                else datetime.utcnow(),
            )

            session.turns = [summary_turn] + recent_turns

    def _summarize_conversation(self, turns: List[ConversationTurn]) -> str:
        """Generate summary of conversation turns."""
        if not turns:
            return ""

        # Simple summarization - in production, use LLM
        user_questions = [t.content for t in turns if t.role == "user"]
        assistant_answers = [t.content for t in turns if t.role == "assistant"]

        return f"Discussed {len(user_questions)} topics including: {', '.join(user_questions[:3])}"

    def clear_session(self, session_id: str):
        """Clear a specific session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_conversation_for_context(
        self, session_id: str, max_tokens: int = 1000
    ) -> str:
        """Get formatted conversation history for context."""
        session = self.get_session(session_id)
        return session.get_formatted_history(token_limit=max_tokens)

    def to_conversation_model(self, session_id: str) -> Conversation:
        """Convert to Pydantic Conversation model."""
        session = self.get_session(session_id)
        return Conversation(
            session_id=session_id,
            turns=session.turns,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )


# Global memory instance
conversation_memory = EnhancedConversationMemory()
