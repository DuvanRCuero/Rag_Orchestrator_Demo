import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.v1.dependencies import get_conversation_memory
from src.application.chains.memory import EnhancedConversationMemory
from src.core.schemas import Conversation

router = APIRouter()


@router.post(
    "/create",
    summary="Create a new session",
    description="Create a new conversation session and return the session ID.",
)
async def create_session(
    metadata: Optional[dict] = None,
    memory: EnhancedConversationMemory = Depends(get_conversation_memory),
):
    """Create a new conversation session."""
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())

        # Initialize session
        session = memory.get_session(session_id)
        if metadata:
            session.metadata.update(metadata)

        return {
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "metadata": session.metadata,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}",
        )


@router.get(
    "/{session_id}",
    response_model=Conversation,
    summary="Get session details",
    description="Get details of a specific conversation session.",
)
async def get_session(
    session_id: str,
    memory: EnhancedConversationMemory = Depends(get_conversation_memory),
):
    """Get conversation session details."""
    try:
        conversation = memory.to_conversation_model(session_id)
        return conversation

    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session: {str(e)}",
        )


@router.get(
    "/",
    summary="List all sessions",
    description="List all active conversation sessions with metadata.",
)
async def list_sessions(
    limit: int = Query(50, description="Maximum number of sessions to return"),
    offset: int = Query(0, description="Number of sessions to skip"),
    memory: EnhancedConversationMemory = Depends(get_conversation_memory),
):
    """List all conversation sessions."""
    try:
        # Get limited sessions (in production, would paginate properly)
        sessions = list(memory.sessions.values())

        # Apply pagination
        paginated_sessions = sessions[offset : offset + limit]

        return {
            "sessions": [
                {
                    "session_id": session.session_id,
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "turn_count": len(session.turns),
                    "metadata": session.metadata,
                }
                for session in paginated_sessions
            ],
            "total": len(sessions),
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sessions: {str(e)}",
        )


@router.delete(
    "/{session_id}",
    summary="Delete a session",
    description="Delete a conversation session and all its history.",
)
async def delete_session(
    session_id: str,
    memory: EnhancedConversationMemory = Depends(get_conversation_memory),
):
    """Delete a conversation session."""
    try:
        memory.clear_session(session_id)

        return {
            "status": "success",
            "message": f"Session {session_id} deleted successfully",
            "session_id": session_id,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete session: {str(e)}",
        )


@router.post(
    "/{session_id}/summarize",
    summary="Summarize session",
    description="Generate a summary of the conversation session.",
)
async def summarize_session(
    session_id: str,
    memory: EnhancedConversationMemory = Depends(get_conversation_memory),
):
    """Generate summary of a conversation session."""
    try:
        # Get session
        session = memory.get_session(session_id)

        if not session.turns:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No conversation history to summarize",
            )

        # Generate summary (simplified - in production, use LLM)
        user_turns = [t for t in session.turns if t.role == "user"]
        assistant_turns = [t for t in session.turns if t.role == "assistant"]

        summary = {
            "total_turns": len(session.turns),
            "user_turns": len(user_turns),
            "assistant_turns": len(assistant_turns),
            "topics": [t.content[:100] for t in user_turns[:3]],
            "time_span": {
                "start": session.turns[0].timestamp.isoformat(),
                "end": session.turns[-1].timestamp.isoformat(),
            },
        }

        return {"session_id": session_id, "summary": summary}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to summarize session: {str(e)}",
        )
