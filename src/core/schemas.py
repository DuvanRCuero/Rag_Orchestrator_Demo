from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class DocumentType(str, Enum):
    TXT = "txt"
    PDF = "pdf"
    MD = "md"
    HTML = "html"
    JSON = "json"


class DocumentChunk(BaseModel):
    """Represents a chunk of a document with metadata."""

    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    document_id: str
    chunk_index: int
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True


class Document(BaseModel):
    """Represents a full document."""

    id: str
    name: str
    content: str
    type: DocumentType
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunks: List[DocumentChunk] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class QueryRequest(BaseModel):
    """Request schema for querying the RAG system."""

    query: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=2.0)
    stream: bool = False

    @validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace")
        return v.strip()


class QueryResponse(BaseModel):
    """Response schema for RAG queries."""

    answer: str
    sources: List[DocumentChunk] = Field(default_factory=list)
    session_id: str
    query_time: float
    token_usage: Optional[Dict[str, int]] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class IngestionRequest(BaseModel):
    """Request schema for document ingestion."""

    documents: List[str]  # Base64 encoded or file paths
    document_type: DocumentType
    metadata: Optional[Dict[str, Any]] = None
    chunk_size: Optional[int] = Field(default=1000, ge=100, le=5000)
    chunk_overlap: Optional[int] = Field(default=200, ge=0, le=1000)


class IngestionResponse(BaseModel):
    """Response schema for document ingestion."""

    document_ids: List[str]
    total_chunks: int
    processing_time: float
    status: str = "success"


class ConversationTurn(BaseModel):
    """Represents a single turn in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Conversation(BaseModel):
    """Represents a conversation session."""

    session_id: str
    turns: List[ConversationTurn] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
