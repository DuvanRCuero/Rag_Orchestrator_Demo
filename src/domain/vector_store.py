"""Base vector store interface for the domain layer."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.core.schemas import DocumentChunk


class VectorStore(ABC):
    """Abstract base class for vector store implementations."""

    @abstractmethod
    async def create_collection(
        self, collection_name: str, vector_size: int
    ) -> bool:
        """Create a new collection."""
        pass

    @abstractmethod
    async def upsert_chunks(
        self, chunks: List[DocumentChunk], embeddings: List[List[float]]
    ) -> bool:
        """Insert or update document chunks with their embeddings."""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Search for similar chunks."""
        pass

    @abstractmethod
    async def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Perform hybrid search combining semantic and keyword search."""
        pass

    @abstractmethod
    async def delete_by_document_id(self, document_id: str) -> bool:
        """Delete all chunks for a given document."""
        pass

    @abstractmethod
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        pass

    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        pass
