"""Base chunking strategy interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

from src.core.schemas import DocumentChunk


@dataclass
class ChunkingConfig:
    """Configuration for chunking."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass

    @abstractmethod
    def chunk(
        self,
        text: str,
        metadata: Dict[str, Any] = None,
    ) -> List[DocumentChunk]:
        """Split text into chunks."""
        pass

    def _create_chunk(
        self,
        content: str,
        index: int,
        metadata: Dict[str, Any],
    ) -> DocumentChunk:
        """Helper to create a DocumentChunk."""
        import hashlib
        chunk_id = hashlib.md5(f"{content[:50]}_{index}".encode()).hexdigest()[:16]
        
        return DocumentChunk(
            id=chunk_id,
            content=content,
            document_id=metadata.get("document_id", ""),
            chunk_index=index,
            metadata={
                **metadata,
                "chunk_strategy": self.name,
                "chunk_size": len(content),
                "word_count": len(content.split()),
            },
        )
