"""Embedding service interface."""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingServiceInterface(ABC):

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        pass
