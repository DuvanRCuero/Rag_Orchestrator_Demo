"""LLM service interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, AsyncIterator


class LLMService(ABC):

    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        pass

    @abstractmethod
    async def generate_json(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def stream_generation(self, messages: List[Dict[str, str]], **kwargs) -> AsyncIterator[str]:
        pass

    @abstractmethod
    async def get_token_usage(self, text: str) -> Dict[str, int]:
        pass

    @property
    @abstractmethod
    def langchain_llm(self):
        pass
