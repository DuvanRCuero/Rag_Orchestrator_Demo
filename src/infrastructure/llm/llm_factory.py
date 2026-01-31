"""LLM Factory - Open for extension, closed for modification."""

from typing import Dict, Type, Optional
from src.domain.interfaces.llm_service import LLMService
from src.core.config import settings
from src.core.config_models import LLMConfig


class LLMFactory:
    _providers: Dict[str, Type[LLMService]] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[LLMService]):
        """Register a new LLM provider."""
        cls._providers[name] = provider_class

    @classmethod
    def create(cls, provider: str = None, config: Optional[LLMConfig] = None) -> LLMService:
        """Create an LLM service instance."""
        provider = provider or settings.LLM_PROVIDER
        if provider not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(f"Unknown LLM provider: {provider}. Available: {available}")
        return cls._providers[provider](config)

    @classmethod
    def available_providers(cls) -> list:
        """List available providers."""
        return list(cls._providers.keys())
