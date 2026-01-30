"""LLM infrastructure package with factory registration."""

from src.infrastructure.llm.llm_factory import LLMFactory
from src.infrastructure.llm.openai_client import AsyncOpenAIService

# Register OpenAI (always available)
LLMFactory.register("openai", AsyncOpenAIService)

# Register Anthropic (optional)
try:
    from src.infrastructure.llm.anthropic_client import AnthropicService
    LLMFactory.register("anthropic", AnthropicService)
except ImportError:
    pass  # anthropic package not installed

# Register Local/Ollama (optional)
try:
    from src.infrastructure.llm.local_client import LocalLLMService
    LLMFactory.register("local", LocalLLMService)
    LLMFactory.register("ollama", LocalLLMService)
except ImportError:
    pass

__all__ = ["LLMFactory", "AsyncOpenAIService"]
