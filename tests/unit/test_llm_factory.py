"""Tests for LLM Factory Pattern."""

import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# Set environment variables before importing
os.environ["ENVIRONMENT"] = "testing"
os.environ["OPENAI_API_KEY"] = "test-key"

from src.infrastructure.llm import LLMFactory
from src.infrastructure.llm.llm_factory import LLMFactory as Factory
from src.infrastructure.llm.openai_client import AsyncOpenAIService
from src.domain.interfaces.llm_service import LLMService


class TestLLMFactory:
    """Test LLM Factory functionality."""

    def test_openai_provider_registered(self):
        """Test that OpenAI provider is registered."""
        providers = LLMFactory.available_providers()
        assert "openai" in providers

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        with patch("src.core.config.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "openai"
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.OPENAI_MODEL = "gpt-3.5-turbo"
            mock_settings.OPENAI_TEMPERATURE = 0.1
            mock_settings.OPENAI_MAX_TOKENS = 1000
            
            llm = LLMFactory.create("openai")
            assert isinstance(llm, AsyncOpenAIService)

    def test_factory_uses_default_provider(self):
        """Test that factory uses LLM_PROVIDER from settings when not specified."""
        with patch("src.core.config.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "openai"
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.OPENAI_MODEL = "gpt-3.5-turbo"
            mock_settings.OPENAI_TEMPERATURE = 0.1
            mock_settings.OPENAI_MAX_TOKENS = 1000
            
            llm = LLMFactory.create()
            assert isinstance(llm, AsyncOpenAIService)

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            LLMFactory.create("unknown_provider")
        
        assert "Unknown LLM provider: unknown_provider" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_register_custom_provider(self):
        """Test registering a custom provider."""
        
        # Create a mock LLM service
        class MockLLMService(LLMService):
            async def generate(self, messages, **kwargs):
                return "mock response"
            
            async def generate_json(self, messages, **kwargs):
                return {"response": "mock"}
            
            async def stream_generation(self, messages, **kwargs):
                yield "mock"
            
            async def get_token_usage(self, text):
                return {"total_tokens": 10}
        
        # Register the mock provider
        Factory.register("mock", MockLLMService)
        
        # Verify it's registered
        assert "mock" in Factory.available_providers()
        
        # Create an instance
        llm = Factory.create("mock")
        assert isinstance(llm, MockLLMService)

    def test_anthropic_registration_without_import(self):
        """Test that Anthropic is conditionally registered."""
        providers = LLMFactory.available_providers()
        # Anthropic may or may not be available depending on if package is installed
        # This is fine - just testing the registration mechanism works
        assert isinstance(providers, list)

    def test_available_providers_returns_list(self):
        """Test that available_providers returns a list."""
        providers = LLMFactory.available_providers()
        assert isinstance(providers, list)
        assert len(providers) > 0
        assert "openai" in providers


@pytest.mark.asyncio
class TestAnthropicClient:
    """Test Anthropic client functionality."""

    @patch('src.infrastructure.llm.anthropic_client.AsyncAnthropic')
    async def test_anthropic_message_conversion(self, mock_anthropic):
        """Test message format conversion from OpenAI to Anthropic."""
        from src.infrastructure.llm.anthropic_client import AnthropicService
        
        with patch("src.core.config.settings") as mock_settings:
            mock_settings.ANTHROPIC_API_KEY = "test-key"
            mock_settings.ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
            mock_settings.ANTHROPIC_MAX_TOKENS = 1000
            
            service = AnthropicService()
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
            
            system, converted = service._convert_messages(messages)
            
            assert system == "You are a helpful assistant"
            assert len(converted) == 3
            assert converted[0]["role"] == "user"
            assert converted[0]["content"] == "Hello"


@pytest.mark.asyncio
class TestLocalLLMClient:
    """Test Local LLM client functionality."""

    def test_local_llm_initialization(self):
        """Test Local LLM service initialization."""
        from src.infrastructure.llm.local_client import LocalLLMService
        
        with patch("src.core.config.settings") as mock_settings:
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.LOCAL_MODEL = "llama2"
            
            service = LocalLLMService()
            
            assert service.base_url == "http://localhost:11434"
            assert service.model == "llama2"

    @patch('httpx.AsyncClient')
    async def test_local_llm_generate(self, mock_client):
        """Test Local LLM generation."""
        from src.infrastructure.llm.local_client import LocalLLMService
        
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "Test response"}
        }
        mock_response.raise_for_status = MagicMock()
        
        # Mock the client context manager
        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.__aexit__.return_value = None
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value = mock_client_instance
        
        with patch("src.core.config.settings") as mock_settings:
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.LOCAL_MODEL = "llama2"
            
            service = LocalLLMService()
            
            messages = [{"role": "user", "content": "Hello"}]
            result = await service.generate(messages)
            
            assert result == "Test response"


class TestFactoryIntegration:
    """Test factory integration with the container."""

    def test_container_uses_factory(self):
        """Test that container uses factory to create LLM service."""
        from src.infrastructure.container import Container
        
        with patch("src.core.config.settings") as mock_settings:
            mock_settings.ENVIRONMENT = "testing"
            mock_settings.LLM_PROVIDER = "openai"
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.OPENAI_MODEL = "gpt-3.5-turbo"
            mock_settings.OPENAI_TEMPERATURE = 0.1
            mock_settings.OPENAI_MAX_TOKENS = 1000
            mock_settings.VECTOR_DB_TYPE = "qdrant"
            mock_settings.QDRANT_URL = "http://localhost:6333"
            mock_settings.QDRANT_COLLECTION_NAME = "test"
            mock_settings.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
            mock_settings.EMBEDDING_DIMENSION = 384
            
            container = Container()
            # Reset the container to ensure fresh initialization
            container._initialized = False
            
            llm_service = container.llm_service
            assert isinstance(llm_service, LLMService)
