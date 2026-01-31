"""Anthropic Claude LLM implementation."""

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from src.domain.interfaces.llm_service import LLMService
from src.core.config import settings
from src.core.config_models import LLMConfig, get_config
from src.core.exceptions import GenerationError


class AnthropicService(LLMService):

    def __init__(self, config: LLMConfig = None):
        self.config = config or get_config().llm
        
        try:
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(api_key=self.config.api_key)
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        
        self.model = self.config.model
        self.max_tokens = self.config.max_tokens
        self._langchain_llm = None

    def _convert_messages(self, messages: List[Dict[str, str]]) -> tuple:
        """Convert OpenAI message format to Anthropic format."""
        system_message = None
        converted = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                converted.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        return system_message, converted

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        try:
            system, msgs = self._convert_messages(messages)
            
            # Build parameters - only include system if present
            params = {
                "model": self.model,
                "max_tokens": max_tokens or self.max_tokens,
                "messages": msgs,
            }
            if system:
                params["system"] = system
            
            response = await self.client.messages.create(**params)
            
            return response.content[0].text
            
        except Exception as e:
            raise GenerationError(
                detail=f"Anthropic generation failed: {str(e)}",
                metadata={"model": self.model},
            )

    async def generate_json(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Dict[str, Any]:
        response = await self.generate(messages, **kwargs)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"response": response, "metadata": {}}

    async def stream_generation(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> AsyncIterator[str]:
        try:
            system, msgs = self._convert_messages(messages)
            
            # Build parameters - only include system if present
            params = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": msgs,
            }
            if system:
                params["system"] = system
            
            async with self.client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text
                    
        except Exception as e:
            raise GenerationError(
                detail=f"Anthropic streaming failed: {str(e)}",
                metadata={"model": self.model},
            )

    async def get_token_usage(self, text: str) -> Dict[str, int]:
        # Approximate token count (Anthropic uses similar tokenization)
        # Using 1.3 as average multiplier: accounts for tokens per word + punctuation
        TOKEN_MULTIPLIER = 1.3
        estimated_tokens = len(text.split()) * TOKEN_MULTIPLIER
        return {
            "estimated_tokens": int(estimated_tokens),
            "provider": "anthropic",
        }

    @property
    def langchain_llm(self):
        if self._langchain_llm is None:
            try:
                from langchain_anthropic import ChatAnthropic
                self._langchain_llm = ChatAnthropic(
                    model=self.model,
                    anthropic_api_key=settings.ANTHROPIC_API_KEY,
                    max_tokens=self.max_tokens,
                )
            except ImportError:
                raise ImportError("langchain-anthropic required. Install with: pip install langchain-anthropic")
        return self._langchain_llm
