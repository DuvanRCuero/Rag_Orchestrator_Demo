"""Local LLM implementation using Ollama."""

import json
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from src.domain.interfaces.llm_service import LLMService
from src.core.config import settings
from src.core.config_models import LLMConfig, get_config
from src.core.exceptions import GenerationError


class LocalLLMService(LLMService):

    def __init__(self, config: LLMConfig = None):
        self.config = config or get_config().llm
        
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = self.config.model
        self._langchain_llm = None

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": temperature or 0.1,
                            "num_predict": max_tokens or 1000,
                        }
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result["message"]["content"]
                
        except Exception as e:
            raise GenerationError(
                detail=f"Local LLM generation failed: {str(e)}",
                metadata={"model": self.model, "base_url": self.base_url},
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
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": True,
                    }
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                yield data["message"]["content"]
                                
        except Exception as e:
            raise GenerationError(
                detail=f"Local LLM streaming failed: {str(e)}",
                metadata={"model": self.model},
            )

    async def get_token_usage(self, text: str) -> Dict[str, int]:
        # Approximate token count
        # Using 1.3 as average multiplier: accounts for tokens per word + punctuation
        TOKEN_MULTIPLIER = 1.3
        estimated_tokens = len(text.split()) * TOKEN_MULTIPLIER
        return {
            "estimated_tokens": int(estimated_tokens),
            "provider": "local",
            "model": self.model,
        }

    @property
    def langchain_llm(self):
        if self._langchain_llm is None:
            try:
                from langchain_community.llms import Ollama
                self._langchain_llm = Ollama(
                    model=self.model,
                    base_url=self.base_url,
                )
            except ImportError:
                raise ImportError("langchain-community required for local LLM")
        return self._langchain_llm
