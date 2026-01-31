import asyncio
import json
from typing import Any, Dict, List, Optional, Union

import backoff
from langchain_classic.callbacks import AsyncIteratorCallbackHandler
from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage, SystemMessage

from openai import APIError, AsyncOpenAI, RateLimitError

from src.core.config import settings
from src.core.exceptions import GenerationError
from src.core.logging import get_logger
from src.domain.interfaces.llm_service import LLMService
from src.infrastructure.resilience import CircuitBreaker

logger = get_logger(__name__)


class AsyncOpenAIService(LLMService):
    """Async OpenAI service with retry, fallback, and streaming support."""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.temperature = settings.OPENAI_TEMPERATURE
        self.max_tokens = settings.OPENAI_MAX_TOKENS

        # Initialize LangChain LLM for LCEL chains
        self.langchain_llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=settings.OPENAI_API_KEY,
            streaming=True,
        )

        # Initialize circuit breaker
        self._circuit_breaker = CircuitBreaker(
            name="openai",
            failure_threshold=5,
            recovery_timeout=30.0,
        )
        logger.info(
            "openai_service_initialized",
            model=self.model,
            temperature=self.temperature,
        )

    @backoff.on_exception(
        backoff.expo, (APIError, RateLimitError), max_tries=3, max_time=30
    )
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, Any]:
        """Generate completion with retry logic."""
        logger.debug(
            "generate_started",
            message_count=len(messages),
            model=self.model,
        )

        async def _do_generate():
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=stream,
                **kwargs,
            )

            if stream:
                return response
            else:
                return response.choices[0].message.content

        try:
            result = await self._circuit_breaker.call(_do_generate)
            if not stream:
                logger.info(
                    "generate_completed",
                    response_length=len(result) if result else 0,
                    model=self.model,
                )
            return result

        except Exception as e:
            logger.error(
                "generate_failed",
                error=str(e),
                model=self.model,
                circuit_state=self._circuit_breaker.state.value,
            )
            raise GenerationError(
                detail=f"OpenAI generation failed: {str(e)}",
                metadata={
                    "model": self.model,
                    "message_count": len(messages),
                    "stream": stream,
                },
            )

    async def generate_with_prompt(
        self, system_prompt: str, user_prompt: str, **kwargs
    ) -> str:
        """Generate with system and user prompts."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self.generate(messages, **kwargs)

    async def generate_json(
        self,
        messages: List[Dict[str, str]],
        response_format: Dict[str, Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate structured JSON response."""
        if response_format is None:
            response_format = {
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "response": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                },
            }

        response = await self.generate(
            messages, response_format=response_format, **kwargs
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"response": response, "metadata": {}}

    async def stream_generation(
        self,
        messages: List[Dict[str, str]],
        callback_handler: Optional[AsyncIteratorCallbackHandler] = None,
    ):
        """Stream generation token by token."""
        try:
            if callback_handler:
                # Use LangChain streaming
                llm = ChatOpenAI(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    api_key=settings.OPENAI_API_KEY,
                    streaming=True,
                    callbacks=[callback_handler],
                )

                # Convert messages to LangChain format
                langchain_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        langchain_messages.append(SystemMessage(content=msg["content"]))
                    elif msg["role"] == "user":
                        langchain_messages.append(HumanMessage(content=msg["content"]))

                # Start generation in background
                asyncio.create_task(llm.agenerate([langchain_messages]))
                raise StopAsyncIteration

            else:
                # Use native OpenAI streaming
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                )

                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content

        except Exception as e:
            raise GenerationError(
                detail=f"Streaming generation failed: {str(e)}",
                metadata={"model": self.model, "streaming": True},
            )

    async def get_token_usage(self, text: str) -> Dict[str, int]:
        """Estimate token usage for text."""
        try:
            import tiktoken

            encoding = tiktoken.encoding_for_model(self.model)
            tokens = encoding.encode(text)
            return {
                "total_tokens": len(tokens),
                "prompt_tokens": len(tokens),
                "completion_tokens": 0,
            }
        except:
            # Fallback estimation
            words = len(text.split())
            return {
                "total_tokens": int(words * 1.3),  # Rough estimation
                "prompt_tokens": int(words * 1.3),
                "completion_tokens": 0,
            }

