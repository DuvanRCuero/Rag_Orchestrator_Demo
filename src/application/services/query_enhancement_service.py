"""Query Enhancement Service - handles multi-query and HyDE."""

import re
from typing import List

from src.domain.interfaces import LLMService, EmbeddingServiceInterface
from src.application.chains.prompts import PromptRegistry


class QueryEnhancementService:

    def __init__(
        self,
        llm_service: LLMService,
        embedding_service: EmbeddingServiceInterface,
    ):
        self.llm = llm_service
        self.embedding_service = embedding_service

    async def generate_query_variations(self, query: str) -> List[str]:
        prompt = PromptRegistry.SYSTEM_PROMPT_MULTI_QUERY.format(question=query)
        messages = [
            {"role": "system", "content": "Generate query variations."},
            {"role": "user", "content": prompt},
        ]

        try:
            output = await self.llm.generate(messages)
            return self._parse_queries(output)
        except Exception:
            return []

    async def generate_hyde_embedding(self, query: str) -> List[float]:
        prompt = f"Write a detailed paragraph that would answer this question: {query}"
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        try:
            hypothetical_doc = await self.llm.generate(messages)
            embeddings = await self.embedding_service.embed_texts([hypothetical_doc])
            return embeddings[0]
        except Exception:
            return await self.embedding_service.embed_query(query)

    def _parse_queries(self, output: str) -> List[str]:
        queries = []
        for line in output.strip().split("\n"):
            clean = re.sub(r"^[\d\.\-•]\s*", "", line.strip())
            if clean and len(clean) > 10:
                queries.append(clean)
        return queries[:3]
