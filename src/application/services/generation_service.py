"""Generation Service - handles LLM generation."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, AsyncIterator

from src.domain.interfaces.llm_service import LLMService
from src.core.schemas import DocumentChunk
from src.core.exceptions import GenerationError
from src.application.chains.prompts import PromptRegistry


@dataclass
class GenerationResult:
    answer: str
    confidence_score: Optional[float] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    generation_time: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)


class GenerationService:

    def __init__(self, llm_service: LLMService):
        self.llm = llm_service

    async def generate(
        self,
        query: str,
        context_chunks: List[DocumentChunk],
        history: str = "",
    ) -> GenerationResult:
        start_time = time.time()

        context = self._format_context(context_chunks)
        prompt = PromptRegistry.SYSTEM_PROMPT_RAG.format(
            context=context,
            history=history,
            question=query,
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]

        try:
            answer = await self.llm.generate(messages)
            token_usage = await self.llm.get_token_usage(answer)
            citations = self._extract_citations(answer, context_chunks)

            return GenerationResult(
                answer=answer,
                citations=citations,
                generation_time=time.time() - start_time,
                token_usage=token_usage,
            )
        except Exception as e:
            raise GenerationError(
                detail=f"Generation failed: {str(e)}",
                metadata={"query": query[:100]},
            )

    async def stream(
        self,
        query: str,
        context_chunks: List[DocumentChunk],
        history: str = "",
    ) -> AsyncIterator[str]:
        context = self._format_context(context_chunks)
        prompt = PromptRegistry.SYSTEM_PROMPT_RAG.format(
            context=context,
            history=history,
            question=query,
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]

        async for token in self.llm.stream_generation(messages):
            yield token

    def _format_context(self, chunks: List[DocumentChunk]) -> str:
        parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get('source', 'Unknown')
            parts.append(f"[Document {i + 1} - Source: {source}]\n{chunk.content}\n")
        return "\n---\n".join(parts)

    def _extract_citations(
        self,
        answer: str,
        chunks: List[DocumentChunk],
    ) -> List[Dict[str, Any]]:
        citations = []
        for i, chunk in enumerate(chunks):
            ref = f"[Document {i + 1}]"
            if ref in answer:
                citations.append({
                    "document_index": i,
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.id,
                    "reference": ref,
                })
        return citations
