import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from src.application.chains.prompts import (CONFIDENCE_PROMPT, HYDE_PROMPT,
                                            MULTI_QUERY_PROMPT, RAG_PROMPT,
                                            SELF_REFLECTION_PROMPT,
                                            VERIFICATION_PROMPT)
from src.core.config import settings
from src.core.exceptions import GenerationError, RetrievalError
from src.core.logging import get_logger
from src.core.schemas import DocumentChunk, QueryRequest, QueryResponse
from src.domain.interfaces import LLMService, EmbeddingServiceInterface, VectorStore

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Enhanced retrieval result with metadata."""

    chunks: List[DocumentChunk]
    retrieval_time: float
    query_variations: List[str] = field(default_factory=list)
    retrieval_method: str = "semantic"
    reranked: bool = False


@dataclass
class GenerationResult:
    """Enhanced generation result with verification."""

    answer: str
    verification_report: Optional[Dict[str, Any]] = None
    reflection_critique: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    generation_time: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)


class AdvancedRAGChain:
    """Production-grade RAG chain with multiple hallucination mitigation techniques."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingServiceInterface,
        llm_service: LLMService,
        conversation_memory=None,
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.conversation_memory = conversation_memory

        # Initialize chains
        self._setup_lcel_chains()
        self._setup_custom_chains()

    def _setup_lcel_chains(self):
        """Setup LCEL chains for standard operations."""

        # Basic RAG chain using LCEL
        self.rag_chain_lcel = (
            {
                "context": RunnablePassthrough(),
                "history": RunnablePassthrough(),
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | self.llm_service.langchain_llm
            | StrOutputParser()
        )

        # Multi-query generation chain
        self.multi_query_chain = (
            {"question": RunnablePassthrough()}
            | MULTI_QUERY_PROMPT
            | self.llm_service.langchain_llm
            | StrOutputParser()
            | RunnableLambda(self._parse_multi_queries)
        )

        # HyDE (Hypothetical Document Embeddings) chain
        self.hyde_chain = (
            {"question": RunnablePassthrough()}
            | HYDE_PROMPT
            | self.llm_service.langchain_llm
            | StrOutputParser()
        )

    def _setup_custom_chains(self):
        """Setup custom chains for complex operations."""
        pass  # Custom logic implemented in methods

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_multi_query: bool = True,
        use_hyde: bool = False,
        use_reranking: bool = True,
    ) -> RetrievalResult:
        """Enhanced retrieval with multiple strategies."""
        import time

        start_time = time.time()

        query_variations = [query]

        # Strategy 1: Multi-Query Generation
        if use_multi_query:
            try:
                query_variations.extend(await self._generate_query_variations(query))
            except Exception as e:
                logger.warning("multi_query_generation_failed", error=str(e))

        # Strategy 2: HyDE (Hypothetical Document Embeddings)
        hyde_embedding = None
        if use_hyde:
            try:
                hypothetical_doc = await self._generate_hypothetical_document(query)
                hyde_embedding = await self.embedding_service.embed_texts(
                    [hypothetical_doc]
                )[0]
            except Exception as e:
                logger.warning("hyde_generation_failed", error=str(e))

        # Retrieve for each query variation
        all_chunks = []
        for q in query_variations:
            # Get embedding for query
            if q == query and hyde_embedding:
                # Use HyDE embedding for original query
                query_embedding = hyde_embedding
            else:
                query_embedding = await self.embedding_service.embed_query(q)

            # Perform search
            if settings.USE_HYBRID_SEARCH:
                chunks = await self.vector_store.hybrid_search(
                    query_embedding=query_embedding,
                    query_text=q,
                    top_k=top_k * 2,  # Get more for deduplication
                )
            else:
                chunks = await self.vector_store.search(
                    query_embedding=query_embedding, top_k=top_k * 2
                )

            all_chunks.extend(chunks)

        # Deduplicate chunks
        unique_chunks = self._deduplicate_chunks(all_chunks)

        # Reranking
        if use_reranking and len(unique_chunks) > top_k:
            unique_chunks = await self.vector_store.search_with_reranking(
                query_embedding=await self.embedding_service.embed_query(query),
                query_text=query,
                top_k=top_k,
                reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            )

        retrieval_time = time.time() - start_time

        return RetrievalResult(
            chunks=unique_chunks[:top_k],
            retrieval_time=retrieval_time,
            query_variations=query_variations,
            retrieval_method="hybrid" if settings.USE_HYBRID_SEARCH else "semantic",
            reranked=use_reranking,
        )

    async def _generate_query_variations(self, query: str) -> List[str]:
        """Generate multiple query variations."""
        # Use LCEL chain
        result = await self.multi_query_chain.ainvoke({"question": query})
        return result if isinstance(result, list) else []

    def _parse_multi_queries(self, output: str) -> List[str]:
        """Parse multi-query generation output."""
        # Try to extract numbered list
        queries = []
        lines = output.strip().split("\n")

        for line in lines:
            # Remove numbering and bullets
            clean_line = re.sub(r"^[\d\.\-•]\s*", "", line.strip())
            if clean_line and len(clean_line) > 10:  # Reasonable length
                queries.append(clean_line)

        # If parsing failed, split by sentences
        if len(queries) < 2:
            sentences = re.split(r"[.!?]+", output)
            queries = [s.strip() for s in sentences if len(s.strip()) > 20][:3]

        return queries[:3]  # Return at most 3 variations

    async def _generate_hypothetical_document(self, query: str) -> str:
        """Generate hypothetical document for HyDE."""
        # Use LCEL chain
        return await self.hyde_chain.ainvoke({"question": query})

    def _deduplicate_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Deduplicate chunks based on content similarity."""
        unique_chunks = []
        seen_hashes = set()

        for chunk in chunks:
            # Create hash of chunk content (first 100 chars + last 100 chars)
            content_hash = hashlib.md5(
                (chunk.content[:100] + chunk.content[-100:]).encode()
            ).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_chunks.append(chunk)

        return unique_chunks

    async def generate_answer(
        self,
        query: str,
        context_chunks: List[DocumentChunk],
        session_id: str = None,
        use_self_reflection: bool = True,
        use_verification: bool = True,
        **kwargs,
    ) -> GenerationResult:
        """Generate answer with hallucination mitigation."""
        import time

        start_time = time.time()

        # Get conversation history
        history = ""
        if session_id and self.conversation_memory:
            history = self.conversation_memory.get_conversation_for_context(session_id)

        # Format context
        context = self._format_context(context_chunks)

        # Generate initial answer
        initial_answer = await self._generate_initial_answer(
            query=query, context=context, history=history
        )

        # Apply hallucination mitigation techniques
        verification_report = None
        reflection_critique = None
        final_answer = initial_answer

        if use_verification:
            verification_report = await self._verify_answer(
                query=query, answer=initial_answer, context=context
            )

            # If verification found issues, regenerate
            if verification_report.get("issues"):
                final_answer = await self._regenerate_with_feedback(
                    query=query,
                    initial_answer=initial_answer,
                    verification_report=verification_report,
                    context=context,
                    history=history,
                )

        if use_self_reflection:
            reflection_critique = await self._self_reflect(
                query=query, answer=final_answer, context=context
            )

        # Extract citations
        citations = self._extract_citations(final_answer, context_chunks)

        # Calculate confidence
        confidence_score = await self._calculate_confidence(
            query=query, answer=final_answer, context=context
        )

        # Estimate token usage
        token_usage = await self.llm_service.get_token_usage(final_answer)

        generation_time = time.time() - start_time

        # Store in memory if session exists
        if session_id and self.conversation_memory:
            self.conversation_memory.add_message(session_id, "user", query)
            self.conversation_memory.add_message(session_id, "assistant", final_answer)

        return GenerationResult(
            answer=final_answer,
            verification_report=verification_report,
            reflection_critique=reflection_critique,
            confidence_score=confidence_score,
            citations=citations,
            generation_time=generation_time,
            token_usage=token_usage,
        )

    async def _generate_initial_answer(
        self, query: str, context: str, history: str
    ) -> str:
        """Generate initial answer using LCEL chain."""
        try:
            result = await self.rag_chain_lcel.ainvoke(
                {"context": context, "history": history, "question": query}
            )
            return result
        except Exception as e:
            raise GenerationError(
                detail=f"Initial answer generation failed: {str(e)}",
                metadata={"query": query[:100]},
            )

    async def _verify_answer(
        self, query: str, answer: str, context: str
    ) -> Dict[str, Any]:
        """Verify answer against context using chain-of-verification."""
        try:
            messages = [
                {"role": "system", "content": VERIFICATION_PROMPT.messages[0].content},
                {
                    "role": "user",
                    "content": f"Question: {query}\nAnswer: {answer}\nContext: {context}",
                },
            ]

            verification_text = await self.llm_service.generate(messages)

            # Parse verification report
            return self._parse_verification_report(verification_text)

        except Exception as e:
            logger.warning("verification_failed", error=str(e))
            return {"issues": [], "verified_claims": []}

    def _parse_verification_report(self, report_text: str) -> Dict[str, Any]:
        """Parse verification report into structured format."""
        issues = []
        verified_claims = []

        # Simple parsing - in production, use more sophisticated parsing
        lines = report_text.split("\n")

        for line in lines:
            line_lower = line.lower()
            if "unverified" in line_lower or "contradiction" in line_lower:
                issues.append(
                    {"type": "unverified", "claim": line.strip(), "severity": "medium"}
                )
            elif "verified" in line_lower or "supported" in line_lower:
                verified_claims.append(line.strip())

        return {
            "issues": issues,
            "verified_claims": verified_claims,
            "total_claims": len(issues) + len(verified_claims),
        }

    async def _regenerate_with_feedback(
        self,
        query: str,
        initial_answer: str,
        verification_report: Dict[str, Any],
        context: str,
        history: str,
    ) -> str:
        """Regenerate answer incorporating verification feedback."""
        feedback_prompt = f"""The previous answer had some unverified claims. 
Please regenerate the answer ensuring all claims are supported by the context.

CONTEXT:
{context}

PREVIOUS ANSWER:
{initial_answer}

VERIFICATION ISSUES:
{json.dumps(verification_report['issues'], indent=2)}

Generate a corrected answer:"""

        messages = [
            {
                "role": "system",
                "content": "You are a careful assistant that only provides verified information.",
            },
            {"role": "user", "content": feedback_prompt},
        ]

        return await self.llm_service.generate(messages)

    async def _self_reflect(
        self, query: str, answer: str, context: str
    ) -> Dict[str, Any]:
        """Perform self-reflection/critique on the answer."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": SELF_REFLECTION_PROMPT.messages[0].content,
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\nQuestion: {query}\nAnswer: {answer}",
                },
            ]

            reflection_text = await self.llm_service.generate(messages)

            # Parse reflection
            return self._parse_reflection(reflection_text)

        except Exception as e:
            logger.warning("self_reflection_failed", error=str(e))
            return {"critique": "", "suggestions": []}

    def _parse_reflection(self, reflection_text: str) -> Dict[str, Any]:
        """Parse self-reflection into structured format."""
        critique = {}
        suggestions = []

        # Extract sections
        sections = reflection_text.split("\n\n")

        for section in sections:
            if "issue" in section.lower() or "problem" in section.lower():
                critique["issues"] = section.strip()
            elif "suggestion" in section.lower() or "improvement" in section.lower():
                suggestions.append(section.strip())

        return {
            "critique": critique.get("issues", ""),
            "suggestions": suggestions,
            "has_issues": bool(critique.get("issues")),
        }

    async def _calculate_confidence(
        self, query: str, answer: str, context: str
    ) -> float:
        """Calculate confidence score for the answer."""
        try:
            messages = [
                {"role": "system", "content": CONFIDENCE_PROMPT.messages[0].content},
                {
                    "role": "user",
                    "content": f"Context: {context}\nQuestion: {query}\nAnswer: {answer}",
                },
            ]

            confidence_json = await self.llm_service.generate_json(messages)

            if isinstance(confidence_json, dict) and "score" in confidence_json:
                return float(confidence_json["score"]) / 100.0  # Convert to 0-1
            return 0.7  # Default confidence

        except Exception as e:
            logger.warning("confidence_scoring_failed", error=str(e))
            return 0.7  # Default confidence

    def _format_context(self, chunks: List[DocumentChunk]) -> str:
        """Format context chunks for the LLM."""
        context_parts = []

        for i, chunk in enumerate(chunks):
            source_info = f"Source: {chunk.metadata.get('source', 'Unknown')}"
            if chunk.metadata.get("page"):
                source_info += f" (Page {chunk.metadata['page']})"

            content = f"[Document {i + 1} - {source_info}]\n{chunk.content}\n"
            context_parts.append(content)

        return "\n---\n".join(context_parts)

    def _extract_citations(
        self, answer: str, chunks: List[DocumentChunk]
    ) -> List[Dict[str, Any]]:
        """Extract citations from answer."""
        citations = []

        # Simple citation extraction - look for references to document numbers
        for i, chunk in enumerate(chunks):
            doc_ref = f"[Document {i + 1}]"
            if doc_ref in answer:
                citations.append(
                    {
                        "document_index": i,
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.id,
                        "reference": doc_ref,
                        "content_preview": chunk.content[:200] + "...",
                    }
                )

        return citations

    async def run_full_pipeline(
        self, query: str, session_id: str = None, top_k: int = 5, **kwargs
    ) -> Tuple[RetrievalResult, GenerationResult]:
        """Run complete RAG pipeline with all enhancements."""
        # Step 1: Enhanced Retrieval
        retrieval_result = await self.retrieve(
            query=query,
            top_k=top_k,
            use_multi_query=True,
            use_hyde=True,
            use_reranking=True,
        )

        # Step 2: Enhanced Generation with Hallucination Mitigation
        generation_result = await self.generate_answer(
            query=query,
            context_chunks=retrieval_result.chunks,
            session_id=session_id,
            use_self_reflection=True,
            use_verification=True,
        )

        return retrieval_result, generation_result

    async def stream_answer(
        self, query: str, context_chunks: List[DocumentChunk], session_id: str = None
    ):
        """Stream answer token by token."""
        from src.application.chains.prompts import PromptRegistry
        
        # Format context
        context = self._format_context(context_chunks)

        # Get history
        history = ""
        if session_id and self.conversation_memory:
            history = self.conversation_memory.get_conversation_for_context(session_id)

        # Prepare messages for streaming - use the string template directly
        system_content = PromptRegistry.SYSTEM_PROMPT_RAG.format(
            context=context, history=history, question=query
        )
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        # Stream response
        async for token in self.llm_service.stream_generation(messages):
            yield token

