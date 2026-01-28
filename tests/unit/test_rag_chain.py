import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.chains.rag_chain import (AdvancedRAGChain,
                                              GenerationResult,
                                              RetrievalResult)
from src.core.exceptions import GenerationError
from src.core.schemas import DocumentChunk


class TestAdvancedRAGChain:
    """Test suite for AdvancedRAGChain."""

    @pytest.fixture
    def mock_chunks(self):
        """Mock document chunks for testing."""
        return [
            DocumentChunk(
                id="chunk_1",
                content="LangChain should be deployed with Qdrant and Redis.",
                metadata={"source": "doc1.md", "score": 0.85},
                document_id="doc1",
                chunk_index=0,
            ),
            DocumentChunk(
                id="chunk_2",
                content="Use monitoring tools like Prometheus and Grafana.",
                metadata={"source": "doc1.md", "score": 0.78},
                document_id="doc1",
                chunk_index=1,
            ),
            DocumentChunk(
                id="chunk_3",
                content="Implement caching to reduce costs and improve performance.",
                metadata={"source": "doc2.md", "score": 0.92},
                document_id="doc2",
                chunk_index=0,
            ),
        ]

    @pytest.mark.asyncio
    async def test_retrieve_with_multi_query(self, test_rag_chain, mock_vector_store):
        """Test retrieval with multi-query generation."""
        query = "How to optimize LangChain?"

        # Mock multi-query generation
        with patch.object(test_rag_chain, "_generate_query_variations") as mock_multi:
            mock_multi.return_value = [
                "LangChain optimization techniques",
                "Best practices for LangChain deployment",
                "Improving LangChain performance",
            ]

            result = await test_rag_chain.retrieve(
                query=query,
                top_k=2,
                use_multi_query=True,
                use_hyde=False,
                use_reranking=False,
            )

        assert isinstance(result, RetrievalResult)
        assert len(result.chunks) > 0
        assert result.retrieval_method in ["semantic", "hybrid"]
        assert result.retrieval_time > 0
        assert len(result.query_variations) >= 1  # Original query at minimum

    @pytest.mark.asyncio
    async def test_retrieve_with_hyde(self, test_rag_chain, mock_vector_store):
        """Test retrieval with HyDE."""
        query = "What is RAG?"

        # Mock HyDE generation
        with patch.object(
            test_rag_chain, "_generate_hypothetical_document"
        ) as mock_hyde:
            mock_hyde.return_value = "RAG stands for Retrieval-Augmented Generation..."
            mock_hyde.embedding = [0.1] * 384

            result = await test_rag_chain.retrieve(
                query=query, use_hyde=True, use_multi_query=False
            )

        assert mock_hyde.called
        assert isinstance(result, RetrievalResult)

    @pytest.mark.asyncio
    async def test_generate_answer_basic(self, test_rag_chain, mock_chunks):
        """Test basic answer generation."""
        query = "How to deploy LangChain?"

        result = await test_rag_chain.generate_answer(
            query=query,
            context_chunks=mock_chunks,
            session_id="test_session",
            history="",
            use_self_reflection=False,
            use_verification=False,
        )

        assert isinstance(result, GenerationResult)
        assert result.answer
        assert len(result.citations) >= 0
        assert result.generation_time > 0
        assert hasattr(result, 'token_usage')
        assert result.confidence_score is not None

    @pytest.mark.asyncio
    async def test_generate_answer_with_verification(self, test_rag_chain, mock_chunks):
        """Test answer generation with verification."""
        query = "What tools should I use?"

        # Mock verification to return no issues
        with patch.object(test_rag_chain, "_verify_answer") as mock_verify:
            mock_verify.return_value = {
                "issues": [],
                "verified_claims": ["Claim 1", "Claim 2"],
            }

            result = await test_rag_chain.generate_answer(
                query=query, context_chunks=mock_chunks, history="", use_verification=True
            )

        assert mock_verify.called
        assert result.verification_report is not None
        assert result.verification_report["issues"] == []

    @pytest.mark.asyncio
    async def test_generate_answer_with_self_reflection(
        self, test_rag_chain, mock_chunks
    ):
        """Test answer generation with self-reflection."""
        query = "Tell me about monitoring"

        # Mock self-reflection
        with patch.object(test_rag_chain, "_self_reflect") as mock_reflect:
            mock_reflect.return_value = {
                "critique": "Answer is accurate",
                "suggestions": [],
                "has_issues": False,
            }

            result = await test_rag_chain.generate_answer(
                query=query, context_chunks=mock_chunks, history="", use_self_reflection=True
            )

        assert mock_reflect.called
        assert result.reflection_critique is not None
        assert result.reflection_critique["has_issues"] == False

    @pytest.mark.asyncio
    async def test_format_context(self, test_rag_chain, mock_chunks):
        """Test context formatting."""
        context = test_rag_chain._format_context(mock_chunks)

        assert isinstance(context, str)
        assert "Document 1" in context
        assert "Document 2" in context
        assert "Source:" in context
        assert "doc1.md" in context
        assert "doc2.md" in context

        # Check all chunk contents are included
        for chunk in mock_chunks:
            assert chunk.content in context

    @pytest.mark.asyncio
    async def test_extract_citations(self, test_rag_chain, mock_chunks):
        """Test citation extraction from answer."""
        answer = "Use Qdrant for vector storage [Document 1] and Redis for caching [Document 3]."

        citations = test_rag_chain._extract_citations(answer, mock_chunks)

        assert isinstance(citations, list)
        # Both Document 1 and Document 3 are referenced
        # Document 1 = chunk 0 (doc1), Document 3 = chunk 2 (doc2)
        assert len(citations) == 2
        assert citations[0]["document_index"] == 0
        assert citations[0]["document_id"] == "doc1"
        assert citations[1]["document_index"] == 2
        assert citations[1]["document_id"] == "doc2"

    @pytest.mark.asyncio
    async def test_calculate_confidence(self, test_rag_chain, mock_chunks):
        """Test confidence score calculation."""
        query = "Test query"
        answer = "Test answer"
        context = test_rag_chain._format_context(mock_chunks)

        confidence = await test_rag_chain._calculate_confidence(
            query=query, answer=answer, context=context
        )

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    @pytest.mark.asyncio
    async def test_run_full_pipeline(self, test_rag_chain):
        """Test complete RAG pipeline."""
        query = "How to optimize production deployment?"

        retrieval_result, generation_result = await test_rag_chain.run_full_pipeline(
            query=query, top_k=3
        )

        assert isinstance(retrieval_result, RetrievalResult)
        assert isinstance(generation_result, GenerationResult)
        assert generation_result.answer
        assert len(retrieval_result.chunks) > 0

    @pytest.mark.asyncio
    async def test_stream_answer(self, test_rag_chain, mock_chunks):
        """Test streaming answer generation."""
        query = "Streaming test query"

        tokens = []
        async for token in test_rag_chain.stream_answer(
            query=query, context_chunks=mock_chunks[:1]
        ):
            tokens.append(token)

        assert len(tokens) > 0
        # Join tokens to form complete answer
        answer = "".join(tokens)
        assert len(answer) > 0

    def test_deduplicate_chunks(self, test_rag_chain):
        """Test chunk deduplication."""
        chunks = [
            DocumentChunk(
                id="1",
                content="This is a duplicate chunk",
                metadata={},
                document_id="doc1",
                chunk_index=0,
            ),
            DocumentChunk(
                id="2",
                content="This is a duplicate chunk",  # Same content
                metadata={},
                document_id="doc2",
                chunk_index=0,
            ),
            DocumentChunk(
                id="3",
                content="This is a different chunk",
                metadata={},
                document_id="doc1",
                chunk_index=1,
            ),
        ]

        deduplicated = test_rag_chain._deduplicate_chunks(chunks)

        # Should have 2 unique chunks
        assert len(deduplicated) == 2
        # Check that unique chunks are preserved
        contents = [chunk.content for chunk in deduplicated]
        assert "This is a duplicate chunk" in contents
        assert "This is a different chunk" in contents

    @pytest.mark.asyncio
    async def test_parse_multi_queries(self, test_rag_chain):
        """Test parsing of multi-query generation output."""
        # Test with numbered list
        output = """1. What are the best practices for LangChain?
        2. How to optimize LangChain performance?
        3. LangChain deployment strategies"""

        queries = test_rag_chain._parse_multi_queries(output)

        assert len(queries) == 3
        assert "best practices for LangChain" in queries[0]
        assert "optimize LangChain performance" in queries[1]
        assert "deployment strategies" in queries[2]

        # Test with bullet points
        output2 = """- First query about RAG
        - Second query about vector databases
        - Third query about embeddings"""

        queries2 = test_rag_chain._parse_multi_queries(output2)
        assert len(queries2) == 3

        # Test with plain sentences
        output3 = """This is the first query. This is second? And third!"""
        queries3 = test_rag_chain._parse_multi_queries(output3)
        assert len(queries3) >= 1  # At least one query should be parsed

    @pytest.mark.asyncio
    async def test_generation_error_handling(self, test_rag_chain, mock_chunks):
        """Test error handling in answer generation."""
        # Mock rag_chain_lcel to raise exception
        test_rag_chain.rag_chain_lcel.ainvoke.side_effect = Exception("LLM error")

        with pytest.raises(GenerationError):
            await test_rag_chain._generate_initial_answer(
                query="test", context="context", history=""
            )
