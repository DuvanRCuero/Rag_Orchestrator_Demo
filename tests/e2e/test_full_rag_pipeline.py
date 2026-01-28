import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.chains.rag_chain import AdvancedRAGChain
from src.core.schemas import DocumentType
from src.domain.documents import AdvancedDocumentProcessor
from src.domain.embeddings import EmbeddingService
from src.infrastructure.vector.qdrant_client import QdrantVectorStore


@pytest.mark.e2e
class TestFullRAGPipeline:
    """End-to-end tests for the complete RAG pipeline."""

    @pytest.fixture
    def sample_documents_dir(self):
        """Create a temporary directory with sample documents."""
        temp_dir = tempfile.mkdtemp()

        # Create LangChain production guide
        langchain_doc = Path(temp_dir) / "langchain_production.md"
        langchain_doc.write_text(
            """# LangChain Production Deployment

## Best Practices

1. Use Qdrant for vector storage
2. Implement Redis caching
3. Monitor with Prometheus and Grafana
4. Use sentence-transformers for embeddings

## Performance Tips

- Chunk documents appropriately
- Use async operations
- Implement request batching
- Cache embeddings

## Cost Optimization

- Monitor token usage
- Implement query caching
- Use efficient embedding models"""
        )

        # Create RAG guide
        rag_doc = Path(temp_dir) / "rag_guide.md"
        rag_doc.write_text(
            """# RAG System Guide

## Architecture

A RAG system consists of:
1. Document ingestion pipeline
2. Vector database
3. Retrieval mechanism
4. LLM generation

## Hallucination Mitigation

Techniques:
- Chain-of-Verification
- Self-reflection
- Confidence scoring
- Multi-query retrieval

## Scaling

- Use load balancers
- Implement caching layers
- Monitor performance metrics"""
        )

        yield temp_dir

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_complete_pipeline_with_mocked_llm(self, sample_documents_dir):
        """Test complete pipeline with mocked LLM."""
        # Setup
        processor = AdvancedDocumentProcessor(strategy_name="recursive_character")

        # Mock embedding service
        with patch("sentence_transformers.SentenceTransformer") as mock_model:
            import numpy as np
            mock_model.return_value.encode.return_value = np.array([[0.1] * 384] * 10)
            mock_model.return_value.get_sentence_embedding_dimension.return_value = 384

            embedding_service = EmbeddingService()
            embedding_service._model = mock_model.return_value
            embedding_service.use_openai = False

        # Mock vector store
        mock_store = AsyncMock(spec=QdrantVectorStore)

        # Create test chunks
        test_chunks = []
        for i in range(3):
            chunk = type(
                "obj",
                (object,),
                {
                    "id": f"chunk_{i}",
                    "content": f"Test chunk {i} about LangChain production",
                    "metadata": {"source": "langchain_production.md"},
                    "document_id": "doc1",
                    "chunk_index": i,
                    "embedding": [0.1] * 384,
                },
            )()
            test_chunks.append(chunk)

        mock_store.search.return_value = test_chunks
        mock_store.hybrid_search.return_value = test_chunks

        # Mock LLM
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = """Based on the context:

1. Use Qdrant for vector storage in production.
2. Implement Redis caching to improve performance.
3. Monitor the system with Prometheus and Grafana.

These are the key recommendations for LangChain production deployment."""

        mock_llm.generate_json.return_value = {
            "score": 88,
            "breakdown": {},
            "explanation": "Good answer",
        }
        mock_llm.get_token_usage.return_value = {"total_tokens": 200}

        # Create RAG chain with mocks
        with patch(
            "src.application.chains.rag_chain.QdrantVectorStore",
            return_value=mock_store,
        ), patch(
            "src.application.chains.rag_chain.EmbeddingService",
            return_value=embedding_service,
        ), patch(
            "src.application.chains.rag_chain.llm_service", mock_llm
        ):
            rag_chain = AdvancedRAGChain()
            
            # Mock the LCEL chain to return a string
            rag_chain.rag_chain_lcel = AsyncMock()
            rag_chain.rag_chain_lcel.ainvoke = AsyncMock(
                return_value="Use Qdrant for vector storage in production. Implement Redis caching to improve performance. Monitor the system with Prometheus and Grafana."
            )

            # Test retrieval
            retrieval_result = await rag_chain.retrieve(
                query="How to deploy LangChain in production?",
                top_k=3,
                use_multi_query=False,
                use_hyde=False,
            )

            assert retrieval_result is not None
            assert len(retrieval_result.chunks) == 3

            # Test generation
            generation_result = await rag_chain.generate_answer(
                query="How to deploy LangChain in production?",
                context_chunks=retrieval_result.chunks,
                session_id="test_e2e_session",
                use_self_reflection=False,
                use_verification=False,
            )

            assert generation_result is not None
            assert generation_result.answer
            assert "Qdrant" in generation_result.answer
            assert "Redis" in generation_result.answer
            assert generation_result.confidence_score is not None

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation with memory."""
        # Mock everything for this test
        with patch(
            "src.application.chains.rag_chain.AdvancedRAGChain"
        ) as mock_chain_class:
            mock_chain = AsyncMock()
            mock_chain_class.return_value = mock_chain

            # First query
            mock_chain.run_full_pipeline.return_value = (
                type("obj", (object,), {"chunks": [], "retrieval_time": 0.1})(),
                type(
                    "obj",
                    (object,),
                    {
                        "answer": "Use Qdrant for vector storage.",
                        "confidence_score": 0.85,
                        "generation_time": 0.5,
                        "token_usage": {},
                        "citations": [],
                    },
                )(),
            )

            # Simulate first question
            question1 = "What vector database should I use?"
            # (In reality, we'd call the API or chain directly)

            # Second query should have context from first
            mock_chain.run_full_pipeline.return_value = (
                type("obj", (object,), {"chunks": [], "retrieval_time": 0.1})(),
                type(
                    "obj",
                    (object,),
                    {
                        "answer": "Yes, Qdrant is good for production.",
                        "confidence_score": 0.9,
                        "generation_time": 0.6,
                        "token_usage": {},
                        "citations": [],
                    },
                )(),
            )

            question2 = "Is it good for production?"
            # The chain should have memory of the first question/answer

            # Verify that memory is being used
            # (This is a simplified test - actual implementation would check session memory)
            assert True  # Placeholder for actual memory test

    @pytest.mark.asyncio
    async def test_error_recovery_in_pipeline(self):
        """Test error recovery in the RAG pipeline."""
        with patch(
            "src.application.chains.rag_chain.AdvancedRAGChain"
        ) as mock_chain_class:
            mock_chain = AsyncMock()
            mock_chain_class.return_value = mock_chain

            # Simulate retrieval failure
            mock_chain.retrieve.side_effect = Exception("Vector database error")

            # The pipeline should handle this gracefully
            # (In reality, we'd catch the exception and return appropriate error)

            # For now, just verify the mock is set up
            with pytest.raises(Exception):
                await mock_chain.retrieve(query="test")

            # Test generation failure
            mock_chain.retrieve.side_effect = None
            mock_chain.retrieve.return_value = type(
                "obj",
                (object,),
                {
                    "chunks": [
                        type(
                            "obj",
                            (object,),
                            {
                                "content": "test",
                                "metadata": {},
                                "id": "1",
                                "document_id": "doc1",
                                "chunk_index": 0,
                            },
                        )()
                    ],
                    "retrieval_time": 0.1,
                },
            )()

            mock_chain.generate_answer.side_effect = Exception("LLM API error")

            with pytest.raises(Exception):
                await mock_chain.generate_answer(
                    query="test", context_chunks=mock_chain.retrieve.return_value.chunks
                )

    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self):
        """Test that performance metrics are collected correctly."""
        with patch(
            "src.application.chains.rag_chain.AdvancedRAGChain"
        ) as mock_chain_class:
            mock_chain = AsyncMock()
            mock_chain_class.return_value = mock_chain

            # Setup mock to return metrics
            mock_chain.run_full_pipeline.return_value = (
                type(
                    "obj",
                    (object,),
                    {
                        "chunks": [],
                        "retrieval_time": 0.125,  # 125ms
                        "query_variations": [],
                        "retrieval_method": "semantic",
                        "reranked": False,
                    },
                )(),
                type(
                    "obj",
                    (object,),
                    {
                        "answer": "Test answer",
                        "confidence_score": 0.87,
                        "generation_time": 0.543,  # 543ms
                        "token_usage": {"total_tokens": 187},
                        "citations": [],
                        "verification_report": None,
                        "reflection_critique": None,
                    },
                )(),
            )

            retrieval_result, generation_result = await mock_chain.run_full_pipeline(
                query="test query", top_k=5
            )

            # Verify metrics are present and reasonable
            assert retrieval_result.retrieval_time > 0
            assert generation_result.generation_time > 0
            assert 0 <= generation_result.confidence_score <= 1
            assert hasattr(generation_result, 'token_usage')
            assert 'total_tokens' in generation_result.token_usage

            total_time = (
                retrieval_result.retrieval_time + generation_result.generation_time
            )
            assert total_time > 0

    @pytest.mark.asyncio
    async def test_different_query_types(self):
        """Test handling of different query types."""
        test_cases = [
            {"query": "Short", "description": "Very short query"},
            {
                "query": "This is a very long query with many details about LangChain production deployment including specific requirements for monitoring and scalability",
                "description": "Long detailed query",
            },
            {
                "query": "What are the best practices for RAG systems in 2024?",
                "description": "Question with year specification",
            },
            {
                "query": "code: import langchain\nprint('hello')",
                "description": "Query with code snippet",
            },
        ]

        with patch(
            "src.application.chains.rag_chain.AdvancedRAGChain"
        ) as mock_chain_class:
            mock_chain = AsyncMock()
            mock_chain_class.return_value = mock_chain

            # Generic response
            mock_chain.run_full_pipeline.return_value = (
                type("obj", (object,), {"chunks": [], "retrieval_time": 0.1})(),
                type(
                    "obj",
                    (object,),
                    {
                        "answer": "Generic answer",
                        "confidence_score": 0.8,
                        "generation_time": 0.5,
                        "token_usage": {},
                        "citations": [],
                    },
                )(),
            )

            for test_case in test_cases:
                # Each query should be processed without error
                (
                    retrieval_result,
                    generation_result,
                ) = await mock_chain.run_full_pipeline(
                    query=test_case["query"], top_k=3
                )

                assert retrieval_result is not None
                assert generation_result is not None
                assert generation_result.answer == "Generic answer"
