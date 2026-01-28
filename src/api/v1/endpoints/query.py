import asyncio
import json
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from src.api.v1.dependencies import get_rag_chain, get_session_id
from src.application.chains.rag_chain import AdvancedRAGChain
from src.core.config import settings
from src.core.schemas import QueryRequest, QueryResponse

router = APIRouter()


@router.post(
    "/ask",
    response_model=QueryResponse,
    summary="Ask a question",
    description="Ask a question and get an answer with citations and confidence score.",
)
async def ask_question(
    request: QueryRequest,
    session_id: Optional[str] = None,
    rag_chain: AdvancedRAGChain = Depends(get_rag_chain),
):
    """Ask a question to the RAG system."""
    try:
        # Get or generate session ID
        actual_session_id = get_session_id(session_id)

        # Run full RAG pipeline
        retrieval_result, generation_result = await rag_chain.run_full_pipeline(
            query=request.query,
            session_id=actual_session_id,
            top_k=request.top_k or 5,
            temperature=request.temperature or 0.1,
        )

        # Calculate total time
        total_time = retrieval_result.retrieval_time + generation_result.generation_time

        return QueryResponse(
            answer=generation_result.answer,
            sources=retrieval_result.chunks,
            session_id=actual_session_id,
            query_time=total_time,
            token_usage=generation_result.token_usage,
            confidence=generation_result.confidence_score,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}",
        )


@router.post(
    "/stream",
    summary="Stream answer",
    description="Stream the answer token by token for real-time experience.",
)
async def stream_answer(
    request: QueryRequest,
    session_id: Optional[str] = None,
    rag_chain: AdvancedRAGChain = Depends(get_rag_chain),
):
    """Stream answer token by token."""
    try:
        # Get session ID
        actual_session_id = get_session_id(session_id)

        # Perform retrieval first
        retrieval_result = await rag_chain.retrieve(
            query=request.query, top_k=request.top_k or 5
        )

        # Create streaming response
        async def generate():
            """Generate streaming response."""
            try:
                # Start with retrieval metadata
                retrieval_data = {
                    "type": "retrieval_complete",
                    "chunks_retrieved": len(retrieval_result.chunks),
                    "retrieval_time": retrieval_result.retrieval_time,
                }
                yield f"data: {json.dumps(retrieval_data)}\n\n"

                # Stream answer
                async for token in rag_chain.stream_answer(
                    query=request.query,
                    context_chunks=retrieval_result.chunks,
                    session_id=actual_session_id,
                ):
                    token_data = {
                        "type": "token",
                        "token": token,
                        "session_id": actual_session_id,
                    }
                    yield f"data: {json.dumps(token_data)}\n\n"

                # End of stream
                complete_data = {"type": "complete", "session_id": actual_session_id}
                yield f"data: {json.dumps(complete_data)}\n\n"

            except Exception as e:
                error_data = {"type": "error", "error": str(e)}
                yield f"data: {json.dumps(error_data)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable buffering for nginx
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stream answer: {str(e)}",
        )


@router.post(
    "/retrieve-only",
    summary="Retrieve only",
    description="Retrieve relevant document chunks without generating an answer.",
)
async def retrieve_only(
    query: str = Query(..., description="The query to search for"),
    top_k: int = Query(5, description="Number of chunks to retrieve"),
    use_multi_query: bool = Query(True, description="Use multi-query expansion"),
    use_reranking: bool = Query(True, description="Use cross-encoder reranking"),
    rag_chain: AdvancedRAGChain = Depends(get_rag_chain),
):
    """Retrieve document chunks without generating an answer."""
    try:
        retrieval_result = await rag_chain.retrieve(
            query=query,
            top_k=top_k,
            use_multi_query=use_multi_query,
            use_reranking=use_reranking,
        )

        return {
            "query": query,
            "chunks": [
                {
                    "id": chunk.id,
                    "content": chunk.content[:500] + "..."
                    if len(chunk.content) > 500
                    else chunk.content,
                    "metadata": chunk.metadata,
                    "document_id": chunk.document_id,
                    "score": chunk.metadata.get("search_score", 0.0),
                }
                for chunk in retrieval_result.chunks
            ],
            "retrieval_metadata": {
                "method": retrieval_result.retrieval_method,
                "time": retrieval_result.retrieval_time,
                "query_variations": retrieval_result.query_variations,
                "reranked": retrieval_result.reranked,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}",
        )


@router.post(
    "/verify",
    summary="Verify answer",
    description="Verify an answer against the context using chain-of-verification.",
)
async def verify_answer(
    question: str = Query(..., description="The original question"),
    answer: str = Query(..., description="The answer to verify"),
    context: str = Query(..., description="The context to verify against"),
    rag_chain: AdvancedRAGChain = Depends(get_rag_chain),
):
    """Verify an answer using chain-of-verification."""
    try:
        # This would use the verification chain directly
        # For now, return a mock response
        return {
            "question": question,
            "answer": answer,
            "verification": {
                "verified_claims": 3,
                "unverified_claims": 1,
                "contradictions": 0,
                "confidence": 0.75,
                "report": "Answer is mostly verified with one unsubstantiated claim.",
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {str(e)}",
        )


@router.get(
    "/suggestions",
    summary="Get query suggestions",
    description="Get suggested queries based on conversation history or popular queries.",
)
async def get_suggestions(
    session_id: Optional[str] = None,
    count: int = Query(5, description="Number of suggestions to return"),
):
    """Get query suggestions."""
    # This would implement query suggestion logic
    # For now, return some example suggestions for LangChain
    suggestions = [
        "How to optimize LangChain for production?",
        "Best practices for chunking documents?",
        "How to reduce hallucination in RAG systems?",
        "What are the best embedding models for retrieval?",
        "How to implement conversation memory in LangChain?",
    ]

    return {"session_id": session_id, "suggestions": suggestions[:count]}
