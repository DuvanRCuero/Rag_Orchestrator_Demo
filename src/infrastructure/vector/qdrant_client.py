import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
import requests

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.core.config import settings
from src.core.exceptions import RetrievalError
from src.core.schemas import DocumentChunk
from src.domain.vector_store import VectorStore


class QdrantVectorStore(VectorStore):
    """Production-grade Qdrant implementation with hybrid search."""

    def __init__(self):
        self._client = None
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self._initialized = False
        # For direct HTTP calls to v1.7.4
        self.http_url = "http://localhost:6333"

    @property
    def client(self):
        """Lazy initialization of Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(
                host="localhost",
                port=6333,
                timeout=60,
                prefer_grpc=False,
                https=False,
            )
            if not self._initialized:
                self._ensure_collection()
                self._initialized = True
        return self._client

    def _ensure_collection(self):
        """Create collection with optimized configuration."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSION,
                        distance=Distance.COSINE,
                    ),
                )
                print(f"✅ Created Qdrant collection: {self.collection_name}")

                # Create payload indexes
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="document_id",
                        field_schema="keyword",
                    )
                except Exception as idx_error:
                    print(f"Warning: Could not create payload indexes: {idx_error}")
            else:
                print(f"✅ Using existing Qdrant collection: {self.collection_name}")

        except Exception as e:
            print(f"Warning: Could not initialize Qdrant collection: {e}")

    async def create_collection(
            self, collection_name: str, vector_size: int
    ) -> bool:
        """Create a new collection."""
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size, distance=Distance.COSINE
                ),
            )
            return True
        except Exception as e:
            print(f"Error creating collection: {e}")
            return False

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception:
            return False

    async def upsert_chunks(
            self, chunks: List[DocumentChunk], embeddings: List[List[float]] = None
    ) -> bool:
        """Upsert chunks with metadata."""
        try:
            points = []

            for i, chunk in enumerate(chunks):
                if embeddings and i < len(embeddings):
                    embedding = embeddings[i]
                elif chunk.embedding:
                    embedding = chunk.embedding
                else:
                    raise ValueError(f"Chunk {chunk.id} has no embedding")

                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "id": chunk.id,
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "created_at": chunk.created_at.isoformat(),
                        "updated_at": datetime.utcnow().isoformat(),
                    },
                )
                points.append(point)

            # Batch upsert
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i: i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
            return True
        except Exception as e:
            print(f"Error upserting chunks: {e}")
            return False

    async def search(
            self,
            query_embedding: List[float],
            top_k: int = 5,
            score_threshold: float = 0.0,
            filters: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Semantic search using direct HTTP API for v1.7.4 compatibility."""
        try:
            # Build the search request payload for Qdrant REST API
            search_payload = {
                "vector": query_embedding,
                "limit": top_k * 2,
                "with_payload": True,
                "with_vector": False,
                "score_threshold": score_threshold or settings.RETRIEVAL_SCORE_THRESHOLD,
            }

            # Add filter if provided
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    filter_conditions.append({
                        "key": f"metadata.{key}",
                        "match": {"value": value}
                    })
                search_payload["filter"] = {"must": filter_conditions}

            # Make direct HTTP POST request to Qdrant REST API
            response = requests.post(
                f"{self.http_url}/collections/{self.collection_name}/points/search",
                json=search_payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            search_results = result.get("result", [])

            # Convert to DocumentChunk objects
            chunks = []
            for item in search_results:
                try:
                    payload = item["payload"]
                    confidence = float(item.get("score", 0.0))

                    chunk = DocumentChunk(
                        id=payload["id"],
                        content=payload["content"],
                        metadata=payload["metadata"],
                        document_id=payload["document_id"],
                        chunk_index=payload["chunk_index"],
                        created_at=datetime.fromisoformat(
                            payload["created_at"].replace("Z", "+00:00")
                        ),
                    )

                    chunk.metadata["search_score"] = confidence
                    chunk.metadata["search_rank"] = len(chunks) + 1
                    chunks.append(chunk)

                except Exception as chunk_error:
                    print(f"Warning: Could not process search result: {chunk_error}")
                    continue

            print(f"✅ Found {len(chunks)} chunks")
            return chunks[:top_k]

        except requests.exceptions.RequestException as e:
            print(f"❌ HTTP request failed: {e}")
            raise RetrievalError(
                detail=f"Vector search HTTP request failed: {str(e)}",
                metadata={
                    "top_k": top_k,
                    "filters": filters,
                    "collection": self.collection_name,
                },
            )
        except Exception as e:
            print(f"❌ Search failed: {e}")
            raise RetrievalError(
                detail=f"Vector search failed: {str(e)}",
                metadata={
                    "top_k": top_k,
                    "filters": filters,
                    "collection": self.collection_name,
                },
            )

    async def hybrid_search(
            self,
            query_embedding: List[float],
            query_text: str,
            top_k: int = 5,
            score_threshold: float = 0.0,
            filters: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """Hybrid search - falls back to semantic for v1.7.4."""
        print(f"Hybrid search requested, using semantic search for Qdrant v1.7.4")

        return await self.search(
            query_embedding=query_embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=filters,
        )

    def _combine_results(self, bm25_results, semantic_results, bm25_weight: float, semantic_weight: float):
        """Combine and rerank results."""
        scored_results = {}

        for i, result in enumerate(bm25_results):
            point_id = result.id
            bm25_score = 1.0 / (i + 1)
            if point_id not in scored_results:
                scored_results[point_id] = {
                    "payload": result.payload,
                    "bm25_score": bm25_score,
                    "semantic_score": 0.0,
                }

        for i, result in enumerate(semantic_results):
            point_id = result.id
            semantic_score = float(result.score)
            if point_id in scored_results:
                scored_results[point_id]["semantic_score"] = semantic_score
            else:
                scored_results[point_id] = {
                    "payload": result.payload,
                    "bm25_score": 0.0,
                    "semantic_score": semantic_score,
                }

        combined_results = []
        for point_id, scores in scored_results.items():
            combined_score = (
                    scores["bm25_score"] * bm25_weight
                    + scores["semantic_score"] * semantic_weight
            )
            combined_results.append(
                {"id": point_id, "score": combined_score, "payload": scores["payload"]}
            )

        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return combined_results

    async def delete_by_document_id(self, document_id: str) -> bool:
        """Delete all chunks of a document."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
            )
            return True
        except Exception as e:
            print(f"Error deleting document {document_id}: {e}")
            return False

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "vectors_count": info.vectors_count,
                "segments_count": info.segments_count,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": str(info.config.params.vectors.distance),
                },
                "status": info.status,
            }
        except Exception as e:
            return {"vectors_count": 0, "error": str(e)}