import asyncio
import base64
import os
import tempfile
from typing import List, Optional

from fastapi import (APIRouter, BackgroundTasks, File, Form, HTTPException,
                     UploadFile, status)
from pydantic import BaseModel

from src.api.v1.dependencies import get_vector_store, get_embedding_service
from src.core.exceptions import IngestionError
from src.core.schemas import DocumentType, IngestionRequest, IngestionResponse
from src.domain.documents import AdvancedDocumentProcessor

router = APIRouter()


class DocumentUpload(BaseModel):
    """Request model for document upload."""

    files: List[UploadFile]
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200
    document_type: Optional[DocumentType] = DocumentType.TXT
    metadata: Optional[dict] = None


@router.post(
    "/upload",
    response_model=IngestionResponse,
    summary="Upload and process documents",
    description="Upload documents, split into chunks, generate embeddings, and store in vector database.",
)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    document_type: DocumentType = Form(DocumentType.TXT),
    metadata: Optional[str] = Form(None),
):
    """Upload multiple documents for processing."""
    try:
        # Parse metadata if provided
        metadata_dict = {}
        if metadata:
            import json

            metadata_dict = json.loads(metadata)

        # Process files
        document_ids = []
        total_chunks = 0

        for upload_file in files:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=upload_file.filename
            ) as tmp_file:
                content = await upload_file.read()
                tmp_file.write(content)
                tmp_path = tmp_file.name

            try:
                # Process document
                processor = AdvancedDocumentProcessor()
                embedding_service = get_embedding_service()
                vector_store = get_vector_store()

                # Load document
                raw_docs = processor.load_document(tmp_path, document_type)

                for doc_idx, raw_doc in enumerate(raw_docs):
                    doc_metadata = {
                        **metadata_dict,
                        **raw_doc["metadata"],
                        "filename": upload_file.filename,
                        "document_type": document_type.value,
                        "source": raw_doc["source"],
                    }

                    # Create document ID
                    import hashlib

                    doc_id = hashlib.md5(
                        f"{upload_file.filename}_{doc_idx}".encode()
                    ).hexdigest()[:16]
                    doc_metadata["document_id"] = doc_id

                    # Create chunks
                    chunks = processor.create_intelligent_chunks(
                        text=raw_doc["content"], metadata=doc_metadata
                    )

                    # Generate embeddings
                    texts = [chunk.content for chunk in chunks]
                    embeddings = await embedding_service.embed_texts(texts)

                    # Add embeddings to chunks
                    for chunk, embedding in zip(chunks, embeddings):
                        chunk.embedding = embedding

                    # Store in vector database
                    await vector_store.upsert_chunks(chunks)

                    document_ids.append(doc_id)
                    total_chunks += len(chunks)

                # Clean up temp file
                os.unlink(tmp_path)

            except Exception as e:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise IngestionError(
                    detail=f"Failed to process {upload_file.filename}: {str(e)}",
                    metadata={"filename": upload_file.filename},
                )

        return IngestionResponse(
            document_ids=document_ids,
            total_chunks=total_chunks,
            processing_time=0.0,  # Would track actual time
            status="success",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Document upload failed: {str(e)}",
        )


@router.post(
    "/batch",
    response_model=IngestionResponse,
    summary="Batch document ingestion",
    description="Ingest multiple documents from base64 encoded strings or URLs.",
)
async def batch_ingest(request: IngestionRequest, background_tasks: BackgroundTasks):
    """Batch ingest documents from various sources."""
    # This would implement batch processing
    # For now, return a placeholder response
    return IngestionResponse(
        document_ids=["batch_001", "batch_002"],
        total_chunks=50,
        processing_time=2.5,
        status="success",
    )


@router.delete(
    "/document/{document_id}",
    summary="Delete a document",
    description="Remove all chunks of a specific document from the vector store.",
)
async def delete_document(document_id: str):
    """Delete a document and its chunks."""
    try:
        vector_store = get_vector_store()
        await vector_store.delete_by_document_id(document_id)

        return {
            "status": "success",
            "message": f"Document {document_id} deleted successfully",
            "document_id": document_id,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Failed to delete document: {str(e)}",
        )


@router.get(
    "/status",
    summary="Get ingestion status",
    description="Get statistics about ingested documents and vector store.",
)
async def get_ingestion_status():
    """Get ingestion statistics."""
    try:
        vector_store = get_vector_store()
        stats = await vector_store.get_collection_stats()

        return {
            "status": "healthy",
            "vector_store": {
                "type": "qdrant",
                "vectors_count": stats.get("vectors_count", 0),
                "collection": stats.get("config", {}),
            },
            "documents": {
                "total_documents": "unknown",  # Would track in metadata store
                "total_chunks": stats.get("vectors_count", 0),
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to get ingestion status: {str(e)}",
        )
