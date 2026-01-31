"""Tests for vector store interface segregation."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.domain.interfaces.vector_store import (
    VectorStore,
    VectorReader,
    VectorWriter,
    VectorAdmin,
)
from src.core.schemas import DocumentChunk


class TestVectorStoreInterfaces:
    """Test vector store interface segregation."""

    def test_vector_reader_interface_exists(self):
        """Test that VectorReader interface exists."""
        assert VectorReader is not None
        # Check that it has the expected methods
        assert hasattr(VectorReader, 'search')
        assert hasattr(VectorReader, 'hybrid_search')

    def test_vector_writer_interface_exists(self):
        """Test that VectorWriter interface exists."""
        assert VectorWriter is not None
        # Check that it has the expected methods
        assert hasattr(VectorWriter, 'upsert_chunks')
        assert hasattr(VectorWriter, 'delete_by_document_id')

    def test_vector_admin_interface_exists(self):
        """Test that VectorAdmin interface exists."""
        assert VectorAdmin is not None
        # Check that it has the expected methods
        assert hasattr(VectorAdmin, 'create_collection')
        assert hasattr(VectorAdmin, 'collection_exists')
        assert hasattr(VectorAdmin, 'get_collection_stats')

    def test_vector_store_inherits_all_interfaces(self):
        """Test that VectorStore inherits from all three interfaces."""
        assert issubclass(VectorStore, VectorReader)
        assert issubclass(VectorStore, VectorWriter)
        assert issubclass(VectorStore, VectorAdmin)

    def test_qdrant_implements_vector_store(self):
        """Test that QdrantVectorStore implements VectorStore."""
        from src.infrastructure.vector.qdrant_client import QdrantVectorStore
        assert issubclass(QdrantVectorStore, VectorStore)
        assert issubclass(QdrantVectorStore, VectorReader)
        assert issubclass(QdrantVectorStore, VectorWriter)
        assert issubclass(QdrantVectorStore, VectorAdmin)

    def test_retrieval_service_uses_vector_reader(self):
        """Test that RetrievalService uses VectorReader interface."""
        from src.application.services.retrieval_service import RetrievalService
        import inspect
        
        sig = inspect.signature(RetrievalService.__init__)
        params = sig.parameters
        
        # Check that vector_store parameter is typed as VectorReader
        assert 'vector_store' in params
        assert params['vector_store'].annotation == VectorReader

    def test_mock_vector_reader_implementation(self):
        """Test that we can create a mock VectorReader implementation."""
        
        class MockVectorReader(VectorReader):
            async def search(self, query_embedding, top_k=5, score_threshold=0.0, filters=None):
                return []
            
            async def hybrid_search(self, query_embedding, query_text, top_k=5, score_threshold=0.0, filters=None):
                return []
        
        mock_reader = MockVectorReader()
        assert isinstance(mock_reader, VectorReader)

    def test_mock_vector_writer_implementation(self):
        """Test that we can create a mock VectorWriter implementation."""
        
        class MockVectorWriter(VectorWriter):
            async def upsert_chunks(self, chunks, embeddings=None):
                return True
            
            async def delete_by_document_id(self, document_id):
                return True
        
        mock_writer = MockVectorWriter()
        assert isinstance(mock_writer, VectorWriter)

    def test_mock_vector_admin_implementation(self):
        """Test that we can create a mock VectorAdmin implementation."""
        
        class MockVectorAdmin(VectorAdmin):
            async def create_collection(self, collection_name, vector_size):
                return True
            
            async def collection_exists(self, collection_name):
                return True
            
            async def get_collection_stats(self):
                return {"vectors_count": 0}
        
        mock_admin = MockVectorAdmin()
        assert isinstance(mock_admin, VectorAdmin)

    def test_segregated_interfaces_enable_focused_implementations(self):
        """Test that segregated interfaces allow focused implementations."""
        
        # Read-only service that only needs VectorReader
        class ReadOnlyService:
            def __init__(self, vector_store: VectorReader):
                self.vector_store = vector_store
        
        # Write-only service that only needs VectorWriter
        class WriteOnlyService:
            def __init__(self, vector_store: VectorWriter):
                self.vector_store = vector_store
        
        # Admin-only service that only needs VectorAdmin
        class AdminOnlyService:
            def __init__(self, vector_store: VectorAdmin):
                self.vector_store = vector_store
        
        # Create a full implementation
        class FullVectorStore(VectorStore):
            async def search(self, query_embedding, top_k=5, score_threshold=0.0, filters=None):
                return []
            
            async def hybrid_search(self, query_embedding, query_text, top_k=5, score_threshold=0.0, filters=None):
                return []
            
            async def upsert_chunks(self, chunks, embeddings=None):
                return True
            
            async def delete_by_document_id(self, document_id):
                return True
            
            async def create_collection(self, collection_name, vector_size):
                return True
            
            async def collection_exists(self, collection_name):
                return True
            
            async def get_collection_stats(self):
                return {"vectors_count": 0}
        
        full_store = FullVectorStore()
        
        # Full store can be used by all services
        read_service = ReadOnlyService(full_store)
        write_service = WriteOnlyService(full_store)
        admin_service = AdminOnlyService(full_store)
        
        assert isinstance(read_service.vector_store, VectorReader)
        assert isinstance(write_service.vector_store, VectorWriter)
        assert isinstance(admin_service.vector_store, VectorAdmin)


class TestInterfaceSegregationPrinciple:
    """Test that Interface Segregation Principle is properly applied."""

    def test_vector_reader_only_has_read_methods(self):
        """Test that VectorReader only exposes read methods."""
        reader_methods = [method for method in dir(VectorReader) 
                         if not method.startswith('_') and callable(getattr(VectorReader, method, None))]
        
        # VectorReader should only have read methods
        assert 'search' in reader_methods
        assert 'hybrid_search' in reader_methods
        
        # VectorReader should not have write or admin methods
        assert 'upsert_chunks' not in reader_methods
        assert 'delete_by_document_id' not in reader_methods
        assert 'create_collection' not in reader_methods

    def test_vector_writer_only_has_write_methods(self):
        """Test that VectorWriter only exposes write methods."""
        writer_methods = [method for method in dir(VectorWriter) 
                         if not method.startswith('_') and callable(getattr(VectorWriter, method, None))]
        
        # VectorWriter should only have write methods
        assert 'upsert_chunks' in writer_methods
        assert 'delete_by_document_id' in writer_methods
        
        # VectorWriter should not have read or admin methods
        assert 'search' not in writer_methods
        assert 'hybrid_search' not in writer_methods
        assert 'create_collection' not in writer_methods

    def test_vector_admin_only_has_admin_methods(self):
        """Test that VectorAdmin only exposes admin methods."""
        admin_methods = [method for method in dir(VectorAdmin) 
                        if not method.startswith('_') and callable(getattr(VectorAdmin, method, None))]
        
        # VectorAdmin should only have admin methods
        assert 'create_collection' in admin_methods
        assert 'collection_exists' in admin_methods
        assert 'get_collection_stats' in admin_methods
        
        # VectorAdmin should not have read or write methods
        assert 'search' not in admin_methods
        assert 'upsert_chunks' not in admin_methods
        assert 'delete_by_document_id' not in admin_methods

    def test_vector_store_combines_all_interfaces(self):
        """Test that VectorStore properly combines all interfaces."""
        store_methods = [method for method in dir(VectorStore) 
                        if not method.startswith('_') and callable(getattr(VectorStore, method, None))]
        
        # VectorStore should have all methods from all interfaces
        # Read methods
        assert 'search' in store_methods
        assert 'hybrid_search' in store_methods
        
        # Write methods
        assert 'upsert_chunks' in store_methods
        assert 'delete_by_document_id' in store_methods
        
        # Admin methods
        assert 'create_collection' in store_methods
        assert 'collection_exists' in store_methods
        assert 'get_collection_stats' in store_methods
