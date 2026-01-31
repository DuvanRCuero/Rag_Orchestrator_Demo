"""Tests for chunking strategies and factory."""

import pytest

from src.domain.chunking import (
    ChunkingStrategy,
    ChunkingConfig,
    ChunkingStrategyFactory,
    RecursiveCharacterStrategy,
    SemanticStrategy,
    MarkdownStrategy,
    CodeAwareStrategy,
)


class TestChunkingConfig:
    """Test ChunkingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ChunkingConfig()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.min_chunk_size == 100

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=100,
            min_chunk_size=50,
        )
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.min_chunk_size == 50


class TestRecursiveCharacterStrategy:
    """Test RecursiveCharacterStrategy."""

    def test_strategy_name(self):
        """Test strategy name property."""
        strategy = RecursiveCharacterStrategy()
        assert strategy.name == "recursive_character"

    def test_chunk_basic_text(self):
        """Test chunking basic text."""
        strategy = RecursiveCharacterStrategy()
        text = "This is a test sentence. " * 100
        chunks = strategy.chunk(text, {"document_id": "test_doc"})
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.document_id == "test_doc"
            assert chunk.metadata["chunk_strategy"] == "recursive_character"
            assert len(chunk.content) <= strategy.config.chunk_size + strategy.config.chunk_overlap

    def test_chunk_with_custom_config(self):
        """Test chunking with custom configuration."""
        config = ChunkingConfig(chunk_size=200, chunk_overlap=50)
        strategy = RecursiveCharacterStrategy(config)
        text = "Short sentence. " * 50
        chunks = strategy.chunk(text, {"document_id": "test_doc"})
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 200 + 50


class TestSemanticStrategy:
    """Test SemanticStrategy."""

    def test_strategy_name(self):
        """Test strategy name property."""
        strategy = SemanticStrategy()
        assert strategy.name == "semantic"

    def test_default_config(self):
        """Test that semantic strategy has smaller default chunk size."""
        strategy = SemanticStrategy()
        assert strategy.config.chunk_size == 500
        assert strategy.config.chunk_overlap == 100

    def test_chunk_with_sentences(self):
        """Test chunking text with sentence boundaries."""
        strategy = SemanticStrategy()
        text = "First sentence. Second sentence! Third sentence? " * 20
        chunks = strategy.chunk(text, {"document_id": "test_doc"})
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata["chunk_strategy"] == "semantic"


class TestMarkdownStrategy:
    """Test MarkdownStrategy."""

    def test_strategy_name(self):
        """Test strategy name property."""
        strategy = MarkdownStrategy()
        assert strategy.name == "markdown"

    def test_chunk_markdown_with_headers(self):
        """Test chunking markdown with headers."""
        strategy = MarkdownStrategy()
        text = """# Main Header
This is content under the main header with sufficient length to meet minimum requirements. We need to add more content here to ensure that the chunks are large enough to pass the filtering.

## Subheader
More content under the subheader with enough text to pass the minimum chunk size filter. Adding even more text here to make sure this section is substantial.

### Sub-subheader
Even more detailed content here with additional words to ensure chunk size requirements are met. Let's keep adding text to make this chunk long enough.
""" * 3  # Repeat to ensure enough content
        chunks = strategy.chunk(text, {"document_id": "test_md"})
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata["chunk_strategy"] == "markdown"

    def test_markdown_fallback(self):
        """Test markdown strategy fallback to recursive when markdown parsing fails."""
        strategy = MarkdownStrategy()
        # Plain text without markdown headers
        text = "Plain text content. " * 100
        chunks = strategy.chunk(text, {"document_id": "test_plain"})
        
        # Should still create chunks via fallback
        assert len(chunks) > 0


class TestCodeAwareStrategy:
    """Test CodeAwareStrategy."""

    def test_strategy_name(self):
        """Test strategy name property."""
        strategy = CodeAwareStrategy()
        assert strategy.name == "code_aware"

    def test_chunk_python_code(self):
        """Test chunking Python code."""
        strategy = CodeAwareStrategy()
        text = """
class MyClass:
    def method_one(self):
        pass

def function_one():
    return True

async def async_function():
    await something()
""" * 10
        chunks = strategy.chunk(text, {"document_id": "test_code"})
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.metadata["chunk_strategy"] == "code_aware"


class TestChunkingStrategyFactory:
    """Test ChunkingStrategyFactory."""

    def test_create_recursive_strategy(self):
        """Test creating recursive character strategy."""
        strategy = ChunkingStrategyFactory.create("recursive_character")
        assert isinstance(strategy, RecursiveCharacterStrategy)
        assert strategy.name == "recursive_character"

    def test_create_recursive_alias(self):
        """Test creating strategy with alias."""
        strategy = ChunkingStrategyFactory.create("recursive")
        assert isinstance(strategy, RecursiveCharacterStrategy)

    def test_create_semantic_strategy(self):
        """Test creating semantic strategy."""
        strategy = ChunkingStrategyFactory.create("semantic")
        assert isinstance(strategy, SemanticStrategy)

    def test_create_markdown_strategy(self):
        """Test creating markdown strategy."""
        strategy = ChunkingStrategyFactory.create("markdown")
        assert isinstance(strategy, MarkdownStrategy)

    def test_create_markdown_alias(self):
        """Test creating markdown strategy with alias."""
        strategy = ChunkingStrategyFactory.create("markdown_aware")
        assert isinstance(strategy, MarkdownStrategy)

    def test_create_code_strategy(self):
        """Test creating code aware strategy."""
        strategy = ChunkingStrategyFactory.create("code")
        assert isinstance(strategy, CodeAwareStrategy)

    def test_create_code_alias(self):
        """Test creating code strategy with alias."""
        strategy = ChunkingStrategyFactory.create("code_aware")
        assert isinstance(strategy, CodeAwareStrategy)

    def test_create_with_custom_config(self):
        """Test creating strategy with custom configuration."""
        config = ChunkingConfig(chunk_size=300, chunk_overlap=50)
        strategy = ChunkingStrategyFactory.create("recursive", config)
        assert strategy.config.chunk_size == 300
        assert strategy.config.chunk_overlap == 50

    def test_create_unknown_strategy(self):
        """Test creating unknown strategy raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ChunkingStrategyFactory.create("unknown_strategy")
        assert "Unknown chunking strategy" in str(exc_info.value)
        assert "unknown_strategy" in str(exc_info.value)

    def test_available_strategies(self):
        """Test listing available strategies."""
        strategies = ChunkingStrategyFactory.available_strategies()
        assert len(strategies) == 4  # 4 unique strategy classes
        
        # Create instances and check names
        names = set()
        for strategy_class in strategies:
            instance = strategy_class()
            names.add(instance.name)
        
        assert "recursive_character" in names
        assert "semantic" in names
        assert "markdown" in names
        assert "code_aware" in names

    def test_get_for_document_type_markdown(self):
        """Test getting strategy for markdown document type."""
        strategy = ChunkingStrategyFactory.get_for_document_type("md")
        assert isinstance(strategy, MarkdownStrategy)
        
        strategy = ChunkingStrategyFactory.get_for_document_type("markdown")
        assert isinstance(strategy, MarkdownStrategy)

    def test_get_for_document_type_code(self):
        """Test getting strategy for code document types."""
        for doc_type in ["py", "python", "js", "ts"]:
            strategy = ChunkingStrategyFactory.get_for_document_type(doc_type)
            assert isinstance(strategy, CodeAwareStrategy)

    def test_get_for_document_type_text(self):
        """Test getting strategy for text document types."""
        for doc_type in ["txt", "pdf"]:
            strategy = ChunkingStrategyFactory.get_for_document_type(doc_type)
            assert isinstance(strategy, RecursiveCharacterStrategy)

    def test_get_for_document_type_unknown(self):
        """Test getting strategy for unknown document type defaults to recursive."""
        strategy = ChunkingStrategyFactory.get_for_document_type("unknown")
        assert isinstance(strategy, RecursiveCharacterStrategy)

    def test_register_custom_strategy(self):
        """Test registering a custom strategy."""
        class CustomStrategy(ChunkingStrategy):
            @property
            def name(self):
                return "custom"
            
            def chunk(self, text, metadata=None):
                return []
        
        ChunkingStrategyFactory.register("custom", CustomStrategy)
        strategy = ChunkingStrategyFactory.create("custom")
        assert isinstance(strategy, CustomStrategy)
        assert strategy.name == "custom"


class TestChunkMetadata:
    """Test chunk metadata generation."""

    def test_chunk_includes_strategy_name(self):
        """Test that chunks include strategy name in metadata."""
        strategy = RecursiveCharacterStrategy()
        text = "Test content. " * 50
        chunks = strategy.chunk(text, {"document_id": "test"})
        
        for chunk in chunks:
            assert "chunk_strategy" in chunk.metadata
            assert chunk.metadata["chunk_strategy"] == "recursive_character"

    def test_chunk_includes_size(self):
        """Test that chunks include size in metadata."""
        strategy = RecursiveCharacterStrategy()
        text = "Test content. " * 50
        chunks = strategy.chunk(text, {"document_id": "test"})
        
        for chunk in chunks:
            assert "chunk_size" in chunk.metadata
            assert chunk.metadata["chunk_size"] == len(chunk.content)

    def test_chunk_preserves_custom_metadata(self):
        """Test that chunks preserve custom metadata."""
        strategy = RecursiveCharacterStrategy()
        text = "Test content. " * 50
        custom_metadata = {
            "document_id": "test",
            "author": "Test Author",
            "source": "test.txt",
        }
        chunks = strategy.chunk(text, custom_metadata)
        
        for chunk in chunks:
            assert chunk.metadata["author"] == "Test Author"
            assert chunk.metadata["source"] == "test.txt"
            assert chunk.document_id == "test"
