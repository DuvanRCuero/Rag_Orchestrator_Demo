"""Factory for creating chunking strategies."""

from typing import Dict, Type, Optional

from src.domain.chunking.base import ChunkingStrategy, ChunkingConfig
from src.domain.chunking.strategies import (
    RecursiveCharacterStrategy,
    SemanticStrategy,
    MarkdownStrategy,
    CodeAwareStrategy,
)


class ChunkingStrategyFactory:
    """Factory for creating chunking strategy instances."""

    _strategies: Dict[str, Type[ChunkingStrategy]] = {
        "recursive_character": RecursiveCharacterStrategy,
        "recursive": RecursiveCharacterStrategy,
        "semantic": SemanticStrategy,
        "markdown": MarkdownStrategy,
        "markdown_aware": MarkdownStrategy,
        "code": CodeAwareStrategy,
        "code_aware": CodeAwareStrategy,
    }

    @classmethod
    def register(cls, name: str, strategy_class: Type[ChunkingStrategy]) -> None:
        """Register a new chunking strategy."""
        cls._strategies[name] = strategy_class

    @classmethod
    def create(
        cls,
        strategy_name: str,
        config: Optional[ChunkingConfig] = None,
    ) -> ChunkingStrategy:
        """Create a chunking strategy instance."""
        if strategy_name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(
                f"Unknown chunking strategy: {strategy_name}. "
                f"Available: {available}"
            )
        
        strategy_class = cls._strategies[strategy_name]
        return strategy_class(config)

    @classmethod
    def available_strategies(cls) -> list:
        """List available strategies."""
        return list(set(cls._strategies.values()))

    @classmethod
    def get_for_document_type(
        cls,
        doc_type: str,
        config: Optional[ChunkingConfig] = None,
    ) -> ChunkingStrategy:
        """Get appropriate strategy based on document type."""
        type_mapping = {
            "md": "markdown",
            "markdown": "markdown",
            "py": "code",
            "python": "code",
            "js": "code",
            "ts": "code",
            "txt": "recursive_character",
            "pdf": "recursive_character",
        }
        
        strategy_name = type_mapping.get(doc_type.lower(), "recursive_character")
        return cls.create(strategy_name, config)
