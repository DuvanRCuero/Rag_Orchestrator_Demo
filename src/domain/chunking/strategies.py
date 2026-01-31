"""Chunking strategy implementations."""

from typing import Any, Dict, List

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

from src.core.schemas import DocumentChunk
from src.domain.chunking.base import ChunkingStrategy, ChunkingConfig


class RecursiveCharacterStrategy(ChunkingStrategy):
    """Recursive character text splitting strategy."""

    @property
    def name(self) -> str:
        return "recursive_character"

    def chunk(
        self,
        text: str,
        metadata: Dict[str, Any] = None,
    ) -> List[DocumentChunk]:
        metadata = metadata or {}
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

        texts = splitter.split_text(text)
        
        return [
            self._create_chunk(content, i, metadata)
            for i, content in enumerate(texts)
            if len(content.strip()) >= self.config.min_chunk_size
        ]


class SemanticStrategy(ChunkingStrategy):
    """Semantic-aware chunking based on sentence boundaries."""

    @property
    def name(self) -> str:
        return "semantic"

    def __init__(self, config: ChunkingConfig = None):
        super().__init__(config or ChunkingConfig(chunk_size=500, chunk_overlap=100))

    def chunk(
        self,
        text: str,
        metadata: Dict[str, Any] = None,
    ) -> List[DocumentChunk]:
        metadata = metadata or {}
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
            keep_separator=False,
        )

        texts = splitter.split_text(text)
        
        return [
            self._create_chunk(content.strip(), i, metadata)
            for i, content in enumerate(texts)
            if len(content.strip()) >= self.config.min_chunk_size
        ]


class MarkdownStrategy(ChunkingStrategy):
    """Markdown-aware chunking that respects headers."""

    @property
    def name(self) -> str:
        return "markdown"

    def __init__(self, config: ChunkingConfig = None):
        super().__init__(config or ChunkingConfig(chunk_size=800, chunk_overlap=150))

    def chunk(
        self,
        text: str,
        metadata: Dict[str, Any] = None,
    ) -> List[DocumentChunk]:
        metadata = metadata or {}
        
        # First split by headers
        headers_to_split = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ]
        
        try:
            md_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split,
            )
            md_docs = md_splitter.split_text(text)
            
            # Then apply recursive splitting to each section
            char_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )
            
            chunks = []
            chunk_index = 0
            
            for doc in md_docs:
                section_metadata = {**metadata, **doc.metadata}
                texts = char_splitter.split_text(doc.page_content)
                
                for content in texts:
                    if len(content.strip()) >= self.config.min_chunk_size:
                        chunks.append(self._create_chunk(
                            content.strip(),
                            chunk_index,
                            section_metadata,
                        ))
                        chunk_index += 1
            
            return chunks
            
        except Exception:
            # Fallback to recursive character splitting
            fallback = RecursiveCharacterStrategy(self.config)
            return fallback.chunk(text, metadata)


class CodeAwareStrategy(ChunkingStrategy):
    """Code-aware chunking that respects function/class boundaries."""

    @property
    def name(self) -> str:
        return "code_aware"

    def chunk(
        self,
        text: str,
        metadata: Dict[str, Any] = None,
    ) -> List[DocumentChunk]:
        metadata = metadata or {}
        
        # Use code-specific separators
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=[
                "\nclass ",
                "\ndef ",
                "\nasync def ",
                "\n\n",
                "\n",
                " ",
            ],
            keep_separator=True,
        )

        texts = splitter.split_text(text)
        
        return [
            self._create_chunk(content, i, metadata)
            for i, content in enumerate(texts)
            if len(content.strip()) >= self.config.min_chunk_size
        ]
