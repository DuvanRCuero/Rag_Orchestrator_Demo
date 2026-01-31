import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


from langchain_community.document_loaders import (PyPDFLoader, TextLoader,
                                                  UnstructuredFileLoader,
                                                  UnstructuredMarkdownLoader)
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

from src.core.config import settings
from src.core.exceptions import IngestionError
from src.core.logging import get_logger
from src.core.schemas import Document, DocumentChunk, DocumentType
from src.domain.chunking import (
    ChunkingStrategy,
    ChunkingStrategyFactory,
    ChunkingConfig,
)

logger = get_logger(__name__)


class AdvancedDocumentProcessor:
    """Document processor using Strategy Pattern for chunking."""

    def __init__(
        self,
        strategy: ChunkingStrategy = None,
        strategy_name: str = None,
        config: ChunkingConfig = None,
    ):
        if strategy:
            self._strategy = strategy
        else:
            name = strategy_name or settings.TEXT_SPLITTER
            self._strategy = ChunkingStrategyFactory.create(name, config)
        
        logger.info(
            "document_processor_initialized",
            strategy=self._strategy.name,
        )

    @property
    def strategy(self) -> ChunkingStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, value: ChunkingStrategy) -> None:
        self._strategy = value
        logger.info("chunking_strategy_changed", strategy=value.name)

    @property
    def strategy_name(self) -> str:
        """Backward compatibility property."""
        return self._strategy.name

    def create_chunks(
        self,
        text: str,
        metadata: Dict[str, Any] = None,
    ) -> List[DocumentChunk]:
        """Create chunks using current strategy."""
        logger.debug(
            "chunking_started",
            strategy=self._strategy.name,
            text_length=len(text),
        )
        
        chunks = self._strategy.chunk(text, metadata)
        
        logger.info(
            "chunking_completed",
            strategy=self._strategy.name,
            chunks_created=len(chunks),
        )
        
        return chunks

    def create_chunks_auto(
        self,
        text: str,
        metadata: Dict[str, Any] = None,
        doc_type: str = "txt",
    ) -> List[DocumentChunk]:
        """Auto-select strategy based on document type."""
        strategy = ChunkingStrategyFactory.get_for_document_type(doc_type)
        return strategy.chunk(text, metadata)

    def load_document(
        self, file_path: str, doc_type: DocumentType
    ) -> List[Dict[str, Any]]:
        """Load document with appropriate loader."""
        try:
            if doc_type == DocumentType.PDF:
                loader = PyPDFLoader(file_path)
            elif doc_type == DocumentType.MD:
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                loader = UnstructuredFileLoader(file_path)

            documents = loader.load()
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": file_path,
                }
                for doc in documents
            ]
        except Exception as e:
            raise IngestionError(
                detail=f"Failed to load document: {str(e)}",
                metadata={"file_path": file_path, "type": doc_type},
            )

    def create_intelligent_chunks(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Create chunks with semantic boundaries and hierarchy awareness."""
        # Use auto-detection for backward compatibility
        doc_type = "txt"
        if metadata.get("type") == DocumentType.MD:
            doc_type = "md"
        elif self._contains_code(text):
            doc_type = "py"
        
        # Use the auto strategy selector
        chunks = self.create_chunks_auto(text, metadata, doc_type)
        
        # Post-process chunks for quality
        chunks = self._post_process_chunks(chunks)
        
        return chunks

    def _contains_code(self, text: str) -> bool:
        """Detect if text contains code."""
        code_patterns = [
            r"def\s+\w+\s*\(",  # Python function
            r"import\s+\w+",  # Import statement
            r"class\s+\w+",  # Class definition
            r"```",  # Code block
            r"\.\w+\s*\(",  # Method call
            r"=\s*\[|\]",  # List/dict assignment
        ]

        for pattern in code_patterns:
            if re.search(pattern, text):
                return True
        return False

    def _post_process_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Post-process chunks for quality."""
        processed_chunks = []

        for chunk in chunks:
            # Remove leading/trailing whitespace
            chunk.content = chunk.content.strip()

            # Ensure chunk ends with complete sentence if possible
            if not chunk.content.endswith((".", "!", "?", "```")):
                # Try to find last sentence boundary
                last_period = chunk.content.rfind(". ")
                last_excl = chunk.content.rfind("! ")
                last_quest = chunk.content.rfind("? ")

                last_boundary = max(last_period, last_excl, last_quest)
                if last_boundary > len(chunk.content) * 0.7:  # If near end
                    chunk.content = chunk.content[: last_boundary + 1]

            # Skip chunks that are too small (except code)
            if len(chunk.content.split()) < 20 and not self._contains_code(
                chunk.content
            ):
                continue

            processed_chunks.append(chunk)

        return processed_chunks

    def analyze_document_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze document for optimal chunking strategy."""
        word_count = len(text.split())
        line_count = len(text.split("\n"))

        # Detect document type patterns
        has_markdown = bool(re.search(r"^#\s+", text, re.MULTILINE))
        has_code = self._contains_code(text)

        # Calculate average sentence length
        sentences = re.split(r"[.!?]+", text)
        avg_sentence_len = sum(len(s.split()) for s in sentences) / max(
            len(sentences), 1
        )

        complexity = {
            "word_count": word_count,
            "line_count": line_count,
            "has_markdown": has_markdown,
            "has_code": has_code,
            "avg_sentence_length": avg_sentence_len,
            "recommended_strategy": self._recommend_strategy(
                has_markdown, has_code, avg_sentence_len
            ),
        }

        return complexity

    def _recommend_strategy(
        self, has_markdown: bool, has_code: bool, avg_sent_len: float
    ) -> str:
        """Recommend optimal chunking strategy."""
        if has_markdown:
            return "markdown_aware"
        elif has_code:
            return "recursive_character"  # Keep code together
        elif avg_sent_len > 25:
            return "semantic"
        else:
            return "recursive_character"
