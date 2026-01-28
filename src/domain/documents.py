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
from src.core.schemas import Document, DocumentChunk, DocumentType


@dataclass
class ChunkingStrategy:
    """Configuration for text chunking strategy."""

    name: str
    chunk_size: int
    chunk_overlap: int
    separators: List[str]
    keep_separator: bool


class AdvancedDocumentProcessor:
    """Production-grade document processor with multiple chunking strategies."""

    STRATEGIES = {
        "recursive_character": ChunkingStrategy(
            name="recursive_character",
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True,
        ),
        "semantic": ChunkingStrategy(
            name="semantic",
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            keep_separator=False,
        ),
        "markdown_aware": ChunkingStrategy(
            name="markdown_aware",
            chunk_size=800,
            chunk_overlap=150,
            separators=["# ", "## ", "### ", "\n\n", "\n", " "],
            keep_separator=True,
        ),
    }

    def __init__(self, strategy_name: str = None):
        self.strategy_name = strategy_name or settings.TEXT_SPLITTER
        self.strategy = self.STRATEGIES.get(
            self.strategy_name, self.STRATEGIES["recursive_character"]
        )

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

        # Detect document type for specialized splitting
        if metadata.get("type") == DocumentType.MD:
            chunks = self._split_markdown_with_hierarchy(text, metadata)
        elif self._contains_code(text):
            chunks = self._split_code_aware(text, metadata)
        else:
            chunks = self._split_recursive(text, metadata)

        # Post-process chunks for quality
        chunks = self._post_process_chunks(chunks)

        return chunks

    def _split_recursive(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Recursive character splitting with semantic boundaries."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.strategy.chunk_size,
            chunk_overlap=self.strategy.chunk_overlap,
            separators=self.strategy.separators,
            keep_separator=self.strategy.keep_separator,
            length_function=len,
            is_separator_regex=False,
        )

        raw_chunks = splitter.split_text(text)
        return self._chunks_to_models(raw_chunks, metadata)

    def _split_markdown_with_hierarchy(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Markdown-aware splitting preserving hierarchy."""
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )

        raw_chunks = markdown_splitter.split_text(text)

        # Convert to our models
        chunks = []
        for idx, chunk in enumerate(raw_chunks):
            chunk_metadata = {**metadata, **chunk.metadata}
            chunk_id = f"{metadata.get('document_id', 'doc')}_{idx}"

            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    content=chunk.page_content,
                    metadata=chunk_metadata,
                    document_id=metadata.get("document_id"),
                    chunk_index=idx,
                )
            )

        return chunks

    def _split_code_aware(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Splitting that preserves code blocks and imports."""
        # Preserve code blocks
        code_blocks = re.findall(r"```[\s\S]*?```", text)

        # Split on natural boundaries but keep code blocks together
        sections = []
        current_section = []
        lines = text.split("\n")

        for line in lines:
            if line.strip().startswith("```") or any(
                code in line for code in code_blocks
            ):
                # Code block - keep together
                current_section.append(line)
            elif len(" ".join(current_section)) > self.strategy.chunk_size:
                sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            sections.append("\n".join(current_section))

        return self._chunks_to_models(sections, metadata)

    def _chunks_to_models(
        self, raw_chunks: List[str], metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Convert raw text chunks to DocumentChunk models."""
        chunks = []
        for idx, content in enumerate(raw_chunks):
            if not content.strip():
                continue

            chunk_id = hashlib.md5(
                f"{metadata.get('document_id')}_{idx}".encode()
            ).hexdigest()[:16]

            chunk_metadata = {
                **metadata,
                "chunk_size": len(content),
                "chunk_strategy": self.strategy_name,
                "has_code": self._contains_code(content),
                "word_count": len(content.split()),
            }

            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    content=content,
                    metadata=chunk_metadata,
                    document_id=metadata.get("document_id"),
                    chunk_index=idx,
                )
            )

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
