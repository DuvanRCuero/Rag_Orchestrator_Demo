"""BM25 scoring using rank_bm25 library."""

import re
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class BM25Result:
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class BM25Scorer:

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []
        self.chunk_metadata: List[Dict[str, Any]] = []
        self._bm25: BM25Okapi = None

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def index_documents(self, chunks: List[Dict[str, Any]]) -> None:
        self.corpus = [self._tokenize(c['content']) for c in chunks]
        self.chunk_metadata = chunks
        self._bm25 = BM25Okapi(self.corpus, k1=self.k1, b=self.b)

    def score(self, query: str, top_k: int = 10) -> List[BM25Result]:
        if not self._bm25:
            return []

        scores = self._bm25.get_scores(self._tokenize(query))
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            BM25Result(
                chunk_id=self.chunk_metadata[i]['id'],
                content=self.chunk_metadata[i]['content'],
                score=float(scores[i]),
                metadata=self.chunk_metadata[i],
            )
            for i in top_indices if scores[i] > 0
        ]

    def invalidate(self) -> None:
        self._bm25 = None
        self.corpus = []
        self.chunk_metadata = []
