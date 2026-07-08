"""Lightweight BM25 + vector fusion for document RAG search."""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1]


class SimpleBM25:
    """In-process BM25 over a fixed document pool (no external index)."""

    def __init__(
        self,
        documents: Sequence[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(doc) for doc in documents]
        self.doc_lens = [len(doc) for doc in self.docs]
        self.avgdl = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0
        df: dict[str, int] = {}
        for doc in self.docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self.df = df
        self.n_docs = len(self.docs)

    def score_document(self, query: str, doc_index: int) -> float:
        if doc_index < 0 or doc_index >= self.n_docs or not self.docs:
            return 0.0
        query_terms = tokenize(query)
        if not query_terms:
            return 0.0
        doc = self.docs[doc_index]
        doc_len = self.doc_lens[doc_index]
        term_freq: dict[str, int] = {}
        for term in doc:
            term_freq[term] = term_freq.get(term, 0) + 1
        total = 0.0
        for term in query_terms:
            freq = term_freq.get(term, 0)
            if freq == 0:
                continue
            df = self.df.get(term, 0)
            idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            denom = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1))
            total += idf * (freq * (self.k1 + 1)) / denom
        return total

    def score_all(self, query: str) -> list[float]:
        return [self.score_document(query, i) for i in range(self.n_docs)]


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi <= lo:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def fuse_hybrid_scores(
    segments: Sequence[Any],
    *,
    query: str,
    vector_weight: float = 0.55,
    lexical_weight: float = 0.45,
) -> list[tuple[Any, float, float, float]]:
    """Return (segment, vector_score, lexical_score, fused_score) sorted by fused desc."""
    if not segments:
        return []
    texts = [(getattr(seg.content, "text", None) or "") for seg in segments]
    bm25 = SimpleBM25(texts)
    lexical_raw = bm25.score_all(query)
    lexical_norm = _normalize_scores(lexical_raw)
    vector_raw = [float(getattr(seg, "relevance_score", 0.0) or 0.0) for seg in segments]
    vector_norm = _normalize_scores(vector_raw)

    fused: list[tuple[Any, float, float, float]] = []
    for seg, vec, lex in zip(segments, vector_norm, lexical_norm, strict=True):
        combined = vector_weight * vec + lexical_weight * lex
        fused.append((seg, vec, lex, combined))
    fused.sort(key=lambda row: row[3], reverse=True)
    return fused
