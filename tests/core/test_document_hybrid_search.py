"""Tests for BM25 + vector fusion helpers."""

from core.document_hybrid_search import SimpleBM25, fuse_hybrid_scores, tokenize


def test_tokenize_strips_short_tokens():
    assert tokenize("ISO-27001 audit report") == ["iso", "27001", "audit", "report"]


def test_bm25_prefers_exact_term_match():
    bm25 = SimpleBM25(
        [
            "cats and dogs are pets",
            "ISO-27001 compliance audit checklist",
            "unrelated weather forecast",
        ]
    )
    scores = bm25.score_all("ISO-27001 audit")
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_fuse_hybrid_scores_combines_vector_and_lexical():
    class Seg:
        def __init__(self, text, relevance):
            self.content = type("C", (), {"text": text})()
            self.relevance_score = relevance

    segments = [
        Seg("alpha beta gamma", 0.2),
        Seg("alpha beta keyword", 0.9),
    ]
    fused = fuse_hybrid_scores(segments, query="keyword alpha")
    assert fused[0][0].content.text == "alpha beta keyword"
