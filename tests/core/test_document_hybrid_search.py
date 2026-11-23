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


class _Seg:
    def __init__(self, text, relevance):
        self.content = type("C", (), {"text": text})()
        self.relevance_score = relevance


CUTOFF = 0.3


def _passing(segments, query):
    return [row for row in fuse_hybrid_scores(segments, query=query) if row[3] >= CUTOFF]


def test_single_relevant_chunk_passes_cutoff():
    rows = _passing([_Seg("meeting notes about the garden project", 0.8)], "garden project notes")
    assert len(rows) == 1


def test_single_irrelevant_chunk_stays_below_cutoff():
    rows = _passing([_Seg("meeting notes about the garden project", 0.05)], "quantum zebra")
    assert rows == []


def test_tied_high_similarity_candidates_pass():
    segs = [_Seg("garden notes one", 0.8), _Seg("garden notes two", 0.8)]
    assert len(_passing(segs, "garden notes")) == 2


def test_tied_low_similarity_candidates_fail():
    segs = [_Seg("alpha beta", 0.05), _Seg("gamma delta", 0.05)]
    assert _passing(segs, "quantum zebra") == []


def test_file_filter_leaving_one_chunk_matches_single_chunk_case():
    # Reason: a file_name filter shrinks the pool to one chunk, same as a one-file corpus.
    only = [_Seg("budget spreadsheet summary for march", 0.7)]
    assert len(_passing(only, "march budget")) == 1


def test_spread_pool_ranking_and_cutoff_unchanged():
    segs = [
        _Seg("alpha beta gamma", 0.2),
        _Seg("alpha beta keyword", 0.9),
        _Seg("nothing relevant here", 0.1),
        _Seg("keyword only", 0.5),
    ]
    fused = fuse_hybrid_scores(segs, query="keyword alpha")
    assert [r[0].content.text for r in fused] == [
        "alpha beta keyword",
        "keyword only",
        "alpha beta gamma",
        "nothing relevant here",
    ]
    passing = [r[0].content.text for r in fused if r[3] >= CUTOFF]
    assert passing == ["alpha beta keyword", "keyword only"]
