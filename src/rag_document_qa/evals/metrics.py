"""Retrieval metrics: Recall@K, MRR, nDCG@K.

Senior-engineer note: the gold-passage match function is pluggable. The
default `passage_overlap_match` uses character-span overlap with a configurable
threshold (default 0.5). For benchmarks like FinDER that ship doc-level golds
plus passage-level evidence, `doc_id_match` is the right lookup.

Edge cases handled:
- Multiple gold passages per query: hit if ANY gold matches a retrieved chunk.
- No retrieval: all metrics return 0 for that query (not NaN).
- nDCG with non-binary relevance: relevance is binary in our setting (gold
  match or not), so nDCG reduces to 1/log2(rank+1) at the first hit.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from rag_document_qa.types import Chunk, GoldPair, RetrievedChunk

GoldMatcher = Callable[[Chunk, GoldPair], bool]


def passage_overlap_match(
    chunk: Chunk,
    pair: GoldPair,
    threshold: float = 0.5,
) -> bool:
    """Match a chunk against a gold pair via character-span containment.

    A chunk matches if the chunk's text contains at least `threshold` fraction
    of any gold passage, OR a gold passage contains at least `threshold` of the
    chunk. Symmetric so partial overlap is captured either direction.
    """
    if pair.gold_doc_id and chunk.doc_id != pair.gold_doc_id:
        # Doc id is a hard prefilter when the benchmark provides it.
        return False
    chunk_norm = " ".join(chunk.text.split()).lower()
    if not chunk_norm:
        return False
    for gold in pair.gold_passages:
        gold_norm = " ".join(gold.split()).lower()
        if not gold_norm:
            continue
        if gold_norm in chunk_norm:
            return True
        if chunk_norm in gold_norm:
            return True
        # Soft overlap: longest common substring length / shorter length
        overlap = _longest_common_substring(chunk_norm, gold_norm)
        shorter = min(len(chunk_norm), len(gold_norm))
        if shorter > 0 and overlap / shorter >= threshold:
            return True
    return False


def doc_id_match(chunk: Chunk, pair: GoldPair) -> bool:
    """Loose match: a hit at doc-id level only. Useful for ablations where
    you want to know "did we even find the right document"."""
    return chunk.doc_id == pair.gold_doc_id


def recall_at_k(
    retrieved: list[RetrievedChunk],
    pair: GoldPair,
    k: int,
    matcher: GoldMatcher = passage_overlap_match,
) -> float:
    """Was any gold passage hit within the top-k retrieved chunks?

    Returns 1.0 if any gold matched; 0.0 otherwise. (Binary, not the
    multi-relevant-doc graded recall.)
    """
    if k <= 0 or not retrieved:
        return 0.0
    for r in retrieved[:k]:
        if matcher(r.chunk, pair):
            return 1.0
    return 0.0


def reciprocal_rank(
    retrieved: list[RetrievedChunk],
    pair: GoldPair,
    matcher: GoldMatcher = passage_overlap_match,
) -> float:
    """1 / rank of the first match. 0 if no match in `retrieved`."""
    for r in retrieved:
        if matcher(r.chunk, pair):
            return 1.0 / r.rank
    return 0.0


def ndcg_at_k(
    retrieved: list[RetrievedChunk],
    pair: GoldPair,
    k: int,
    matcher: GoldMatcher = passage_overlap_match,
) -> float:
    """nDCG@k for binary relevance. Reduces to 1/log2(rank+1) at the first hit
    if it's within k, normalized by the ideal score (which is 1.0)."""
    if k <= 0 or not retrieved:
        return 0.0
    dcg = 0.0
    for r in retrieved[:k]:
        if matcher(r.chunk, pair):
            dcg += 1.0 / math.log2(r.rank + 1)
            break  # binary relevance: at most one hit counted
    # ideal DCG for binary single-relevant: 1 / log2(2) = 1.0 (rank 1)
    return dcg


def aggregate(values: list[float]) -> float:
    """Mean of a list of per-query scores. 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def _longest_common_substring(a: str, b: str) -> int:
    """Length of the longest common substring of `a` and `b`.

    Rolling 1-D DP for O(min(len_a, len_b)) memory. Used by
    `passage_overlap_match` to score partial overlap between a chunk and a
    gold passage when neither contains the other.
    """
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    best = 0
    for ch_a in a:
        curr = [0] * (len(b) + 1)
        for j, ch_b in enumerate(b, start=1):
            if ch_a == ch_b:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best:
                    best = curr[j]
        prev = curr
    return best
