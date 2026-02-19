"""Expand level-k vertex IDs to their base K_n edge sets."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Dict, List, Tuple

from grahamtools.kn.levels import Endpoints


def expand_to_simple_base_edges_id(
    v: int,
    level: int,
    endpoints_by_level: Dict[int, List[Endpoints]],
) -> List[Tuple[int, int]]:
    """Expand a level-k vertex to its unique base edges of K_n (0-based).

    Returns a sorted list of (i, j) with i < j.
    """

    @lru_cache(maxsize=None)
    def rec(v_id: int, lvl: int) -> frozenset[Tuple[int, int]]:
        if lvl == 0:
            return frozenset()
        if lvl == 1:
            a, b = endpoints_by_level[1][v_id]
            return frozenset({(a, b) if a < b else (b, a)})
        a, b = endpoints_by_level[lvl][v_id]
        return rec(a, lvl - 1) | rec(b, lvl - 1)

    return sorted(rec(v, level))


def expand_to_base_edge_multiset_id(
    v: int,
    level: int,
    endpoints_by_level: Dict[int, List[Endpoints]],
) -> Counter[Tuple[int, int]]:
    """Expand a level-k vertex to a multiset of base edges, preserving multiplicities.

    Returns a Counter mapping (i, j) -> count, with i < j.
    """

    @lru_cache(maxsize=None)
    def rec(v_id: int, lvl: int) -> Tuple[Tuple[Tuple[int, int], int], ...]:
        if lvl == 0:
            return ()
        if lvl == 1:
            a, b = endpoints_by_level[1][v_id]
            if a > b:
                a, b = b, a
            return (((a, b), 1),)
        a, b = endpoints_by_level[lvl][v_id]
        A = Counter(dict(rec(a, lvl - 1)))
        B = Counter(dict(rec(b, lvl - 1)))
        C = A + B
        return tuple(sorted(C.items()))

    return Counter(dict(rec(v, level)))
