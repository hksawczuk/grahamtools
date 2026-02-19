"""Core level generation for iterated line graphs of K_n.

Level 0 vertices are 0..n-1 (base vertices of K_n).
Level t >= 1 vertices are indexed 0..|V_t|-1, with each vertex
storing its two parent endpoint IDs from level t-1.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from typing import Dict, List, Set, Tuple

Endpoints = Tuple[int, int]


def canon_pair_int(a: int, b: int) -> Endpoints:
    """Canonical ordering of an endpoint pair."""
    return (a, b) if a <= b else (b, a)


@lru_cache(maxsize=None)
def _pair_to_idx_list(n: int) -> list[list[int]]:
    """Return a 2D table idx[i][j] = bit position for 0<=i<j<n, else -1."""
    idx = [[-1] * n for _ in range(n)]
    t = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx[i][j] = t
            t += 1
    return idx


@lru_cache(maxsize=None)
def _idx_to_pair(n: int) -> list[Endpoints]:
    """Inverse mapping: bit index -> (i, j) with 0<=i<j<n."""
    pairs: list[Endpoints] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return pairs


def _is_forest_edgebit(bits: int, n: int) -> bool:
    """Check if the graph encoded by edge-bitset is acyclic (a forest)."""
    if bits == 0:
        return True
    if bin(bits).count("1") >= n:
        return False

    pairs = _idx_to_pair(n)
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    tmp = bits
    while tmp:
        lsb = tmp & -tmp
        idx = lsb.bit_length() - 1
        tmp ^= lsb
        u, v = pairs[idx]
        if not union(u, v):
            return False
    return True


def generate_levels_Kn_ids(
    n: int,
    k: int,
    prune_cycles: bool = False,
) -> tuple[Dict[int, List[int]], Dict[int, List[Endpoints]]]:
    """Build levels 0..k of vertices of L^t(K_n) by incidence recursion.

    Parameters
    ----------
    n : int
        Number of base vertices (K_n).
    k : int
        Maximum iterate level to build.
    prune_cycles : bool
        If True, only keep vertices whose base-edge set is a forest (tree).
        This is useful for fiber coefficient analysis.

    Returns
    -------
    V_by_level : dict[int, list[int]]
        V_by_level[t] = list of vertex IDs [0..|V_t|-1].
    endpoints_by_level : dict[int, list[Endpoints]]
        endpoints_by_level[t][v] = (a, b) endpoint pair from level t-1, for t >= 1.
    """
    if n <= 0:
        return {0: []}, {}

    V_by_level: Dict[int, List[int]] = {0: list(range(n))}
    endpoints_by_level: Dict[int, List[Endpoints]] = {}

    if k == 0:
        return V_by_level, endpoints_by_level

    # Level 1: edges of K_n
    ep1: List[Endpoints] = []
    bit1: List[int] = []
    idx = _pair_to_idx_list(n)
    for i, j in combinations(range(n), 2):
        ep1.append((i, j))
        bit1.append(1 << idx[i][j])
    V_by_level[1] = list(range(len(ep1)))
    endpoints_by_level[1] = ep1

    bits_prev = bit1
    ok_prev = [True] * len(bits_prev)

    for level in range(2, k + 1):
        ep_prev = endpoints_by_level[level - 1]

        incidence: Dict[int, List[int]] = defaultdict(list)
        for e_id, (a, b) in enumerate(ep_prev):
            if prune_cycles and not ok_prev[e_id]:
                continue
            incidence[a].append(e_id)
            incidence[b].append(e_id)

        next_pairs: Set[Endpoints] = set()
        bits_next_map: Dict[Endpoints, int] = {}

        for inc_edges in incidence.values():
            for e1, e2 in combinations(inc_edges, 2):
                p = canon_pair_int(e1, e2)
                if p in next_pairs:
                    continue

                if prune_cycles:
                    b = bits_prev[e1] | bits_prev[e2]
                    if bin(b).count("1") >= n:
                        continue
                    if not _is_forest_edgebit(b, n):
                        continue
                    bits_next_map[p] = b

                next_pairs.add(p)

        ep_next = sorted(next_pairs)
        endpoints_by_level[level] = ep_next
        V_by_level[level] = list(range(len(ep_next)))

        if prune_cycles:
            bits_next = [bits_next_map[p] for p in ep_next]
            ok_next = [True] * len(bits_next)
        else:
            bits_next = [0] * len(ep_next)
            ok_next = [True] * len(ep_next)

        bits_prev, ok_prev = bits_next, ok_next

    return V_by_level, endpoints_by_level
