"""Classification of K_n iterated line graph vertices by isomorphism class."""
from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple

from grahamtools.kn.levels import Endpoints
from grahamtools.kn.expand import expand_to_simple_base_edges_id
from grahamtools.external.nauty import nauty_available, edgelist_to_g6, canon_g6
from grahamtools.utils.automorphisms import aut_size_edges


def _canon_key_bruteforce_bitset(edges: list[tuple[int, int]], n: int) -> int:
    """Canonical key under S_n: min edge-bitset over all permutations.

    Fallback when nauty is not available. Only practical for small n.
    """
    from functools import lru_cache
    from itertools import permutations

    @lru_cache(maxsize=None)
    def _pair_to_idx_list_local(n: int) -> list[list[int]]:
        idx = [[-1] * n for _ in range(n)]
        t = 0
        for i in range(n):
            for j in range(i + 1, n):
                idx[i][j] = t
                t += 1
        return idx

    @lru_cache(maxsize=None)
    def _all_perms(n: int) -> list[tuple[int, ...]]:
        return list(permutations(range(n)))

    if not edges:
        return 0

    idx = _pair_to_idx_list_local(n)
    perms = _all_perms(n)

    best = None
    for p in perms:
        bits = 0
        for u, v in edges:
            pu, pv = p[u], p[v]
            if pu > pv:
                pu, pv = pv, pu
            bits |= 1 << idx[pu][pv]
        if best is None or bits < best:
            best = bits

    return best or 0


# Cache for canon_key results (shared across calls)
_canon_key_cache: Dict[tuple, object] = {}


def canon_key(edges: list[tuple[int, int]], n: int) -> object:
    """Canonical key for a graph on {0..n-1} under S_n relabeling.

    Uses nauty when available (returns canonical g6 string).
    Falls back to brute-force bitset method for small n.
    """
    cache_key = (tuple(edges), n)
    if cache_key in _canon_key_cache:
        return _canon_key_cache[cache_key]

    if nauty_available():
        g6 = edgelist_to_g6(edges, n)
        result: object = canon_g6(g6)
    else:
        result = _canon_key_bruteforce_bitset(edges, n)

    _canon_key_cache[cache_key] = result
    return result


def iso_classes_with_stats(
    Vk: List[int],
    n: int,
    k: int,
    endpoints_by_level: Dict[int, List[Endpoints]],
) -> List[dict]:
    """Group level-k vertices by isomorphism class and compute statistics.

    Returns a list of dicts, one per class:
      {
        "key": canonical key,
        "rep": representative vertex id,
        "freq": number of labeled copies,
        "aut": |Aut(H)|,
        "orbit": size of S_n orbit,
        "coeff": freq / orbit (fiber coefficient),
        "edges_rep": representative edge list,
      }
    """
    buckets: Dict[object, dict] = {}

    for v in Vk:
        edges = expand_to_simple_base_edges_id(v, k, endpoints_by_level)
        key = canon_key(edges, n)

        if key not in buckets:
            buckets[key] = {"key": key, "rep": v, "freq": 0, "edges_rep": edges}
        buckets[key]["freq"] += 1

    out = []
    for b in buckets.values():
        edges = b["edges_rep"]
        aut = aut_size_edges(edges, n)
        orbit = math.factorial(n) // aut
        freq = b["freq"]
        coeff = freq // orbit if orbit else 0

        out.append({
            "key": b["key"],
            "rep": b["rep"],
            "freq": freq,
            "aut": aut,
            "orbit": orbit,
            "coeff": coeff,
            "edges_rep": edges,
        })

    out.sort(key=lambda d: (-d["orbit"], -d["freq"], str(d["key"])))
    return out


def reps_by_graph_iso_ids(
    Vk: List[int],
    n: int,
    k: int,
    endpoints_by_level: Dict[int, List[Endpoints]],
) -> List[int]:
    """Return one representative vertex per isomorphism class at level k."""
    seen: Set[object] = set()
    reps: List[int] = []

    for v in Vk:
        edges = expand_to_simple_base_edges_id(v, k, endpoints_by_level)
        key = canon_key(edges, n)
        if key not in seen:
            seen.add(key)
            reps.append(v)

    return reps
