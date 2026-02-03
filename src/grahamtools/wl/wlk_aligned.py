from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

import networkx as nx


TupleK = Tuple[Hashable, ...]


def _initial_tuple_signature(G: nx.Graph, t: TupleK) -> Tuple[Tuple[int, ...], ...]:
    """
    Initial WL-k signature for ordered tuple t:
    encodes equality and adjacency pattern among coordinates.

    Returns a k x k matrix over {0,1,2} as a tuple of rows:
      0 if i==j (same vertex),
      1 if edge between t[i], t[j],
      2 otherwise.
    """
    k = len(t)
    M: List[Tuple[int, ...]] = []
    for i in range(k):
        row: List[int] = []
        for j in range(k):
            if t[i] == t[j]:
                row.append(0)
            else:
                row.append(1 if G.has_edge(t[i], t[j]) else 2)
        M.append(tuple(row))
    return tuple(M)


def _all_ordered_k_tuples(nodes: Sequence[Hashable], k: int) -> List[TupleK]:
    """
    Deterministic ordered k-tuples in lex order.
    Warning: size is n^k.
    """
    nodes = list(nodes)
    return list(itertools.product(nodes, repeat=k))


def _shared_compress(sigA: Dict[TupleK, object], sigB: Dict[TupleK, object]) -> Tuple[Dict[TupleK, int], Dict[TupleK, int], int]:
    """
    Assign integer color ids using the UNION of signatures across both graphs.
    This makes 'color i' consistent between A and B.
    """
    uniq = sorted(set(sigA.values()) | set(sigB.values()))
    mp = {s: i for i, s in enumerate(uniq)}
    colA = {t: mp[sigA[t]] for t in sigA}
    colB = {t: mp[sigB[t]] for t in sigB}
    return colA, colB, len(uniq)


@dataclass(frozen=True)
class WLKAlignedResult:
    """
    Result of aligned WL-k refinement on two graphs.
    """
    k: int
    rounds: int
    distinguishable: bool
    witness_color: Optional[int]
    histA: Dict[int, int]
    histB: Dict[int, int]


def wlk_tuple_coloring_aligned(
    GA: nx.Graph,
    GB: nx.Graph,
    k: int,
    *,
    max_iter: int = 50,
) -> Tuple[Dict[TupleK, int], Dict[TupleK, int], int]:
    """
    Compute WL-k tuple coloring on GA and GB in lockstep with shared compression.
    Returns:
      (colA, colB, rounds)
    where colA/colB map ordered k-tuples to shared color ids.

    NOTE: This is expensive; intended for small graphs.
    """
    if k <= 0:
        raise ValueError("k must be >= 1.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    nodesA = sorted(GA.nodes())
    nodesB = sorted(GB.nodes())

    tuplesA = _all_ordered_k_tuples(nodesA, k)
    tuplesB = _all_ordered_k_tuples(nodesB, k)

    # Initial signatures
    sigA = {t: _initial_tuple_signature(GA, t) for t in tuplesA}
    sigB = {t: _initial_tuple_signature(GB, t) for t in tuplesB}
    colA, colB, numc = _shared_compress(sigA, sigB)

    rounds = 0
    for _ in range(max_iter):
        rounds += 1

        newSigA: Dict[TupleK, object] = {}
        for t in tuplesA:
            parts: List[object] = [colA[t]]
            t_list = list(t)
            for i in range(k):
                counts = Counter()
                old = t_list[i]
                for v in nodesA:
                    t_list[i] = v
                    counts[colA[tuple(t_list)]] += 1
                t_list[i] = old
                parts.append(tuple(sorted(counts.items())))
            newSigA[t] = tuple(parts)

        newSigB: Dict[TupleK, object] = {}
        for t in tuplesB:
            parts = [colB[t]]
            t_list = list(t)
            for i in range(k):
                counts = Counter()
                old = t_list[i]
                for v in nodesB:
                    t_list[i] = v
                    counts[colB[tuple(t_list)]] += 1
                t_list[i] = old
                parts.append(tuple(sorted(counts.items())))
            newSigB[t] = tuple(parts)

        newColA, newColB, new_numc = _shared_compress(newSigA, newSigB)

        stableA = all(newColA[t] == colA[t] for t in tuplesA)
        stableB = all(newColB[t] == colB[t] for t in tuplesB)
        if new_numc == numc and stableA and stableB:
            return newColA, newColB, rounds

        colA, colB, numc = newColA, newColB, new_numc

    return colA, colB, rounds


def wlk_distinguishable_aligned(
    GA: nx.Graph,
    GB: nx.Graph,
    k: int,
    *,
    max_iter: int = 50,
) -> Tuple[bool, WLKAlignedResult]:
    """
    WL-k distinguishes iff the stabilized histograms of tuple colors differ.
    Uses aligned (shared) color ids.

    Returns:
      (distinguishable, result)
    """
    colA, colB, rounds = wlk_tuple_coloring_aligned(GA, GB, k, max_iter=max_iter)
    histA = Counter(colA.values())
    histB = Counter(colB.values())
    dist = histA != histB

    witness = None
    if dist:
        for c in sorted(set(histA) | set(histB)):
            if histA.get(c, 0) != histB.get(c, 0):
                witness = c
                break

    result = WLKAlignedResult(
        k=k,
        rounds=rounds,
        distinguishable=dist,
        witness_color=witness,
        histA=dict(histA),
        histB=dict(histB),
    )
    return dist, result


def vertex_colors_from_diagonal_aligned(col: Dict[TupleK, int], nodes: Iterable[Hashable], k: int) -> Dict[Hashable, int]:
    """
    Induce vertex colors from diagonal tuples (v,v,...,v).
    Requires that col is a full WL-k tuple-color map for ordered k-tuples.
    """
    nodes = sorted(nodes)
    return {v: col[tuple([v] * k)] for v in nodes}


def bucket_tuples_by_color(col: Dict[TupleK, int]) -> Dict[int, List[TupleK]]:
    """
    Helper: bucket ordered k-tuples by color id.
    """
    buckets: Dict[int, List[TupleK]] = defaultdict(list)
    for t, c in col.items():
        buckets[c].append(t)
    for c in buckets:
        buckets[c].sort()
    return buckets
