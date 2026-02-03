from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from grahamtools.io.graph6 import g6_to_adjlist
from grahamtools.external.nauty import canon_g6
from grahamtools.linegraph.adjlist import is_regular, line_graph_adj


@dataclass(frozen=True)
class GrahamSignature:
    """
    Collision key for Graham-style iteration experiments.

    prefix: tuple (N0, N1, ..., Nt) where Ni = |V(L^i(G))|
            computed until stopping condition triggers.
    tail:   if we stop early because the current iterate is regular,
            record (Nt, dt) where dt is its degree.
    stopped_reason: "regular" | "k_cap" | "N_cap"
    """

    prefix: Tuple[int, ...]
    tail: Optional[Tuple[int, int]]
    stopped_reason: str


def graham_sequence_adj(adj0: Sequence[Sequence[int]], k_max: int) -> List[int]:
    """
    Graham vertex-count prefix using adjlist iteration:
      [|V(L^0)|, |V(L^1)|, ..., |V(L^{k_max})|]
    stopping early if the iterate becomes edgeless.
    """
    seq = [len(adj0)]
    cur = [list(neigh) for neigh in adj0]
    for _ in range(k_max):
        cur = line_graph_adj(cur)
        seq.append(len(cur))
        if len(cur) == 0:
            break
    return seq


def graham_sequence_g6(g6: str, k_max: int) -> List[int]:
    """
    Graham vertex-count prefix from graph6.
    """
    return graham_sequence_adj(g6_to_adjlist(g6), k_max=k_max)


def graham_signature_canon_g6(
    canon0: str,
    *,
    k_cap: int,
    N_cap: int,
    cache: Optional[Dict[str, GrahamSignature]] = None,
) -> GrahamSignature:
    """
    Compute GrahamSignature starting from a *canonical* graph6 string canon0.

    cache: optional dict for per-process memoization keyed by canon0.
    """
    if cache is not None and canon0 in cache:
        return cache[canon0]

    adj = g6_to_adjlist(canon0)
    prefix: List[int] = [len(adj)]

    for _k in range(k_cap):
        reg, d = is_regular(adj)
        if reg:
            sig = GrahamSignature(prefix=tuple(prefix), tail=(prefix[-1], d), stopped_reason="regular")
            if cache is not None:
                cache[canon0] = sig
            return sig

        adj = line_graph_adj(adj)
        prefix.append(len(adj))

        if prefix[-1] > N_cap:
            sig = GrahamSignature(prefix=tuple(prefix), tail=None, stopped_reason="N_cap")
            if cache is not None:
                cache[canon0] = sig
            return sig

    sig = GrahamSignature(prefix=tuple(prefix), tail=None, stopped_reason="k_cap")
    if cache is not None:
        cache[canon0] = sig
    return sig


def graham_signature_g6(g6: str, *, k_cap: int, N_cap: int) -> tuple[GrahamSignature, str]:
    """
    Compute (signature, canonical_g6) from an arbitrary graph6 input.
    """
    canon0 = canon_g6(g6)
    sig = graham_signature_canon_g6(canon0, k_cap=k_cap, N_cap=N_cap, cache=None)
    return sig, canon0
