from __future__ import annotations

from typing import List, Sequence, Tuple


def edges_from_adj(adj: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    """
    Return undirected edges as (u,v) with u < v.
    """
    eds: List[Tuple[int, int]] = []
    for u, neigh in enumerate(adj):
        for v in neigh:
            if v > u:
                eds.append((u, v))
    return eds


def is_regular(adj: Sequence[Sequence[int]]) -> tuple[bool, int]:
    """
    Returns (is_regular, degree_if_regular_else_-1).
    Empty graph is considered regular of degree 0.
    """
    n = len(adj)
    if n == 0:
        return True, 0
    d0 = len(adj[0])
    for u in range(1, n):
        if len(adj[u]) != d0:
            return False, -1
    return True, d0


def line_graph_adj(adj: Sequence[Sequence[int]]) -> List[List[int]]:
    """
    Build adjacency list of the line graph L(H) using an edge-index method.

    Vertices of L(H) correspond to edges of H.
    Two are adjacent iff the original edges share an endpoint.

    Complexity: O(m + sum_v deg(v)^2).
    """
    n = len(adj)
    eds = edges_from_adj(adj)
    m = len(eds)
    if m == 0:
        return []

    incident: List[List[int]] = [[] for _ in range(n)]
    for ei, (u, v) in enumerate(eds):
        incident[u].append(ei)
        incident[v].append(ei)

    ladj_sets = [set() for _ in range(m)]
    for inc in incident:
        L = len(inc)
        # clique among edges incident to this vertex
        for i in range(L):
            ei = inc[i]
            for j in range(i + 1, L):
                ej = inc[j]
                ladj_sets[ei].add(ej)
                ladj_sets[ej].add(ei)

    return [sorted(s) for s in ladj_sets]
