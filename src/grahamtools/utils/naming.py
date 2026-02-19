from __future__ import annotations

from collections import defaultdict


def tree_name(edges: list[tuple[int, int]], n: int | None = None) -> str:
    """Human-readable name for a tree from its edge list.

    Handles: K1, K2, P{n}, K1,{r}, fork, and general T{nv}[{deg_seq}].
    If *n* is given, the degree array is sized to n (for embedding in K_n);
    otherwise n is inferred from the edges.
    """
    if not edges:
        return "K1"

    nedges = len(edges)
    nv = nedges + 1  # a tree on m edges has m+1 vertices

    if n is None:
        n = max(max(u, v) for u, v in edges) + 1

    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    active_degs = sorted((d for d in deg if d > 0), reverse=True)
    max_d = active_degs[0] if active_degs else 0

    if nedges == 1:
        return "K2"

    # Path: all degrees <= 2
    if max_d <= 2:
        return f"P{nv}"

    # Star: one hub with degree = nedges, rest are leaves
    if active_degs.count(1) == nedges and max_d == nedges:
        return f"K1,{nedges}"

    # Fork (degree-3 vertex in a 4-edge tree on 5 vertices)
    if nv == 5 and max_d == 3 and nedges == 4:
        return "fork"

    # General: use degree sequence
    ds_str = "".join(str(d) for d in active_degs)
    return f"T{nv}[{ds_str}]"


def describe_graph(edges: list[tuple[int, int]]) -> str:
    """Human-readable description of a small graph.

    Returns recognizable names for common structures (K2, Pn, K1,r, Cn, etc.)
    and a generic descriptor with vertex/edge counts and degree sequence
    for everything else.
    """
    if not edges:
        return "empty"

    m = len(edges)
    adj: dict[int, set[int]] = defaultdict(set)
    verts: set[int] = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        verts.add(u)
        verts.add(v)
    n = len(verts)
    deg_seq = sorted((len(adj[v]) for v in verts), reverse=True)

    if m == 1:
        return "K2"

    if m == n - 1:
        # Tree
        if all(d <= 2 for d in deg_seq):
            return f"P{n}"
        if deg_seq.count(1) == n - 1:
            return f"K1,{n - 1}"
        ds_str = "".join(str(d) for d in deg_seq)
        return f"Tree({n}v,{ds_str})"

    if all(d == 2 for d in deg_seq) and m == n:
        return f"C{n}"

    if m == n:
        return f"Unicyclic({n}v,{m}e)"

    return f"Graph({n}v,{m}e)"
