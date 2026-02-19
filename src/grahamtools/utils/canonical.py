from __future__ import annotations

from itertools import permutations

from grahamtools.external.nauty import nauty_available, canon_g6, edgelist_to_g6


def canonical_graph_bruteforce(
    edges: list[tuple[int, int]],
    vertices: set[int] | None = None,
) -> tuple[tuple[int, int], ...]:
    """Canonical form via min over all vertex permutations.

    Only practical for small graphs (n <= 10).
    Raises ValueError for n > 10; use canonical_graph_nauty instead.

    Returns a sorted tuple of (u, v) edge pairs with u < v, representing
    the lexicographically smallest relabeling.
    """
    if not edges:
        return ()

    if vertices is None:
        vertices = set()
        for u, v in edges:
            vertices.add(u)
            vertices.add(v)

    vlist = sorted(vertices)
    n = len(vlist)

    if n > 10:
        raise ValueError(
            f"Brute-force canonicalization is impractical for n={n}. "
            "Use canonical_graph_nauty() instead."
        )

    best: tuple[tuple[int, int], ...] | None = None
    for perm in permutations(range(n)):
        v_map = {vlist[i]: perm[i] for i in range(n)}
        relabeled = tuple(sorted(
            (min(v_map[u], v_map[v]), max(v_map[u], v_map[v]))
            for u, v in edges
        ))
        if best is None or relabeled < best:
            best = relabeled
    return best  # type: ignore[return-value]


def canonical_graph_nauty(
    edges: list[tuple[int, int]],
    n: int | None = None,
) -> str:
    """Canonical form via nauty shortg. Returns a canonical graph6 string.

    If *n* is not given, it is inferred from the edges (max vertex + 1).
    Raises RuntimeError if nauty is not available.
    """
    if not nauty_available():
        raise RuntimeError(
            "nauty not available for canonical_graph_nauty. "
            "Install nauty (geng, shortg) or use canonical_graph_bruteforce."
        )
    if not edges:
        if n is None:
            n = 0
        return edgelist_to_g6([], n)

    if n is None:
        n = max(max(u, v) for u, v in edges) + 1
    g6 = edgelist_to_g6(edges, n)
    return canon_g6(g6)
