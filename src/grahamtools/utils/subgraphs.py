from __future__ import annotations

from itertools import combinations

from grahamtools.utils.connectivity import is_connected_edges
from grahamtools.utils.canonical import (
    canonical_graph_bruteforce,
    canonical_graph_nauty,
)
from grahamtools.external.nauty import nauty_available


def _canonical(edges: list[tuple[int, int]]) -> object:
    """Return a canonical form for a small edge list.

    Relabels vertices to 0..n-1 first so that graphs on different
    vertex sets but with the same structure get the same canonical form.
    Uses nauty when available, otherwise brute-force (n <= 10).
    """
    if not edges:
        return ()
    # Relabel to compact vertex set 0..n-1
    verts = sorted({v for e in edges for v in e})
    v_map = {v: i for i, v in enumerate(verts)}
    relabeled = [(v_map[u], v_map[v]) for u, v in edges]
    if nauty_available():
        return canonical_graph_nauty(relabeled, n=len(verts))
    return canonical_graph_bruteforce(relabeled)


def enumerate_connected_subgraphs(
    edges: list[tuple[int, int]],
    *,
    max_size: int | None = None,
) -> dict[object, tuple[int, list[tuple[int, int]], int, int]]:
    """Enumerate connected subgraphs by edge subsets.

    For each isomorphism class of connected subgraphs, returns:
      canonical_form -> (count, representative_edges, n_vertices, n_edges)
    """
    n_edges = len(edges)
    if max_size is None:
        max_size = n_edges

    counts: dict[object, list] = {}

    for size in range(1, min(max_size, n_edges) + 1):
        for subset in combinations(range(n_edges), size):
            sub_edges = [edges[i] for i in subset]
            if not is_connected_edges(sub_edges):
                continue

            canon = _canonical(sub_edges)
            if canon not in counts:
                verts: set[int] = set()
                for u, v in sub_edges:
                    verts.add(u)
                    verts.add(v)
                counts[canon] = [0, sub_edges, len(verts), size]
            counts[canon][0] += 1

    return {k: (v[0], v[1], v[2], v[3]) for k, v in counts.items()}
