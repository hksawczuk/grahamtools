from __future__ import annotations

from collections import defaultdict


def is_connected_edges(
    edges: list[tuple[int, int]],
    vertices: set[int] | None = None,
) -> bool:
    """Check whether an edge list forms a connected graph.

    If *vertices* is provided, connectivity is checked over that vertex set
    (allowing isolated vertices).  Otherwise the vertex set is inferred from
    the edges.

    Semantics for degenerate cases:
      - No edges, no vertices (or empty set) -> True  (vacuously connected)
      - No edges, one vertex               -> True
      - No edges, two or more vertices      -> False
    """
    if vertices is not None:
        verts = set(vertices)
    else:
        verts = set()
        for u, v in edges:
            verts.add(u)
            verts.add(v)

    if len(verts) <= 1:
        return True

    adj: dict[int, set[int]] = defaultdict(set)
    for u, v in edges:
        if u in verts and v in verts:
            adj[u].add(v)
            adj[v].add(u)

    start = next(iter(verts))
    visited: set[int] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for nbr in adj[node]:
            if nbr not in visited:
                stack.append(nbr)
    return len(visited) == len(verts)


def connected_components_edges(
    edges: list[tuple[int, int]],
) -> list[tuple[set[int], list[tuple[int, int]]]]:
    """Return connected components as (vertex_set, edge_list) pairs."""
    adj: dict[int, set[int]] = defaultdict(set)
    verts: set[int] = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        verts.add(u)
        verts.add(v)

    remaining = set(verts)
    components: list[tuple[set[int], list[tuple[int, int]]]] = []

    while remaining:
        start = next(iter(remaining))
        comp_verts: set[int] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in comp_verts:
                continue
            comp_verts.add(node)
            for nbr in adj[node]:
                if nbr not in comp_verts:
                    stack.append(nbr)
        comp_edges = [(u, v) for u, v in edges if u in comp_verts and v in comp_verts]
        components.append((comp_verts, comp_edges))
        remaining -= comp_verts

    return components
