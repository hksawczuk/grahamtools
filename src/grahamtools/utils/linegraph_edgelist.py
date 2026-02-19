from __future__ import annotations

from collections import defaultdict


def line_graph_edgelist(
    edges: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], int]:
    """Compute the line graph from an edge list.

    Each edge in the input becomes a vertex in the line graph.
    Two line-graph vertices are adjacent iff the corresponding
    original edges share an endpoint.

    Returns
    -------
    new_edges : list[tuple[int, int]]
        Sorted edge list of the line graph (vertex ids are edge indices).
    n : int
        Number of vertices in the line graph (= len(edges)).
    """
    m = len(edges)
    if m == 0:
        return [], 0

    incident: dict[int, list[int]] = defaultdict(list)
    for idx, (u, v) in enumerate(edges):
        incident[u].append(idx)
        incident[v].append(idx)

    new_edges: set[tuple[int, int]] = set()
    for inc in incident.values():
        for i in range(len(inc)):
            for j in range(i + 1, len(inc)):
                a, b = inc[i], inc[j]
                new_edges.add((a, b) if a < b else (b, a))

    return sorted(new_edges), m


def _is_star(edges: list[tuple[int, int]]) -> int | None:
    """Return r if edges form K_{1,r}, else None."""
    if not edges:
        return None
    deg: dict[int, int] = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    degs = sorted(deg.values(), reverse=True)
    m = len(edges)
    if degs[0] == m and all(d == 1 for d in degs[1:]):
        return m
    return None


def _gamma_sequence_star(r: int, max_k: int) -> list[int]:
    """Analytical gamma sequence for K_{1,r}.

    L(K_{1,r}) = K_r which is (r-1)-regular.
    For regular graphs: |V(L(G))| = |V|*d/2, degree of L(G) = 2d-2.
    """
    seq = [r + 1]  # gamma_0 = |V(K_{1,r})| = r + 1
    if max_k < 1:
        return seq

    v = r          # |V(K_r)| = r
    d = r - 1      # K_r is (r-1)-regular
    seq.append(v)  # gamma_1

    for _ in range(2, max_k + 1):
        e = v * d // 2   # |E| of current graph
        v = e             # |V| of next line graph
        d = 2 * d - 2    # degree of next (regular stays regular)
        seq.append(v)

    return seq


def gamma_sequence_edgelist(
    edges: list[tuple[int, int]],
    max_k: int,
    *,
    max_edges: int = 10_000_000,
) -> list[int | None]:
    """Compute gamma_0, ..., gamma_max_k where gamma_k = |V(L^k(G))|.

    Uses an analytical formula for star graphs.
    Returns None for grades that exceed the max_edges threshold.
    """
    # Star shortcut
    r = _is_star(edges)
    if r is not None:
        return _gamma_sequence_star(r, max_k)

    if not edges:
        return [0] * (max_k + 1)

    # Relabel to 0..n-1
    verts = set()
    for u, v in edges:
        verts.add(u)
        verts.add(v)

    v_map = {v: i for i, v in enumerate(sorted(verts))}
    current_edges: list[tuple[int, int]] = sorted(
        (min(v_map[u], v_map[v]), max(v_map[u], v_map[v]))
        for u, v in edges
    )

    seq: list[int | None] = [len(verts)]

    for _ in range(1, max_k + 1):
        gamma_k = len(current_edges)
        seq.append(gamma_k)

        if gamma_k == 0:
            while len(seq) <= max_k:
                seq.append(0)
            break

        # Estimate next size before building
        inc: dict[int, int] = defaultdict(int)
        for u, v in current_edges:
            inc[u] += 1
            inc[v] += 1
        est_next = sum(d * (d - 1) // 2 for d in inc.values())

        if est_next > max_edges:
            while len(seq) <= max_k:
                seq.append(None)
            break

        new_edges, _ = line_graph_edgelist(current_edges)
        current_edges = new_edges

    return seq
