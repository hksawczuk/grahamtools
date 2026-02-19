from __future__ import annotations

import math
from itertools import permutations

from grahamtools.external.nauty import (
    dreadnaut_available,
    edgelist_to_g6,
    aut_size_g6,
)


def _aut_size_bruteforce(edges: list[tuple[int, int]], n: int) -> int:
    """Count automorphisms by brute force over S_n.

    Only practical for very small n (n <= 8 or so).
    """
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    count = 0
    for perm in permutations(range(n)):
        is_auto = True
        for u in range(n):
            mapped_neighbors = 0
            nbr = adj[u]
            while nbr:
                lsb = nbr & -nbr
                v = lsb.bit_length() - 1
                nbr ^= lsb
                mapped_neighbors |= 1 << perm[v]
            if mapped_neighbors != adj[perm[u]]:
                is_auto = False
                break
        if is_auto:
            count += 1
    return count


def _equitable_partition_bitset(adj: list[int], n: int) -> tuple[int, ...]:
    """WL-1 color refinement on bitset adjacency.

    Returns a tuple of color ids, one per vertex.
    Used to reduce the automorphism search space.
    """
    colors = [0] * n
    # Initial coloring by degree
    for i in range(n):
        colors[i] = bin(adj[i]).count("1")

    for _ in range(n):
        # Compute signature: (current_color, sorted neighbor colors)
        sigs = []
        for v in range(n):
            nbr_colors = []
            nbr = adj[v]
            while nbr:
                lsb = nbr & -nbr
                u = lsb.bit_length() - 1
                nbr ^= lsb
                nbr_colors.append(colors[u])
            nbr_colors.sort()
            sigs.append((colors[v], tuple(nbr_colors)))

        # Compress signatures to integer colors
        unique = sorted(set(sigs))
        sig_to_color = {s: i for i, s in enumerate(unique)}
        new_colors = [sig_to_color[s] for s in sigs]

        if new_colors == colors:
            break
        colors = new_colors

    return tuple(colors)


def _color_classes(colors: tuple[int, ...]) -> list[list[int]]:
    """Group vertices by color, sorted by (color, vertex)."""
    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for v, c in enumerate(colors):
        groups[c].append(v)
    return [groups[c] for c in sorted(groups)]


def _aut_size_wl_backtrack(edges: list[tuple[int, int]], n: int) -> int:
    """Count automorphisms using WL equitable partition + backtracking.

    The equitable partition restricts which vertices can map to which,
    pruning the search space significantly compared to brute force.
    """
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u

    colors = _equitable_partition_bitset(adj, n)
    classes = _color_classes(colors)
    classes_perms = [list(permutations(cls)) for cls in classes]

    count = 0
    p = [-1] * n

    def backtrack(i: int) -> None:
        nonlocal count
        if i == len(classes):
            # Verify this is a valid automorphism
            for u in range(n):
                mapped = 0
                nbr = adj[u]
                while nbr:
                    lsb = nbr & -nbr
                    v = lsb.bit_length() - 1
                    nbr ^= lsb
                    mapped |= 1 << p[v]
                if mapped != adj[p[u]]:
                    return
            count += 1
            return

        cls = classes[i]
        for perm in classes_perms[i]:
            ok = True
            for a, b in zip(cls, perm):
                if p[a] != -1:
                    ok = False
                    break
                p[a] = b
            if ok:
                backtrack(i + 1)
            for a in cls:
                p[a] = -1

    backtrack(0)
    return count


def aut_size_edges(edges: list[tuple[int, int]], n: int) -> int:
    """Compute |Aut(G)| for a graph on vertices {0..n-1}.

    Strategy:
      1. If dreadnaut is available, use nauty (fast, exact).
      2. Otherwise, use WL equitable partition + backtracking (exact but slow).
    """
    if dreadnaut_available():
        g6 = edgelist_to_g6(edges, n)
        return aut_size_g6(g6)
    return _aut_size_wl_backtrack(edges, n)


def orbit_size_under_Sn(edges: list[tuple[int, int]], n: int) -> int:
    """Orbit size of the labeled graph under the S_n action.

    Equals n! / |Aut(G)| where G is the graph on {0..n-1} with the
    given edges.  Automorphisms include permutation of isolated vertices.
    """
    aut = aut_size_edges(edges, n)
    return math.factorial(n) // aut
