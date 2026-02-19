#!/usr/bin/env python3
"""
Compute count(τ, Q_d) for small trees τ and small d by direct enumeration.
Check whether Q_d separates trees that K_{n,m} cannot (same bipartition).

Q_d: vertices = {0,1}^d, edges connect vertices differing in 1 bit.
d-regular, 2^d vertices, bipartite.

Usage: python3 count_Qd.py [max_d]
"""

import sys
from itertools import combinations
from collections import defaultdict

from grahamtools.utils.automorphisms import aut_size_edges


def qd_adj(v1, v2):
    """Check if two integers (as bit vectors) differ in exactly 1 bit."""
    x = v1 ^ v2
    return x != 0 and (x & (x - 1)) == 0


def count_subgraph_in_Qd(tree_edges, d):
    """Count copies of tree (given by edge list on {0..nv-1}) in Q_d.
    Brute force: try all injections of V(tree) into V(Q_d)."""
    verts = set()
    for u, v in tree_edges:
        verts.add(u)
        verts.add(v)
    verts = sorted(verts)
    nv = len(verts)
    n_qd = 1 << d

    if nv > n_qd:
        return 0

    # By vertex-transitivity, fix first vertex to 0 and multiply by 2^d / nv
    # But some trees have non-trivial vertex orbits, so safer to fix v0 -> 0
    # and multiply by 2^d, then divide by nv * |Aut| ... actually just do
    # full brute force for tiny cases, or root-based for larger.

    # For small d, just enumerate all injections from verts to V(Q_d)
    # with first vertex fixed to 0 (vertex-transitivity).
    # Count rooted labeled embeddings, then:
    # count = 2^d * (rooted_count) / |Aut(tree)| ... no, that's not right either.
    #
    # Simplest correct approach: count all injective maps V(tree) -> V(Q_d)
    # preserving adjacency, divide by |Aut(tree)|.
    #
    # For efficiency, use vertex-transitivity: fix verts[0] -> 0,
    # enumerate remaining, multiply by 2^d at end, divide by (nv * |Aut|)...
    # No, that overcounts if tree has vertex orbits.
    #
    # Just use: fix verts[0] -> 0, count all completions, multiply by 2^d.
    # This counts each labeled embedding exactly nv times (once per choice of
    # which tree vertex maps to 0)... No, only if tree is vertex-transitive.
    #
    # Safest: full brute force for small d.

    # For d <= 6 (64 vertices), enumerate via backtracking
    adj_list = defaultdict(list)
    for u, v in tree_edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    # Order vertices by BFS from verts[0]
    order = []
    visited = set()
    queue = [verts[0]]
    visited.add(verts[0])
    parent = {verts[0]: None}
    while queue:
        v = queue.pop(0)
        order.append(v)
        for u in adj_list[v]:
            if u not in visited:
                visited.add(u)
                queue.append(u)
                parent[u] = v

    # For each vertex in BFS order, it must be adjacent to its parent's image
    # and distinct from all previously assigned images.
    count = 0

    def backtrack(idx, assignment):
        nonlocal count
        if idx == nv:
            count += 1
            return

        v = order[idx]
        p = parent[v]

        if p is None:
            # Root: try all Q_d vertices
            candidates = range(n_qd)
        else:
            # Must be adjacent to parent's image in Q_d
            p_img = assignment[p]
            # Neighbors of p_img in Q_d: flip each bit
            candidates = [p_img ^ (1 << b) for b in range(d)]

        used = set(assignment.values())
        for c in candidates:
            if c in used:
                continue
            # Check adjacency with ALL already-assigned neighbors (not just parent)
            ok = True
            for u in adj_list[v]:
                if u in assignment:
                    if not qd_adj(c, assignment[u]):
                        ok = False
                        break
            if ok:
                assignment[v] = c
                backtrack(idx + 1, assignment)
                del assignment[v]

    backtrack(0, {})

    # count = number of labeled embeddings = |Aut(τ)| * count(τ, Q_d)
    # So count(τ, Q_d) = count / |Aut(τ)|
    # But we want to return the raw labeled count for now and divide later
    return count


def aut_size(edges):
    """Compute |Aut| using grahamtools."""
    n = max(v for e in edges for v in e) + 1 if edges else 1
    return aut_size_edges(edges, n)


def main():
    max_d = int(sys.argv[1]) if len(sys.argv) > 1 else 6

    # Define collision groups (trees with same bipartition)
    collision_groups = {
        "(2,3)": [
            ("P5", [(0,1),(1,2),(2,3),(3,4)]),
            ("fork", [(0,1),(0,2),(0,3),(1,4)]),
        ],
        "(3,3)": [
            ("P6", [(0,1),(1,2),(2,3),(3,4),(4,5)]),
            ("catA", [(0,1),(1,2),(2,3),(0,4),(0,5)]),
            ("dblstar", [(0,1),(0,2),(0,3),(3,4),(3,5)]),
        ],
        "(2,4)": [
            ("T6_spider", [(0,1),(0,2),(0,3),(0,4),(1,5)]),
            ("T6_cat", [(0,1),(1,2),(2,3),(3,4),(0,5)]),
        ],
        "(3,4)": [
            ("P7", [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6)]),
            ("T7a", [(0,1),(1,2),(2,3),(0,4),(0,5),(4,6)]),
            ("T7b", [(0,1),(0,2),(0,3),(1,4),(1,5),(2,6)]),
            ("T7c", [(0,1),(0,2),(0,3),(1,4),(4,5),(4,6)]),
            ("T7d", [(0,1),(0,2),(0,3),(3,4),(3,5),(1,6)]),
            ("T7e", [(0,1),(0,2),(0,3),(0,4),(1,5),(2,6)]),
            ("T7f", [(0,1),(0,2),(0,3),(0,4),(4,5),(5,6)]),
        ],
    }

    print(f"Computing count(τ, Q_d) for d = 1..{max_d}\n")

    for group_name, trees in collision_groups.items():
        print(f"{'='*60}")
        print(f"  Bipartition {group_name}")
        print(f"{'='*60}")

        # Compute |Aut| for each tree
        auts = {}
        for name, edges in trees:
            auts[name] = aut_size(edges)

        # Compute counts
        header = f"  {'d':>3s}"
        for name, _ in trees:
            header += f"  {name:>12s}"
        print(header)

        for d in range(1, max_d + 1):
            row = f"  {d:>3d}"
            for name, edges in trees:
                raw = count_subgraph_in_Qd(edges, d)
                cnt = raw // auts[name]
                row += f"  {cnt:>12d}"
            print(row)

        # Check if columns are linearly independent (pairwise non-proportional)
        print(f"\n  Proportionality check (ratio col_i / col_j at each d):")
        for i in range(len(trees)):
            for j in range(i+1, len(trees)):
                ni, ei = trees[i]
                nj, ej = trees[j]
                ratios = []
                for d in range(1, max_d + 1):
                    ci = count_subgraph_in_Qd(ei, d) // auts[ni]
                    cj = count_subgraph_in_Qd(ej, d) // auts[nj]
                    if cj == 0:
                        ratios.append("inf" if ci != 0 else "0/0")
                    elif ci == 0:
                        ratios.append("0")
                    else:
                        from fractions import Fraction
                        ratios.append(str(Fraction(ci, cj)))
                print(f"    {ni}/{nj}: {ratios}")
        print()


if __name__ == "__main__":
    main()