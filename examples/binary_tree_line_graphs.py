"""
Iterated line graphs of complete binary trees of depth d = 3, 4, 5.

For each tree T_d, compute L^k(T_d) for k = 0..5 and print:
  - vertex count |V|
  - edge count |E|
  - degree distribution {degree: count}
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from typing import List, Sequence

import networkx as nx

sys.path.insert(0, "/Users/hamiltonsawczuk/grahamtools/.claude/worktrees/amazing-kilby/src")

from grahamtools.linegraph.adjlist import line_graph_adj, edges_from_adj


def complete_binary_tree_adj(depth: int) -> List[List[int]]:
    """Build a complete binary tree of given depth as an adjacency list.

    Depth 0 = single root node (1 node).
    Depth d = root + two subtrees of depth d-1.
    Total nodes = 2^(d+1) - 1.
    """
    G = nx.balanced_tree(r=2, h=depth)
    n = G.number_of_nodes()
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in G.edges():
        adj[u].append(v)
        adj[v].append(u)
    return [sorted(neigh) for neigh in adj]


def degree_distribution(adj: Sequence[Sequence[int]]) -> dict[int, int]:
    """Return {degree: count} sorted by degree."""
    c = Counter(len(neigh) for neigh in adj)
    return dict(sorted(c.items()))


def edge_count(adj: Sequence[Sequence[int]]) -> int:
    return len(edges_from_adj(adj))


def main():
    k_max = 6

    for depth in [3, 4, 5]:
        n_nodes = 2 ** (depth + 1) - 1
        n_edges = n_nodes - 1  # tree
        print(f"{'=' * 70}")
        print(f"Complete binary tree T_{depth}  (depth={depth}, {n_nodes} nodes, {n_edges} edges)")
        print(f"{'=' * 70}")

        adj = complete_binary_tree_adj(depth)

        cur = adj
        for k in range(k_max + 1):
            V = len(cur)
            E = edge_count(cur)
            dd = degree_distribution(cur)

            dd_str = ", ".join(f"{deg}:{cnt}" for deg, cnt in dd.items())
            print(f"  L^{k}: |V|={V:>8,}  |E|={E:>10,}  deg dist: {{{dd_str}}}")

            if V == 0 or E == 0:
                print(f"  (graph is edgeless, stopping)")
                break

            # Check if next iteration might be too large
            # |V(L(G))| = |E(G)|, |E(L(G))| = sum_v C(deg(v),2) = (sum_v deg(v)^2)/2 - |E|
            sum_deg_sq = sum(len(neigh) ** 2 for neigh in cur)
            next_V = E
            next_E = sum_deg_sq // 2 - E
            if next_V > 5_000_000:
                print(f"  (next iterate would have {next_V:,} vertices — skipping to avoid blowup)")
                break

            if k < k_max:
                t0 = time.perf_counter()
                cur = line_graph_adj(cur)
                dt = time.perf_counter() - t0
                if dt > 0.1:
                    print(f"    (computed in {dt:.2f}s)")

        print()


if __name__ == "__main__":
    main()
