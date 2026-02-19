"""
Iterated line graphs of complete binary trees T_{2,d} for d=3,4,5, k=0..6.

At each iteration, classify every edge by the sorted degree pair (deg(u), deg(v))
of its endpoints and print the counts.
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from typing import List, Sequence, Tuple

import networkx as nx

sys.path.insert(0, "/Users/hamiltonsawczuk/grahamtools/.claude/worktrees/amazing-kilby/src")

from grahamtools.linegraph.adjlist import line_graph_adj, edges_from_adj


def complete_binary_tree_adj(depth: int) -> List[List[int]]:
    G = nx.balanced_tree(r=2, h=depth)
    n = G.number_of_nodes()
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in G.edges():
        adj[u].append(v)
        adj[v].append(u)
    return [sorted(neigh) for neigh in adj]


def edge_count(adj: Sequence[Sequence[int]]) -> int:
    return len(edges_from_adj(adj))


def edge_type_counts(adj: Sequence[Sequence[int]]) -> dict[Tuple[int, int], int]:
    """For each edge, classify by sorted (deg(u), deg(v)). Return counts."""
    degs = [len(neigh) for neigh in adj]
    c: Counter[Tuple[int, int]] = Counter()
    n = len(adj)
    for u, neigh in enumerate(adj):
        for v in neigh:
            if v > u:
                pair = (min(degs[u], degs[v]), max(degs[u], degs[v]))
                c[pair] += 1
    return dict(sorted(c.items()))


def main():
    k_max = 6

    for depth in [3, 4, 5]:
        n_nodes = 2 ** (depth + 1) - 1
        n_edges = n_nodes - 1
        print(f"{'=' * 80}")
        print(f"Complete binary tree T_{{2,{depth}}}  (depth={depth}, {n_nodes} nodes, {n_edges} edges)")
        print(f"{'=' * 80}")

        adj = complete_binary_tree_adj(depth)
        cur = adj

        for k in range(k_max + 1):
            V = len(cur)
            E = edge_count(cur)

            print(f"\n  L^{k}: |V|={V:>8,}  |E|={E:>10,}")

            if V == 0 or E == 0:
                print(f"  (edgeless, stopping)")
                break

            et = edge_type_counts(cur)
            # Print edge types in a compact table
            print(f"  {'Edge type (d_u,d_v)':>22}  {'Count':>10}  {'Fraction':>10}")
            print(f"  {'-'*22}  {'-'*10}  {'-'*10}")
            for (du, dv), cnt in et.items():
                frac = cnt / E
                print(f"  {f'({du},{dv})':>22}  {cnt:>10,}  {frac:>10.4f}")

            # Safety check for next iteration
            sum_deg_sq = sum(len(neigh) ** 2 for neigh in cur)
            next_V = E
            if next_V > 5_000_000:
                print(f"\n  (next iterate would have {next_V:,} vertices — stopping)")
                break

            if k < k_max:
                t0 = time.perf_counter()
                cur = line_graph_adj(cur)
                dt = time.perf_counter() - t0
                if dt > 0.5:
                    print(f"    (computed in {dt:.2f}s)")

        print()


if __name__ == "__main__":
    main()
