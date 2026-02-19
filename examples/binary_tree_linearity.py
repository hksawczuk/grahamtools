"""
Check whether γ_k(T_{2,d}) = |V(L^k(T_{2,d}))| is linear in 2^d for fixed k.

For each k, compute γ_k across depths d=2..8 and check the ratio γ_k / 2^d.
If it converges to a constant, the growth is linear in 2^d.
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
    G = nx.balanced_tree(r=2, h=depth)
    n = G.number_of_nodes()
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in G.edges():
        adj[u].append(v)
        adj[v].append(u)
    return [sorted(neigh) for neigh in adj]


def edge_count(adj: Sequence[Sequence[int]]) -> int:
    return len(edges_from_adj(adj))


def main():
    k_max = 6
    depths = list(range(2, 9))  # d = 2..8

    # gamma[k][d] = |V(L^k(T_{2,d}))|
    # edges[k][d] = |E(L^k(T_{2,d}))|
    gamma: dict[int, dict[int, int]] = {}
    edge_data: dict[int, dict[int, int]] = {}

    for depth in depths:
        adj = complete_binary_tree_adj(depth)
        cur = adj

        for k in range(k_max + 1):
            V = len(cur)
            E = edge_count(cur)

            gamma.setdefault(k, {})[depth] = V
            edge_data.setdefault(k, {})[depth] = E

            if V == 0 or E == 0:
                break

            # Safety: skip if next iterate too large
            sum_deg_sq = sum(len(neigh) ** 2 for neigh in cur)
            next_V = E
            if next_V > 5_000_000:
                break

            if k < k_max:
                t0 = time.perf_counter()
                cur = line_graph_adj(cur)
                dt = time.perf_counter() - t0
                if dt > 0.5:
                    print(f"  d={depth}, k={k+1}: computed in {dt:.2f}s")

    # Print vertex count table
    print("=" * 90)
    print("γ_k(T_{2,d}) = |V(L^k(T_{2,d}))|")
    print("=" * 90)
    header = f"{'k \\ d':>6}" + "".join(f"{d:>12}" for d in depths)
    print(header)
    print("-" * len(header))
    for k in range(k_max + 1):
        row = f"{k:>6}"
        for d in depths:
            v = gamma.get(k, {}).get(d)
            row += f"{v:>12,}" if v is not None else f"{'—':>12}"
        print(row)

    # Print edge count table
    print()
    print("=" * 90)
    print("|E(L^k(T_{2,d}))|")
    print("=" * 90)
    header = f"{'k \\ d':>6}" + "".join(f"{d:>12}" for d in depths)
    print(header)
    print("-" * len(header))
    for k in range(k_max + 1):
        row = f"{k:>6}"
        for d in depths:
            v = edge_data.get(k, {}).get(d)
            row += f"{v:>12,}" if v is not None else f"{'—':>12}"
        print(row)

    # Check linearity: γ_k / 2^d
    print()
    print("=" * 90)
    print("γ_k(T_{2,d}) / 2^d   (should converge if linear in 2^d)")
    print("=" * 90)
    header = f"{'k \\ d':>6}" + "".join(f"{d:>12}" for d in depths)
    print(header)
    print("-" * len(header))
    for k in range(k_max + 1):
        row = f"{k:>6}"
        for d in depths:
            v = gamma.get(k, {}).get(d)
            if v is not None:
                ratio = v / (2 ** d)
                row += f"{ratio:>12.4f}"
            else:
                row += f"{'—':>12}"
        print(row)

    # Check linearity: γ_k / 2^d, ratio of consecutive depths
    print()
    print("=" * 90)
    print("γ_k(T_{2,d}) / γ_k(T_{2,d-1})   (should → 2 if linear in 2^d)")
    print("=" * 90)
    header = f"{'k \\ d':>6}" + "".join(f"{d:>12}" for d in depths[1:])
    print(header)
    print("-" * len(header))
    for k in range(k_max + 1):
        row = f"{k:>6}"
        for d in depths[1:]:
            v = gamma.get(k, {}).get(d)
            v_prev = gamma.get(k, {}).get(d - 1)
            if v is not None and v_prev is not None and v_prev > 0:
                ratio = v / v_prev
                row += f"{ratio:>12.4f}"
            else:
                row += f"{'—':>12}"
        print(row)

    # Also check edge counts
    print()
    print("=" * 90)
    print("|E(L^k)| / 2^d")
    print("=" * 90)
    header = f"{'k \\ d':>6}" + "".join(f"{d:>12}" for d in depths)
    print(header)
    print("-" * len(header))
    for k in range(k_max + 1):
        row = f"{k:>6}"
        for d in depths:
            v = edge_data.get(k, {}).get(d)
            if v is not None:
                ratio = v / (2 ** d)
                row += f"{ratio:>12.4f}"
            else:
                row += f"{'—':>12}"
        print(row)


if __name__ == "__main__":
    main()
