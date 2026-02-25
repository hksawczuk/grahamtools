#!/usr/bin/env python3
"""
Compute Graham sequences for spider trees T(a_1, a_2, ..., a_k) via WL-1 quotient.

T(a_1, ..., a_k) is the tree with one central hub and k paths of lengths
a_1, ..., a_k emanating from it.

  - |V| = 1 + sum(a_i)
  - |E| = sum(a_i)
  - hub degree = k

Special cases:
  - T(1,1,...,1) with n ones = K_{1,n} = S_n
  - T(1,...,1, m+1) with n-1 ones = old T(n,m) star-path tree

Usage:
    python3 spider_graham.py [--max-k K]
"""

from __future__ import annotations

import argparse
import time

from grahamtools.quotient import QuotientGraph, graham_sequence_wl1


# ------------------------------------------------------------------
#  Spider tree construction
# ------------------------------------------------------------------

def spider_adj(arms: tuple[int, ...]) -> list[list[int]]:
    """Build adjacency list for T(a_1, ..., a_k).

    Vertex layout:
      0 = hub
      Then for each arm i, a_i vertices numbered consecutively.

    Returns sorted adjacency lists.
    """
    nv = 1 + sum(arms)
    adj: list[list[int]] = [[] for _ in range(nv)]

    def add(u: int, v: int) -> None:
        adj[u].append(v)
        adj[v].append(u)

    next_v = 1
    for length in arms:
        # First vertex of this arm connects to hub.
        add(0, next_v)
        # Chain the rest of the arm.
        for j in range(length - 1):
            add(next_v + j, next_v + j + 1)
        next_v += length

    return [sorted(neigh) for neigh in adj]


def spider_label(arms: tuple[int, ...]) -> str:
    """Human-readable label like T(1,1,1,2)."""
    return "T(" + ",".join(str(a) for a in arms) + ")"


# ------------------------------------------------------------------
#  Compute Graham sequence with class tracking
# ------------------------------------------------------------------

def compute(arms: tuple[int, ...], max_k: int) -> tuple[list[int], list[int], float]:
    """Return (gamma_sequence, class_counts, elapsed_seconds)."""
    adj = spider_adj(arms)
    t0 = time.time()

    q = QuotientGraph.from_adj(adj)
    seq = [q.num_vertices]
    classes = [q.num_classes]

    for _ in range(max_k):
        if q.num_edges == 0:
            seq.append(0)
            classes.append(0)
            break
        q = q.line_graph_quotient().compress()
        seq.append(q.num_vertices)
        classes.append(q.num_classes)

    elapsed = time.time() - t0
    return seq, classes, elapsed


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Graham sequences for spider trees T(a_1,...,a_k) via WL-1 quotient"
    )
    parser.add_argument(
        "--max-k", type=int, default=7,
        help="maximum line-graph iteration depth (default: 7)",
    )
    args = parser.parse_args()
    max_k: int = args.max_k

    # ------------------------------------------------------------------
    #  Comparison groups
    # ------------------------------------------------------------------

    groups: list[tuple[str, list[tuple[int, ...]]]] = [
        # Same edge count comparisons: sum(a_i) constant
        ("Same edge count = 6", [
            (1, 1, 1, 1, 1, 1),  # S_6 = K_{1,6}
            (1, 1, 1, 1, 2),     # 5 arms, one longer
            (1, 1, 1, 3),        # 4 arms
            (1, 1, 2, 2),        # 4 arms, two length-2
            (1, 2, 3),           # 3 arms
            (2, 2, 2),           # 3 equal arms
            (1, 1, 4),           # 3 arms
            (1, 5),              # 2 arms
            (3, 3),              # 2 equal arms
            (6,),                # path P_7
        ]),

        ("Same edge count = 8", [
            (1, 1, 1, 1, 1, 1, 1, 1),  # S_8
            (1, 1, 1, 1, 1, 1, 2),
            (1, 1, 1, 1, 2, 2),
            (2, 2, 2, 2),
            (1, 1, 1, 5),
            (1, 1, 2, 4),
            (1, 3, 4),
            (2, 2, 4),
            (4, 4),
            (1, 7),
            (8,),
        ]),

        ("Same hub degree = 4, varying arm lengths", [
            (1, 1, 1, 1),  # S_4
            (1, 1, 1, 2),
            (1, 1, 2, 2),
            (1, 1, 1, 3),
            (2, 2, 2, 2),
            (1, 1, 3, 3),
            (1, 2, 3, 4),
            (2, 2, 3, 3),
            (3, 3, 3, 3),
        ]),
    ]

    for group_name, spiders in groups:
        print("=" * 78)
        print(f"  {group_name}")
        print("=" * 78)

        results: list[tuple[tuple[int, ...], list[int], list[int]]] = []

        for arms in spiders:
            seq, classes, elapsed = compute(arms, max_k)
            label = spider_label(arms)
            e = sum(arms)
            deg = len(arms)
            print(
                f"\n  {label:>20s}: {e} edges, hub deg {deg}  "
                f"({elapsed:.3f}s, max {max(classes)} classes)"
            )
            print(f"    {'classes:':>10s} {classes}")
            print(f"    {'gamma:':>10s} {seq}")
            results.append((arms, seq, classes))

        # Summary table for this group.
        max_k_actual = 0
        for _, seq, _ in results:
            for k in range(len(seq)):
                if seq[k] > 0:
                    max_k_actual = max(max_k_actual, k)

        max_val = max(
            seq[k]
            for _, seq, _ in results
            for k in range(min(len(seq), max_k_actual + 1))
        )
        col_w = max(len(f"{max_val:,}"), 6) + 2

        print(f"\n  {'':>20s}", end="")
        for k in range(max_k_actual + 1):
            print(f"  {'k=' + str(k):>{col_w}s}", end="")
        print()
        print("  " + "-" * (22 + (col_w + 2) * (max_k_actual + 1)))

        for arms, seq, _ in results:
            label = spider_label(arms)
            print(f"  {label:>20s}", end="")
            for k in range(max_k_actual + 1):
                val = seq[k] if k < len(seq) else ""
                if isinstance(val, int):
                    print(f"  {val:>{col_w},}", end="")
                else:
                    print(f"  {'':>{col_w}s}", end="")
            print()

        print()


if __name__ == "__main__":
    main()
