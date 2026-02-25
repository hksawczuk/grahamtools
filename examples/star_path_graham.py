#!/usr/bin/env python3
"""
Compute Graham sequences for star-path trees T(n,m) via WL-1 quotient.

T(n,m) is the tree formed by taking the star K_{1,n} and identifying
one leaf with an endpoint of the path P_m (m edges, m+1 vertices).

  - T(n,m) has n+m edges and n+m+1 vertices.
  - T(n,0) = K_{1,n}.

The WL-1 quotient exploits the S_{n-1} symmetry acting on the free
leaves, keeping the representation at roughly m+2 classes regardless
of n.  This lets us push to much higher grades than brute-force line
graph construction.

Usage:
    python3 star_path_graham.py [--max-k K] [--verify]
"""

from __future__ import annotations

import argparse
import sys
import time
from math import prod

from grahamtools.quotient import QuotientGraph, graham_sequence_wl1


# ------------------------------------------------------------------
#  T(n,m) construction
# ------------------------------------------------------------------

def star_path_adj(n: int, m: int) -> list[list[int]]:
    """Build adjacency list for T(n,m).

    Vertex layout:
      0        = hub of the star (degree n)
      1        = junction leaf / path start (degree 1+[m>0])
      2 .. n   = free leaves (degree 1), only present when n >= 2
      n+1 .. n+m = path interior + endpoint, only present when m >= 1

    Edges:
      hub -- junction:      (0, 1)
      hub -- free leaf i:   (0, i)  for i = 2 .. n
      path:                 (1, n+1), (n+1, n+2), ..., (n+m-1, n+m)
    """
    nv = n + m + 1
    adj: list[list[int]] = [[] for _ in range(nv)]

    def add(u: int, v: int) -> None:
        adj[u].append(v)
        adj[v].append(u)

    # Star edges: hub (0) to all leaves (1 .. n)
    for i in range(1, n + 1):
        add(0, i)

    # Path edges: junction (1) to n+1, n+1 to n+2, ...
    if m >= 1:
        add(1, n + 1)
        for j in range(n + 1, n + m):
            add(j, j + 1)

    return [sorted(neigh) for neigh in adj]


# ------------------------------------------------------------------
#  Known formula for K_{1,n}
# ------------------------------------------------------------------

def gamma_star(n: int, k: int) -> int:
    """Exact gamma_k(K_{1,n}) for k >= 0.

    k=0: n+1
    k=1: n  (= number of edges)
    k>=2: via the recurrence for iterated line graphs of regular graphs.
          L(K_{1,n}) = K_n, which is (n-1)-regular on n vertices.
          For a d-regular graph on v vertices:
            L has v*d/2 vertices and is (2d-2)-regular.

    Closed form for k >= 1:
      g_k = n * prod_{j=1}^{k-1} ((n-3)*2^{j-2} + 1)

    (The product is empty for k=1, giving g_1 = n.)
    """
    if k == 0:
        return n + 1
    if n <= 1:
        return n if k == 1 else 0

    # L^1(K_{1,n}) = K_n:  n vertices, (n-1)-regular
    v = n
    d = n - 1
    for _ in range(k - 1):
        if v == 0 or d == 0:
            return 0
        v = v * d // 2
        d = 2 * d - 2
    return v


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Graham sequences for star-path trees T(n,m) via WL-1 quotient"
    )
    parser.add_argument(
        "--max-k", type=int, default=12,
        help="maximum line-graph iteration depth (default: 12)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="verify T(n,0)=K_{1,n} against known formula",
    )
    args = parser.parse_args()
    max_k: int = args.max_k

    n_values = [2, 3, 4, 5, 6]
    m_values = [0, 1, 2, 3, 4]

    # ------------------------------------------------------------------
    #  Sanity check: T(n,0) = K_{1,n}
    # ------------------------------------------------------------------

    if args.verify:
        print("=" * 70)
        print("  Sanity check: T(n,0) = K_{1,n} vs known formula")
        print("=" * 70)

        all_ok = True
        for n in n_values:
            adj = star_path_adj(n, 0)
            seq = graham_sequence_wl1(adj, max_k)
            expected = [gamma_star(n, k) for k in range(len(seq))]

            match = seq == expected
            status = "OK" if match else "MISMATCH"
            if not match:
                all_ok = False

            print(f"\n  T({n},0) = K_{{1,{n}}}:  {status}")
            print(f"    WL1:      {seq}")
            print(f"    Formula:  {expected}")

        print(f"\n  {'All checks passed.' if all_ok else 'SOME CHECKS FAILED!'}\n")

    # ------------------------------------------------------------------
    #  Compute all T(n,m)
    # ------------------------------------------------------------------

    print("=" * 70)
    print("  Graham sequences for T(n,m) via WL-1 quotient")
    print(f"  n = {n_values}, m = {m_values}, max_k = {max_k}")
    print("=" * 70)

    results: dict[tuple[int, int], list[int]] = {}
    quotient_info: dict[tuple[int, int], list[int]] = {}

    for n in n_values:
        for m in m_values:
            adj = star_path_adj(n, m)
            t0 = time.time()

            # Track quotient class counts through iterations.
            q = QuotientGraph.from_adj(adj)
            seq = [q.num_vertices]
            class_counts = [q.num_classes]

            for _ in range(max_k):
                if q.num_edges == 0:
                    seq.append(0)
                    class_counts.append(0)
                    break
                q = q.line_graph_quotient().compress()
                seq.append(q.num_vertices)
                class_counts.append(q.num_classes)

            elapsed = time.time() - t0
            results[(n, m)] = seq
            quotient_info[(n, m)] = class_counts

            print(
                f"\n  T({n},{m}): {n+m} edges, {n+m+1} vertices  "
                f"({elapsed:.3f}s, max {max(class_counts)} classes)"
            )
            print(f"    classes: {class_counts}")
            print(f"    gamma:   {seq}")

    # ------------------------------------------------------------------
    #  Summary table
    # ------------------------------------------------------------------

    # Find the max k where at least one entry is nonzero.
    max_k_actual = 0
    for seq in results.values():
        for k in range(len(seq)):
            if seq[k] > 0:
                max_k_actual = max(max_k_actual, k)

    print(f"\n{'=' * 70}")
    print(f"  Summary table: gamma_k(T(n,m))")
    print(f"{'=' * 70}")

    # Column widths: adapt to the largest number.
    max_val = max(
        seq[k]
        for seq in results.values()
        for k in range(min(len(seq), max_k_actual + 1))
    )
    col_w = max(len(f"{max_val:,}"), 6) + 2

    # Header row.
    hdr = f"  {'(n,m)':>8s}"
    for k in range(max_k_actual + 1):
        hdr += f"  {'k=' + str(k):>{col_w}s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for n in n_values:
        for m in m_values:
            seq = results[(n, m)]
            row = f"  {'(' + str(n) + ',' + str(m) + ')':>8s}"
            for k in range(max_k_actual + 1):
                val = seq[k] if k < len(seq) else ""
                if isinstance(val, int):
                    row += f"  {val:>{col_w},}"
                else:
                    row += f"  {'':>{col_w}s}"
            print(row)
        print()  # blank line between n groups

    # ------------------------------------------------------------------
    #  Quotient class count table
    # ------------------------------------------------------------------

    print(f"\n{'=' * 70}")
    print(f"  Quotient class counts at each grade")
    print(f"{'=' * 70}")

    cw = 6
    hdr = f"  {'(n,m)':>8s}"
    for k in range(max_k_actual + 1):
        hdr += f"  {'k=' + str(k):>{cw}s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for n in n_values:
        for m in m_values:
            cc = quotient_info[(n, m)]
            row = f"  {'(' + str(n) + ',' + str(m) + ')':>8s}"
            for k in range(max_k_actual + 1):
                val = cc[k] if k < len(cc) else ""
                if isinstance(val, int):
                    row += f"  {val:>{cw}d}"
                else:
                    row += f"  {'':>{cw}s}"
            print(row)
        print()


if __name__ == "__main__":
    main()
