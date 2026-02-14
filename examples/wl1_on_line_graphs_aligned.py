#!/usr/bin/env python3
"""
Run WL-1 (via existing aligned WL-k with k=1) on L^t(E`dg) and L^t(E`ow).

Usage:
  python3 examples/wl1_on_line_graphs_aligned.py --iter 1
  python3 examples/wl1_on_line_graphs_aligned.py --iter 2
"""

from __future__ import annotations

import argparse
from collections import Counter
import networkx as nx

from grahamtools.io.graph6 import g6_to_nx
from grahamtools.wl.wlk_aligned import wlk_tuple_coloring_aligned


def line_iterate(G: nx.Graph, t: int) -> nx.Graph:
    H = G
    for _ in range(t):
        if H.number_of_edges() == 0:
            break
        H = nx.line_graph(H)
    return H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g6A", default="E`dg")
    ap.add_argument("--g6B", default="E`ow")
    ap.add_argument("--iter", type=int, default=1, help="compare L^iter(A) vs L^iter(B); iter=1 means line graphs")
    ap.add_argument("--max-iter", type=int, default=50, help="max WL refinement rounds")
    args = ap.parse_args()

    GA0 = g6_to_nx(args.g6A)
    GB0 = g6_to_nx(args.g6B)

    GA = line_iterate(GA0, args.iter)
    GB = line_iterate(GB0, args.iter)

    # WL-1 via your aligned WL-k implementation with k=1
    colA, colB, rounds = wlk_tuple_coloring_aligned(GA, GB, k=1, max_iter=args.max_iter)

    # Convert 1-tuple colors to vertex colors
    vcolA = {t[0]: c for t, c in colA.items()}
    vcolB = {t[0]: c for t, c in colB.items()}

    histA = Counter(vcolA.values())
    histB = Counter(vcolB.values())

    print(f"A={args.g6A}  B={args.g6B}  comparing L^{args.iter}(A) vs L^{args.iter}(B)")
    print(f"A: |V|={GA.number_of_nodes()} |E|={GA.number_of_edges()}")
    print(f"B: |V|={GB.number_of_nodes()} |E|={GB.number_of_edges()}")
    print(f"WL-1 stabilized in {rounds} rounds (aligned ids).")
    print("Hist A:", dict(sorted(histA.items())))
    print("Hist B:", dict(sorted(histB.items())))
    print("WL-1 distinguishes?", histA != histB)

    if histA != histB:
        # print a witness color where counts differ
        for c in sorted(set(histA) | set(histB)):
            if histA.get(c, 0) != histB.get(c, 0):
                print(f"Witness color {c}: A has {histA.get(c,0)}, B has {histB.get(c,0)}")
                break


if __name__ == "__main__":
    main()
