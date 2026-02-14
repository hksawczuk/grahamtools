#!/usr/bin/env python3
"""
Plot the tree of walks of a graph up to length (n-1), where n = number of vertices.

Two modes:
- Full walk tree: children append any neighbor of the endpoint.
- Universal cover model (standard): NON-BACKTRACKING walk tree:
    forbid immediate backtracking v_{k+1} = v_{k-1}.

Nodes are walk tuples (v0, v1, ..., vk).
Root is (root_vertex,).

Supports:
- graph6 input
- depth default = n-1
- label style: endpoint only (like your sketches) or full walk sequence
- symmetric "half-tree" drawing by restricting which neighbors of the root to expand
- optional non-backtracking (universal cover) behavior

Usage:
  python3 examples/walk_tree_plot.py --g6 'E`ow' --root 1
  python3 examples/walk_tree_plot.py --g6 'E`ow' --root 1 --nonbacktracking
  python3 examples/walk_tree_plot.py --g6 'E`ow' --root 1 --root-neighbors 2,6
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from grahamtools.io.graph6 import g6_to_nx


Walk = Tuple[int, ...]


@dataclass
class WalkTreeOptions:
    max_len: int                      # maximum walk length in edges
    root: int
    label: str = "endpoint"           # "endpoint" or "seq"
    root_neighbors: Optional[List[int]] = None  # restrict depth-1 expansion
    nonbacktracking: bool = False     # if True, forbid immediate backtracking


def build_walk_tree(G: nx.Graph, opts: WalkTreeOptions) -> nx.DiGraph:
    """
    Build the directed walk tree (a rooted tree as a DiGraph).

    If opts.nonbacktracking is True, children of (.., a, b) cannot append 'a'.
    """
    if opts.root not in G:
        raise ValueError(f"root={opts.root} not in graph nodes {sorted(G.nodes())}")

    T = nx.DiGraph()
    root_walk: Walk = (opts.root,)
    T.add_node(root_walk)

    frontier: List[Walk] = [root_walk]
    depth = 0

    root_nbr_set = set(opts.root_neighbors) if opts.root_neighbors is not None else None

    while frontier and depth < opts.max_len:
        new_frontier: List[Walk] = []

        for w in frontier:
            u = w[-1]
            nbrs = sorted(G.neighbors(u))

            # Optional symmetry saving: only expand selected neighbors at depth 1 from the root
            if depth == 0 and root_nbr_set is not None:
                nbrs = [v for v in nbrs if v in root_nbr_set]

            # Non-backtracking constraint: forbid v_next = w[-2]
            if opts.nonbacktracking and len(w) >= 2:
                prev = w[-2]
                nbrs = [v for v in nbrs if v != prev]

            for v in nbrs:
                w2 = w + (v,)
                T.add_node(w2)
                T.add_edge(w, w2)
                new_frontier.append(w2)

        frontier = new_frontier
        depth += 1

    return T


def hierarchy_pos_tree(
    T: nx.DiGraph,
    root: Walk,
    *,
    x_gap: float = 1.0,
    y_gap: float = 1.2,
) -> Dict[Walk, Tuple[float, float]]:
    """
    Deterministic top-down layout for a rooted tree.
    Depth controls y-coordinate; subtree leaf counts allocate horizontal space.
    """
    children = {u: list(T.successors(u)) for u in T.nodes()}
    for u in children:
        children[u].sort()

    leaf_count: Dict[Walk, int] = {}

    def count_leaves(u: Walk) -> int:
        ch = children.get(u, [])
        if not ch:
            leaf_count[u] = 1
            return 1
        s = 0
        for v in ch:
            s += count_leaves(v)
        leaf_count[u] = s
        return s

    count_leaves(root)

    pos: Dict[Walk, Tuple[float, float]] = {}

    def assign(u: Walk, x_left: float, depth: int) -> float:
        ch = children.get(u, [])
        y = -depth * y_gap
        if not ch:
            x = x_left + 0.5 * x_gap
            pos[u] = (x, y)
            return x

        x_cursor = x_left
        centers = []
        for v in ch:
            width = leaf_count[v] * x_gap
            cx = assign(v, x_cursor, depth + 1)
            centers.append(cx)
            x_cursor += width

        x = sum(centers) / len(centers)
        pos[u] = (x, y)
        return x

    assign(root, 0.0, 0)
    return pos


def node_labels(T: nx.DiGraph, label: str) -> Dict[Walk, str]:
    if label == "endpoint":
        return {w: str(w[-1]) for w in T.nodes()}
    if label == "seq":
        return {w: ",".join(map(str, w)) for w in T.nodes()}
    raise ValueError("label must be 'endpoint' or 'seq'")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g6", type=str, required=True, help="graph6 string (quote if it contains backticks)")
    ap.add_argument("--root", type=int, required=True, help="root vertex (integer node label)")
    ap.add_argument("--max-len", type=int, default=None, help="max walk length in edges; default = n-1")
    ap.add_argument("--label", choices=["endpoint", "seq"], default="endpoint")

    ap.add_argument(
        "--root-neighbors",
        type=str,
        default=None,
        help="comma-separated neighbors of root to expand at depth 1 (symmetry/half-tree). Example: 2,6",
    )

    ap.add_argument(
        "--nonbacktracking",
        action="store_true",
        help="use non-backtracking walks (universal cover model)",
    )

    ap.add_argument("--x-gap", type=float, default=1.0)
    ap.add_argument("--y-gap", type=float, default=1.2)
    ap.add_argument("--node-size", type=int, default=450)
    ap.add_argument("--font-size", type=int, default=10)
    ap.add_argument("--save", type=str, default=None, help="path to save PNG instead of showing")

    args = ap.parse_args()

    G = g6_to_nx(args.g6)
    n = G.number_of_nodes()
    max_len = args.max_len if args.max_len is not None else (n - 1)

    root_neighbors = None
    if args.root_neighbors is not None:
        root_neighbors = [int(x.strip()) for x in args.root_neighbors.split(",") if x.strip()]

    opts = WalkTreeOptions(
        max_len=max_len,
        root=args.root,
        label=args.label,
        root_neighbors=root_neighbors,
        nonbacktracking=args.nonbacktracking,
    )

    T = build_walk_tree(G, opts)
    root_walk = (opts.root,)

    pos = hierarchy_pos_tree(T, root_walk, x_gap=args.x_gap, y_gap=args.y_gap)
    labels = node_labels(T, opts.label)

    title_mode = "non-backtracking (universal cover)" if opts.nonbacktracking else "all walks"
    plt.figure(figsize=(12, 6))
    plt.title(f"Walk tree ({title_mode}), depth ≤ {opts.max_len}")

    nx.draw_networkx_edges(T, pos, arrows=False, width=1.2)
    nx.draw_networkx_nodes(T, pos, node_size=args.node_size)
    nx.draw_networkx_labels(T, pos, labels=labels, font_size=args.font_size)

    plt.axis("off")
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=250)
        plt.close()
        print(f"Saved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
