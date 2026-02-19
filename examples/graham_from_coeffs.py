#!/usr/bin/env python3
"""
Compute Graham sequence γ_k(T) for arbitrary trees using the fiber formula:

    γ_k(T) = Σ_{τ connected subtree of T} coeff_k(τ) · count(τ, T)

where coeff_k(τ) are precomputed fiber coefficients (from K5/K6/K7 data),
and count(τ, T) = number of edge-subsets of T forming a connected subgraph
isomorphic to τ.

This avoids any line graph iteration on T itself. The bottleneck is
enumerating connected subtrees of T (up to 2^|E(T)| subsets), but we
only need subtrees with ≤ max_grade edges.

Usage:
    python3 graham_from_coeffs.py [--verify]
"""

import sys
import time
from collections import defaultdict
from itertools import combinations

from grahamtools.utils.connectivity import is_connected_edges
from grahamtools.utils.linegraph_edgelist import gamma_sequence_edgelist


# ============================================================
#  Precomputed fiber coefficients from K5/K6/K7 experiments
#  coeff_k(τ) for all tree types with ≤ 6 edges
# ============================================================

# Format: COEFFS[canonical_string] = {k: coeff_k}
# Canonical strings use the AHU center-based encoding.
# We also store the edge count for convenience.

# Trees are identified by canonical form:
#   K2 = "()"                   [1 edge]
#   P3 = "(())"                 [2 edges]
#   K1,3 = "(()()()"            ... etc
# Let's use name-based keys and map to canonical forms.

# Coefficient data from our experiments:
COEFF_DATA = {
    # 1 edge: K2
    "K2": {"nedges": 1, "coeffs": {1: 1}},
    # 2 edges: P3
    "P3": {"nedges": 2, "coeffs": {2: 1}},
    # 3 edges
    "K1,3": {"nedges": 3, "coeffs": {3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3}},
    "P4":   {"nedges": 3, "coeffs": {3: 1}},
    # 4 edges
    "K1,4":      {"nedges": 4, "coeffs": {4: 24, 5: 168, 6: 1608, 7: 27528, 8: 908808}},
    "T5[32111]": {"nedges": 4, "coeffs": {4: 5, 5: 15, 6: 61, 7: 393, 8: 4549}},
    "P5":        {"nedges": 4, "coeffs": {4: 1}},
    # 5 edges
    "K1,5":           {"nedges": 5, "coeffs": {5: 480, 6: 14880}},
    "T6[421111]":     {"nedges": 5, "coeffs": {5: 75, 6: 1458}},
    "T6[331111]":     {"nedges": 5, "coeffs": {5: 48, 6: 656}},
    "T6[322111]_a":   {"nedges": 5, "coeffs": {5: 11, 6: 114}},
    "T6[322111]_b":   {"nedges": 5, "coeffs": {5: 7, 6: 39}},
    "P6":             {"nedges": 5, "coeffs": {5: 1}},
    # 6 edges
    "K1,6":              {"nedges": 6, "coeffs": {6: 23040}},
    "T7[5211111]":       {"nedges": 6, "coeffs": {6: 2928}},
    "T7[4311111]":       {"nedges": 6, "coeffs": {6: 1452}},
    "T7[4221111]_a4":    {"nedges": 6, "coeffs": {6: 342}},
    "T7[4221111]_b6":    {"nedges": 6, "coeffs": {6: 168}},
    "T7[3321111]_a2":    {"nedges": 6, "coeffs": {6: 168}},
    "T7[3321111]_b8":    {"nedges": 6, "coeffs": {6: 80}},
    "T7[3222111]_a6":    {"nedges": 6, "coeffs": {6: 33}},
    "T7[3222111]_b1":    {"nedges": 6, "coeffs": {6: 19}},
    "T7[3222111]_c2":    {"nedges": 6, "coeffs": {6: 9}},
    "P7":                {"nedges": 6, "coeffs": {6: 1}},
}


# ============================================================
#  Tree canonical forms
# ============================================================

def canonical_tree(adj):
    """Canonical string for a connected tree given as adjacency dict."""
    vertices = sorted(adj.keys())
    n = len(vertices)
    if n == 0:
        return "()"
    if n == 1:
        return "()"
    if n == 2:
        return "(())"

    deg = {v: len(adj[v]) for v in vertices}
    leaves = [v for v in vertices if deg[v] <= 1]
    removed = set()
    remaining = n

    while remaining > 2:
        new_leaves = []
        for v in leaves:
            removed.add(v)
            remaining -= 1
            for u in adj[v]:
                if u not in removed:
                    deg[u] -= 1
                    if deg[u] == 1:
                        new_leaves.append(u)
        leaves = new_leaves

    centers = [v for v in vertices if v not in removed]

    def rc(root, parent):
        ch = sorted(rc(u, root) for u in adj[root] if u != parent)
        return "(" + "".join(ch) + ")"

    if len(centers) == 1:
        return rc(centers[0], None)
    else:
        c0, c1 = centers[0], centers[1]
        ch0 = sorted(rc(u, c0) for u in adj[c0] if u != c1)
        ch1 = sorted(rc(u, c1) for u in adj[c1] if u != c0)
        opt1 = (tuple(ch0), tuple(ch1))
        opt2 = (tuple(ch1), tuple(ch0))
        return "E" + str(min(opt1, opt2))


def edges_to_adj(edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return dict(adj)


def is_connected(edges):
    """Check if edge set forms a connected graph."""
    return is_connected_edges(edges)


# ============================================================
#  Build canonical form -> name mapping from reference trees
# ============================================================

def build_reference_trees():
    """Build all reference trees and compute their canonical forms."""
    ref_trees = {
        "K2": [(0, 1)],
        "P3": [(0, 1), (1, 2)],
        "K1,3": [(0, 1), (0, 2), (0, 3)],
        "P4": [(0, 1), (1, 2), (2, 3)],
        "K1,4": [(0, 1), (0, 2), (0, 3), (0, 4)],
        "T5[32111]": [(0, 1), (0, 2), (0, 3), (1, 4)],
        "P5": [(0, 1), (1, 2), (2, 3), (3, 4)],
        "K1,5": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
        "T6[421111]": [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)],
        "T6[331111]": [(0, 1), (0, 2), (0, 3), (3, 4), (3, 5)],
        "T6[322111]_a": [(0, 1), (1, 2), (2, 3), (1, 4), (2, 5)],
        "T6[322111]_b": [(0, 1), (1, 2), (2, 3), (1, 4), (3, 5)],
        "P6": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
        "K1,6": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)],
        "T7[5211111]": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 6)],
        "T7[4311111]": [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (4, 6)],
        "T7[4221111]_a4": [(0, 1), (0, 2), (0, 3), (0, 5), (1, 4), (5, 6)],
        "T7[4221111]_b6": [(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (5, 6)],
        "T7[3321111]_a2": [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (3, 6)],
        "T7[3321111]_b8": [(0, 1), (1, 2), (1, 3), (2, 4), (4, 5), (4, 6)],
        "T7[3222111]_a6": [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (3, 6)],
        "T7[3222111]_b1": [(0, 1), (0, 2), (0, 3), (1, 4), (4, 5), (5, 6)],
        "T7[3222111]_c2": [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (3, 6)],
        "P7": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
    }

    canon_to_name = {}
    for name, edges in ref_trees.items():
        adj = edges_to_adj(edges)
        canon = canonical_tree(adj)
        canon_to_name[canon] = name

    return ref_trees, canon_to_name


# ============================================================
#  Count connected subtrees of T by type
# ============================================================

def count_subtree_types(tree_edges, max_edges=6):
    """Count occurrences of each connected subtree type in T,
    up to max_edges edges.

    Returns dict: canonical_string -> count
    """
    n_edges = len(tree_edges)
    counts = defaultdict(int)

    for size in range(1, min(max_edges, n_edges) + 1):
        for subset in combinations(range(n_edges), size):
            edges = [tree_edges[i] for i in subset]
            if not is_connected(edges):
                continue
            adj = edges_to_adj(edges)
            canon = canonical_tree(adj)
            counts[canon] += 1

    return dict(counts)


# ============================================================
#  Compute γ_k(T) from coefficients and subtree counts
# ============================================================

def gamma_from_coeffs(subtree_counts, canon_to_name, max_k):
    """Compute γ_k(T) = Σ_τ coeff_k(τ) · count(τ, T) for k = 1..max_k.

    Returns dict: k -> γ_k(T)
    Also returns breakdown by subtree type for debugging.
    """
    gamma = defaultdict(int)
    breakdown = defaultdict(dict)

    for canon, count in subtree_counts.items():
        name = canon_to_name.get(canon)
        if name is None:
            # Unknown subtree type — not in our coefficient table
            continue

        coeffs = COEFF_DATA.get(name, {}).get("coeffs", {})
        for k, c in coeffs.items():
            if k > max_k:
                continue
            contrib = c * count
            gamma[k] += contrib
            breakdown[k][name] = contrib

    return dict(gamma), dict(breakdown)


# ============================================================
#  Brute-force line graph iteration (for verification)
# ============================================================

def gamma_bruteforce(edges, max_k, max_edge_limit=5_000_000):
    seq_list = gamma_sequence_edgelist(edges, max_k, max_edges=max_edge_limit)
    seq = {}
    for k, val in enumerate(seq_list):
        if val is not None:
            seq[k] = val
    return seq


# ============================================================
#  Main
# ============================================================

def main():
    do_verify = "--verify" in sys.argv
    max_k = 8

    for i, arg in enumerate(sys.argv):
        if arg == "--max-k" and i + 1 < len(sys.argv):
            max_k = int(sys.argv[i + 1])

    ref_trees, canon_to_name = build_reference_trees()

    # Max grade we have coefficients for
    max_coeff_k = 6  # most types only have data through k=6
    # K1,3 goes to 8, K1,4 and T5[32111] go to 8, but 5-edge types only to 6
    # We'll compute what we can

    # ---- Verification on small trees ----
    if do_verify:
        print("=" * 60)
        print("  Verification: γ_k from coefficients vs brute-force")
        print("=" * 60)

        test_trees = {
            "K1,3": [(0, 1), (0, 2), (0, 3)],
            "P4": [(0, 1), (1, 2), (2, 3)],
            "K1,4": [(0, 1), (0, 2), (0, 3), (0, 4)],
            "T5[32111]": [(0, 1), (0, 2), (0, 3), (1, 4)],
            "P5": [(0, 1), (1, 2), (2, 3), (3, 4)],
            "K1,5": [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)],
        }

        for name, edges in test_trees.items():
            print(f"\n  {name} (edges: {edges})")

            # Count subtree types
            t0 = time.time()
            counts = count_subtree_types(edges, max_edges=len(edges))
            t_count = time.time() - t0

            print(f"    Subtree types found: {len(counts)}  ({t_count:.3f}s)")
            for canon, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                cname = canon_to_name.get(canon, f"??? ({canon})")
                print(f"      {cname}: {cnt}")

            # Compute γ_k from formula
            gamma_formula, breakdown = gamma_from_coeffs(counts, canon_to_name, max_k)

            # Compute γ_k brute-force
            gamma_bf = gamma_bruteforce(edges, max_k)

            print(f"    {'k':>4s} {'formula':>10s} {'brute':>10s} {'match':>6s}")
            print(f"    {'-' * 34}")
            for k in range(1, max_k + 1):
                gf = gamma_formula.get(k)
                gb = gamma_bf.get(k)
                if gf is not None and gb is not None:
                    eq = "✓" if gf == gb else "✗"
                    print(f"    {k:>4d} {gf:>10d} {gb:>10d} {eq:>6s}")
                elif gb is not None:
                    print(f"    {k:>4d} {'—':>10s} {gb:>10d}")
                elif gf is not None:
                    print(f"    {k:>4d} {gf:>10d} {'—':>10s}")

        print()

    # ---- DDS-equivalent pair ----
    print("=" * 60)
    print("  Graham sequences for DDS-equivalent pair (n=18)")
    print("=" * 60)

    dds_trees = {
        "Tree A (clustered)": [
            (0,1),(0,2),(0,3),(0,4),(0,5),
            (1,6),(1,7),(1,8),(1,9),
            (2,10),(2,11),(3,12),(3,13),
            (6,14),(7,15),(8,16),(9,17),
        ],
        "Tree B (balanced)": [
            (0,1),(0,2),(0,3),(0,4),(0,5),
            (1,6),(1,7),(1,8),(1,9),
            (2,10),(2,11),(3,12),(4,13),
            (6,14),(6,15),(7,16),(8,17),
        ],
    }

    results = {}
    all_breakdowns = {}

    for name, edges in dds_trees.items():
        print(f"\n  {name}:")
        t0 = time.time()
        counts = count_subtree_types(edges, max_edges=6)
        t_count = time.time() - t0
        print(f"    Subtree counting: {len(counts)} types ({t_count:.2f}s)")

        # Show subtree counts
        for canon, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            cname = canon_to_name.get(canon, f"unknown ({canon[:30]}...)")
            print(f"      {cname:>22s}: {cnt}")

        gamma, breakdown = gamma_from_coeffs(counts, canon_to_name, max_k)
        results[name] = gamma
        all_breakdowns[name] = breakdown

    # Compare
    names = list(dds_trees.keys())
    ga = results[names[0]]
    gb = results[names[1]]

    print(f"\n{'=' * 60}")
    print(f"  Comparison")
    print(f"{'=' * 60}")
    print(f"  {'k':>3s} {'Tree A':>14s} {'Tree B':>14s} {'diff':>14s} {'match':>7s}")
    print(f"  {'-' * 55}")

    for k in range(1, max_k + 1):
        va = ga.get(k, None)
        vb = gb.get(k, None)
        if va is not None and vb is not None:
            diff = va - vb
            eq = "✓" if va == vb else "✗ ←←←"
            print(f"  {k:>3d} {va:>14,} {vb:>14,} {diff:>14,} {eq:>7s}")
        else:
            sa = f"{va:,}" if va is not None else "—"
            sb = f"{vb:,}" if vb is not None else "—"
            print(f"  {k:>3d} {sa:>14s} {sb:>14s} {'':>14s}")

    # Show breakdown for first differing k
    for k in range(1, max_k + 1):
        va = ga.get(k)
        vb = gb.get(k)
        if va is not None and vb is not None and va != vb:
            print(f"\n  Breakdown at k={k}:")
            ba = all_breakdowns[names[0]].get(k, {})
            bb = all_breakdowns[names[1]].get(k, {})
            all_types = sorted(set(list(ba.keys()) + list(bb.keys())))
            print(f"    {'type':>22s} {'Tree A':>10s} {'Tree B':>10s} {'diff':>10s}")
            print(f"    {'-' * 56}")
            for t in all_types:
                a = ba.get(t, 0)
                b = bb.get(t, 0)
                d = a - b
                mark = " ←" if d != 0 else ""
                print(f"    {t:>22s} {a:>10,} {b:>10,} {d:>10,}{mark}")
            break


if __name__ == "__main__":
    main()