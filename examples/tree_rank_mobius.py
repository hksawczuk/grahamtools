#!/usr/bin/env python3
"""
Tree coefficient matrix via Möbius inversion.

Convention (matching wl1_optimized.py):
  coeff_k(τ) = γ_k(τ) - Σ_{σ ⊊ τ} count(σ,τ) · coeff_k(σ)

where count(σ,τ) = number of edge-subsets of τ isomorphic to σ,
and γ_k(τ) = |V(L^k(τ))|.

Computes γ_k by iterating line graphs on each small tree directly.
Much faster than building L^k(K_n).

Usage: python3 tree_rank_mobius.py [--max-k 12] [--max-tree-edges 6]
"""

from __future__ import annotations
import sys
import time
import math
import argparse
from collections import defaultdict
from fractions import Fraction
from itertools import combinations, permutations
from typing import Dict, List, Tuple, Optional


# ============================================================
#  Line graph
# ============================================================

def line_graph(edges):
    """Compute line graph. Returns (new_edges, n_vertices=len(edges))."""
    m = len(edges)
    if m == 0:
        return [], 0

    incident = defaultdict(list)
    for idx, (u, v) in enumerate(edges):
        incident[u].append(idx)
        incident[v].append(idx)

    new_edges = set()
    for inc in incident.values():
        for i in range(len(inc)):
            for j in range(i + 1, len(inc)):
                a, b = inc[i], inc[j]
                new_edges.add((a, b) if a < b else (b, a))

    return sorted(new_edges), m


def gamma_sequence_star(r, max_k):
    """Analytical γ sequence for K_{1,r}. L(K_{1,r}) = K_r, then iterate.
    
    K_r is regular: v_1 = r, d_1 = r-1 (since K_r is (r-1)-regular).
    Recurrence: v_{j+1} = v_j * d_j / 2, d_{j+1} = 2*d_j - 2.
    γ_0 = r+1, γ_k = v_k for k >= 1.
    """
    seq = [r + 1]  # γ_0 = |V(K_{1,r})| = r+1
    if max_k < 1:
        return seq
    
    v = r        # |V(L(K_{1,r}))| = |E(K_{1,r})| = r = |V(K_r)|
    d = r - 1    # K_r is (r-1)-regular
    seq.append(v)  # γ_1 = r
    
    for k in range(2, max_k + 1):
        e = v * d // 2  # |E| of current graph
        v = e            # |V| of next line graph
        d = 2 * d - 2   # degree of next line graph (regular graph stays regular)
        seq.append(v)
    
    return seq


def is_star(edges):
    """Check if edge list is a star K_{1,r}. Returns r or None."""
    if not edges:
        return None
    adj = defaultdict(int)
    for u, v in edges:
        adj[u] += 1
        adj[v] += 1
    degs = sorted(adj.values(), reverse=True)
    m = len(edges)
    if degs[0] == m and all(d == 1 for d in degs[1:]):
        return m
    return None


def gamma_sequence(edges, max_k, max_lg_edges=5_000_000):
    """Compute γ_0, ..., γ_{max_k} where γ_k = |V(L^k(G))|.
    
    Uses analytical formula for stars, iterative line graph otherwise.
    """
    # Check for star
    r = is_star(edges)
    if r is not None:
        return gamma_sequence_star(r, max_k)
    
    verts = set()
    for u, v in edges:
        verts.add(u)
        verts.add(v)

    v_map = {v: i for i, v in enumerate(sorted(verts))}
    current_edges = sorted((min(v_map[u], v_map[v]), max(v_map[u], v_map[v]))
                           for u, v in edges)

    seq = [len(verts)]  # γ_0

    for k in range(1, max_k + 1):
        gamma_k = len(current_edges)  # |V(L^k)| = |E(L^{k-1})|
        seq.append(gamma_k)

        if gamma_k == 0:
            while len(seq) <= max_k:
                seq.append(0)
            break

        # Estimate next line graph size before computing
        incident = defaultdict(int)
        for u, v in current_edges:
            incident[u] += 1
            incident[v] += 1
        est_next = sum(d * (d - 1) // 2 for d in incident.values())

        if est_next > max_lg_edges:
            while len(seq) <= max_k:
                seq.append(None)
            break

        new_edges, _ = line_graph(current_edges)
        current_edges = new_edges

    return seq


# ============================================================
#  Canonical forms
# ============================================================

def canonical_graph(edges):
    """Canonical form of a graph given by edge list."""
    if not edges:
        return ()

    verts = set()
    for u, v in edges:
        verts.add(u)
        verts.add(v)
    vlist = sorted(verts)
    n = len(vlist)

    best = None
    for perm in permutations(range(n)):
        v_map = {vlist[i]: perm[i] for i in range(n)}
        relabeled = tuple(sorted(
            (min(v_map[u], v_map[v]), max(v_map[u], v_map[v]))
            for u, v in edges
        ))
        if best is None or relabeled < best:
            best = relabeled
    return best


# ============================================================
#  Connectivity and subgraph enumeration
# ============================================================

def is_connected(edges):
    """Check if edge set forms a connected graph."""
    if not edges:
        return False
    adj = defaultdict(set)
    verts = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        verts.add(u)
        verts.add(v)
    start = next(iter(verts))
    visited = {start}
    stack = [start]
    while stack:
        v = stack.pop()
        for u in adj[v]:
            if u not in visited:
                visited.add(u)
                stack.append(u)
    return len(visited) == len(verts)


def enumerate_connected_subgraphs(edges, max_size=None):
    """Enumerate connected subgraphs by edge subsets.
    Returns dict: canon_form -> (count, example_edges, n_verts, n_edges)."""
    n_edges = len(edges)
    if max_size is None:
        max_size = n_edges

    counts = {}
    for size in range(1, min(max_size, n_edges) + 1):
        for subset in combinations(range(n_edges), size):
            sub_edges = [edges[i] for i in subset]
            if not is_connected(sub_edges):
                continue
            canon = canonical_graph(sub_edges)
            if canon not in counts:
                verts = set()
                for u, v in sub_edges:
                    verts.add(u)
                    verts.add(v)
                counts[canon] = [0, sub_edges, len(verts), size]
            counts[canon][0] += 1

    return {k: tuple(v) for k, v in counts.items()}


# ============================================================
#  Generate all trees up to given edge count
# ============================================================

def prufer_to_edges(seq):
    """Convert Prüfer sequence to edge list."""
    n = len(seq) + 2
    degree = [1] * n
    for v in seq:
        degree[v] += 1
    edges = []
    for v in seq:
        for u in range(n):
            if degree[u] == 1:
                edges.append((u, v))
                degree[u] -= 1
                degree[v] -= 1
                break
    # Last edge: two remaining degree-1 vertices
    last = [u for u in range(n) if degree[u] == 1]
    edges.append((last[0], last[1]))
    return edges


def generate_trees(max_edges):
    """Generate all non-isomorphic trees with 1..max_edges edges.
    Uses Prüfer sequences for exhaustive generation + canonical dedup.
    Returns list of (canon, edges, name, n_verts, n_edges)."""

    all_trees = []
    seen_canons = set()

    # Known names for small trees
    known_names = {
        1: {0: "K2"},
        2: {0: "P3"},
        3: {},  # filled below
        4: {},
        5: {},
        6: {},
    }

    for ne in range(1, max_edges + 1):
        nv = ne + 1
        count = 0

        if nv == 2:
            # Only one tree: K2
            edges = [(0, 1)]
            canon = canonical_graph(edges)
            if canon not in seen_canons:
                seen_canons.add(canon)
                all_trees.append((canon, edges, "K2", nv, ne))
                count += 1
        elif nv == 3:
            edges = [(0, 1), (1, 2)]
            canon = canonical_graph(edges)
            if canon not in seen_canons:
                seen_canons.add(canon)
                all_trees.append((canon, edges, "P3", nv, ne))
                count += 1
        else:
            # Enumerate all Prüfer sequences of length nv-2
            from itertools import product
            new_this_size = []
            for seq in product(range(nv), repeat=nv - 2):
                edges = prufer_to_edges(list(seq))
                canon = canonical_graph(edges)
                if canon not in seen_canons:
                    seen_canons.add(canon)
                    new_this_size.append((canon, edges, nv, ne))
                    count += 1

            # Sort by degree sequence for consistent naming
            def deg_seq(edges):
                d = defaultdict(int)
                for u, v in edges:
                    d[u] += 1
                    d[v] += 1
                return tuple(sorted(d.values(), reverse=True))

            new_this_size.sort(key=lambda t: deg_seq(t[1]))

            # Auto-name
            for canon, edges, nv2, ne2 in new_this_size:
                ds = deg_seq(edges)
                if ds[-1] == 1 and ds[0] == ne:
                    name = f"K1_{ne}"
                elif all(d <= 2 for d in ds):
                    name = f"P{nv}"
                elif ne == 3 and ds == (2, 2, 1, 1):
                    name = "P4"
                elif ne == 4 and ds == (3, 2, 1, 1, 1):
                    name = "fork"
                elif ne == 5 and ds == (3, 3, 1, 1, 1, 1):
                    name = "dblstar"
                elif ne == 5 and ds == (4, 2, 1, 1, 1, 1):
                    name = "spider"
                else:
                    ds_str = "".join(str(d) for d in ds)
                    name = f"T{nv}_{ds_str}"
                # Deduplicate names
                existing = [n for _, _, n, _, _ in all_trees if n == name or n.startswith(name + "_")]
                if existing:
                    name = f"{name}_{chr(ord('a') + len(existing))}"
                all_trees.append((canon, edges, name, nv2, ne2))

        print(f"  {ne}-edge trees: {count} types")

    return all_trees


# ============================================================
#  Describe graph
# ============================================================

def describe_graph(edges):
    if not edges:
        return "empty"
    adj = defaultdict(set)
    verts = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        verts.add(u)
        verts.add(v)
    n = len(verts)
    m = len(edges)
    deg_seq = sorted([len(adj[v]) for v in verts], reverse=True)

    if m == 1:
        return "K2"
    if m == n - 1:
        if all(d <= 2 for d in deg_seq):
            return f"P{n}"
        if deg_seq.count(1) == n - 1:
            return f"K1,{n-1}"
        ds = "".join(str(d) for d in deg_seq)
        return f"Tree({n}v,{ds})"
    if all(d == 2 for d in deg_seq) and m == n:
        return f"C{n}"
    return f"Graph({n}v,{m}e)"


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--max-tree-edges", type=int, default=6)
    parser.add_argument("--max-lg-edges", type=int, default=5_000_000)
    args = parser.parse_args()

    max_k = args.max_k
    max_tree_edges = args.max_tree_edges
    max_lg_edges = args.max_lg_edges

    print(f"Generating trees with up to {max_tree_edges} edges...")
    trees = generate_trees(max_tree_edges)
    print(f"  {len(trees)} trees total\n")

    # Sort by edge count then name
    trees.sort(key=lambda t: (t[4], t[2]))

    # Enumerate all connected subgraph types that appear in any tree
    # Process in order of increasing edge count for Möbius inversion
    # We need coefficients for ALL connected subgraph types, not just trees

    print(f"Enumerating connected subgraphs of all trees...")
    t0 = time.time()

    # Collect all connected subgraph types across all trees
    all_types = {}     # canon -> (example_edges, n_verts, n_edges)
    tree_subtypes = {} # tree_canon -> {sub_canon: count}

    for canon, edges, name, nv, ne in trees:
        subs = enumerate_connected_subgraphs(edges, max_size=ne)  # include self
        tree_subtypes[canon] = {}
        for sub_canon, (count, sub_edges, sub_nv, sub_ne) in subs.items():
            tree_subtypes[canon][sub_canon] = count
            if sub_canon not in all_types:
                all_types[sub_canon] = (sub_edges, sub_nv, sub_ne)

    print(f"  {len(all_types)} connected subgraph types ({time.time()-t0:.1f}s)\n")

    # Sort all types by edge count
    sorted_types = sorted(all_types.keys(), key=lambda c: (all_types[c][2], c))

    # Compute γ sequences for all types
    print(f"Computing γ sequences (max_k={max_k})...")
    gammas = {}  # canon -> [γ_0, γ_1, ..., γ_{max_k}]
    for canon in sorted_types:
        sub_edges, sub_nv, sub_ne = all_types[canon]
        desc = describe_graph(sub_edges)
        t0 = time.time()
        gamma = gamma_sequence(sub_edges, max_k, max_lg_edges)
        dt = time.time() - t0
        gammas[canon] = gamma
        max_valid = max(k for k in range(len(gamma)) if gamma[k] is not None) if gamma else 0
        if dt > 0.1:
            print(f"  {desc} ({sub_ne}e): γ computed through k={max_valid} ({dt:.1f}s)")

    print()

    # Möbius inversion: process in order of increasing edge count
    print(f"Computing coefficients via Möbius inversion...")
    coeffs = {}  # canon -> [coeff_0, coeff_1, ..., coeff_{max_k}]

    for canon in sorted_types:
        sub_edges, sub_nv, sub_ne = all_types[canon]
        gamma = gammas[canon]

        coeff = list(gamma)

        # Find this type's subtypes (proper subgraphs)
        # We need the subtypes of this specific graph
        proper_subs = enumerate_connected_subgraphs(sub_edges, max_size=sub_ne - 1) if sub_ne > 1 else {}

        for sub_canon, (count, _, _, _) in proper_subs.items():
            if sub_canon in coeffs:
                for k in range(min(len(coeff), len(coeffs[sub_canon]))):
                    if coeff[k] is not None and coeffs[sub_canon][k] is not None:
                        coeff[k] -= coeffs[sub_canon][k] * count

        coeffs[canon] = coeff

    print()

    # Extract tree coefficients and build matrix
    print(f"{'#'*70}")
    print(f"# TREE COEFFICIENT MATRIX")
    print(f"{'#'*70}")

    tree_canons = [canon for canon, edges, name, nv, ne in trees]
    tree_names = {canon: name for canon, edges, name, nv, ne in trees}
    tree_nedges = {canon: ne for canon, edges, name, nv, ne in trees}

    # Print coefficient sequences
    print(f"\nCoefficient sequences:")
    for canon, edges, name, nv, ne in trees:
        coeff = coeffs.get(canon, [])
        vals = []
        for k in range(max_k + 1):
            if k < len(coeff) and coeff[k] is not None:
                vals.append(str(coeff[k]))
            else:
                vals.append("–")
        print(f"  {name:>10s} ({ne}e): {', '.join(vals[1:])}")  # skip k=0

    # Build and display matrix (grades as rows, tree types as columns)
    # Only include grades where at least one tree has data
    valid_grades = []
    for k in range(1, max_k + 1):
        has_data = False
        for canon in tree_canons:
            c = coeffs.get(canon, [])
            if k < len(c) and c[k] is not None:
                has_data = True
                break
        if has_data:
            valid_grades.append(k)

    names = [tree_names[c] for c in tree_canons]
    col_w = max(max(len(n) for n in names), 10) + 2

    print(f"\nFull matrix:")
    header = "  grade" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)

    for k in valid_grades:
        vals = []
        for canon in tree_canons:
            c = coeffs.get(canon, [])
            if k < len(c) and c[k] is not None:
                vals.append(f"{c[k]:>{col_w}}")
            else:
                vals.append(f"{'–':>{col_w}}")
        print(f"  k={k:<3d}" + "".join(vals))

    # Rank analysis by edge count
    print(f"\n{'#'*70}")
    print(f"# RANK ANALYSIS")
    print(f"{'#'*70}")

    trees_by_edges = defaultdict(list)
    for canon, edges, name, nv, ne in trees:
        trees_by_edges[ne].append(canon)

    for ne in sorted(trees_by_edges.keys()):
        tkeys = trees_by_edges[ne]
        n_trees = len(tkeys)
        tnames = [tree_names[c] for c in tkeys]

        if n_trees <= 1:
            print(f"\n  {ne}-edge trees: 1 type ({tnames[0]}) — trivially independent")
            continue

        # Build matrix including all grades from ne onward
        grades = [k for k in valid_grades if k >= ne]
        M = []
        for k in grades:
            row = []
            for canon in tkeys:
                c = coeffs.get(canon, [])
                if k < len(c) and c[k] is not None:
                    row.append(c[k])
                else:
                    row.append(0)  # treat missing as 0 (conservative)
            M.append(row)

        n_rows = len(M)

        print(f"\n  {'='*60}")
        print(f"  {ne}-edge trees ({n_trees} types): {tnames}")
        print(f"  {'='*60}")

        header = "  grade" + "".join(f"{n:>{col_w}}" for n in tnames)
        print(header)
        for ki, k in enumerate(grades):
            row_str = f"  k={k:<3d}"
            for ti in range(n_trees):
                c = coeffs.get(tkeys[ti], [])
                if k < len(c) and c[k] is not None:
                    row_str += f"{M[ki][ti]:>{col_w}}"
                else:
                    row_str += f"{'–':>{col_w}}"
            print(row_str)

        rank = exact_rank(M, n_rows, n_trees)
        print(f"\n  Rank: {rank} / {n_trees}", end="")
        if rank == n_trees:
            print(" — FULL RANK ✓")
        else:
            print(f" — need {n_trees - rank} more")

        # Cumulative
        print("  Cumulative rank:")
        for nk in range(1, min(n_rows + 1, n_trees + 3)):
            r = exact_rank(M[:nk], nk, n_trees)
            print(f"    Through grade {grades[nk-1]}: rank {r}")
            if r == n_trees:
                print(f"    → Full rank at grade {grades[nk-1]}")
                break

    # All trees combined
    all_tkeys = tree_canons
    n_all = len(all_tkeys)
    all_names = [tree_names[c] for c in all_tkeys]
    min_e = min(tree_nedges[c] for c in all_tkeys)
    grades = [k for k in valid_grades if k >= min_e]

    M = []
    for k in grades:
        row = []
        for canon in all_tkeys:
            c = coeffs.get(canon, [])
            if k < len(c) and c[k] is not None:
                row.append(c[k])
            else:
                row.append(0)
        M.append(row)

    n_rows = len(M)

    print(f"\n  {'='*60}")
    print(f"  ALL TREES ({n_all} types)")
    print(f"  {'='*60}")
    rank = exact_rank(M, n_rows, n_all)
    print(f"  Rank: {rank} / {n_all}", end="")
    if rank == n_all:
        print(" — FULL RANK ✓")
    else:
        print(f" — need {n_all - rank} more")

    print("  Cumulative rank:")
    for nk in range(1, min(n_rows + 1, n_all + 3)):
        r = exact_rank(M[:nk], nk, n_all)
        print(f"    Through grade {grades[nk-1]}: rank {r}")
        if r == n_all:
            print(f"    → Full rank at grade {grades[nk-1]}")
            break


def exact_rank(M, n_rows, n_cols):
    Mf = [[Fraction(M[i][j]) for j in range(n_cols)] for i in range(n_rows)]
    rank = 0
    rp = 0
    for col in range(n_cols):
        piv = None
        for r in range(rp, n_rows):
            if Mf[r][col] != 0:
                piv = r
                break
        if piv is None:
            continue
        Mf[rp], Mf[piv] = Mf[piv], Mf[rp]
        for r in range(n_rows):
            if r != rp and Mf[r][col] != 0:
                f = Mf[r][col] / Mf[rp][col]
                for c in range(n_cols):
                    Mf[r][c] -= f * Mf[rp][c]
        rp += 1
        rank += 1
    return rank


if __name__ == "__main__":
    main()