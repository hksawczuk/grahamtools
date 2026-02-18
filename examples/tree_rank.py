#!/usr/bin/env python3
"""
Extract tree coefficient vectors from L^k(K_n) and analyze linear independence.

Uses prune_cycles=True to avoid building non-forest vertices,
giving a massive speedup at higher grades.

Usage:
    python tree_rank.py [--max-k MAX_K]

Place alongside labels_kn.py in the grahamtools/examples directory.
"""

from __future__ import annotations
import sys
import time
import math
import argparse
from collections import defaultdict
from fractions import Fraction
from typing import Dict, List, Tuple

from labels_kn import (
    generate_levels_Kn_ids,
    expand_to_simple_base_edges_id,
    canon_key_bruteforce_bitset,
    aut_size_via_color_classes,
)


# ─────────────────────────────────────────────
# Tree classification
# ─────────────────────────────────────────────

def classify_tree(edges_0based, n):
    """Classify a tree by isomorphism type. Returns key or None."""
    m = len(edges_0based)
    if m == 0:
        return None
    
    verts = set()
    for u, v in edges_0based:
        verts.add(u)
        verts.add(v)
    nv = len(verts)
    
    if m != nv - 1:
        return None  # not a tree (forest with multiple components)
    
    # Check connectivity
    adj = defaultdict(set)
    for u, v in edges_0based:
        adj[u].add(v)
        adj[v].add(u)
    
    start = next(iter(verts))
    visited = {start}
    stack = [start]
    while stack:
        v = stack.pop()
        for u in adj[v]:
            if u not in visited:
                visited.add(u)
                stack.append(u)
    if len(visited) != nv:
        return None
    
    deg_seq = tuple(sorted((len(adj[v]) for v in verts), reverse=True))
    
    # Canonical signature: sorted (degree, sorted neighbor degrees) per vertex
    vertex_sigs = []
    for v in sorted(verts):
        nbr_degs = tuple(sorted((len(adj[u]) for u in adj[v]), reverse=True))
        vertex_sigs.append((len(adj[v]), nbr_degs))
    vertex_sig = tuple(sorted(vertex_sigs, reverse=True))
    
    return (deg_seq, vertex_sig)


def tree_name(deg_seq, nv):
    """Human-readable name from degree sequence."""
    if nv == 2: return "K2"
    if nv == 3: return "P3"
    if nv == 4:
        return "K1_3" if deg_seq[0] == 3 else "P4"
    if nv == 5:
        if deg_seq[0] == 4: return "K1_4"
        if deg_seq[0] == 3: return "fork"
        return "P5"
    if nv == 6:
        if deg_seq[0] == 5: return "K1_5"
        if deg_seq == (4, 2, 1, 1, 1, 1): return "spider"
        if deg_seq == (3, 3, 1, 1, 1, 1): return "dblstar"
        if deg_seq == (3, 2, 2, 1, 1, 1): return "cat322"
        if deg_seq == (2, 2, 2, 2, 1, 1): return "P6"
    if nv == 7:
        if deg_seq[0] == 6: return "K1_6"
    ds_str = "".join(str(d) for d in deg_seq)
    return f"T{nv}_{ds_str}"


# ─────────────────────────────────────────────
# Coefficient extraction
# ─────────────────────────────────────────────

def extract_tree_coefficients(n, max_k, verbose=True):
    """Build L^k(K_n) with forest pruning, extract tree coefficients."""
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Computing tree coefficients in K_{n}, grades 1..{max_k}")
        print(f"{'='*60}")
    
    t_total = time.time()
    
    if verbose:
        print(f"  Building L^k(K_{n}) with prune_cycles=True...", flush=True)
    
    t0 = time.time()
    V_by_level, endpoints_by_level = generate_levels_Kn_ids(n, max_k, prune_cycles=True)
    if verbose:
        print(f"  Done ({time.time()-t0:.1f}s)")
        for k in sorted(V_by_level.keys()):
            print(f"    L^{k}(K_{n}) [forests only]: {len(V_by_level[k])} vertices")
    
    coeffs = {}
    tree_info = {}
    
    for k in range(1, max_k + 1):
        if k not in endpoints_by_level:
            break
        
        vk_size = len(endpoints_by_level[k])
        if vk_size == 0:
            break
        
        if verbose:
            print(f"\n  Grade {k}: {vk_size} forest vertices", flush=True)
        
        if vk_size > 50_000_000:
            if verbose:
                print(f"    Too large, stopping.")
            break
        
        t0 = time.time()
        
        # Classify each forest vertex — only canonicalize trees
        buckets = {}
        n_trees_total = 0
        n_forests_skipped = 0
        
        for v in range(vk_size):
            edges = expand_to_simple_base_edges_id(v, k, endpoints_by_level)
            
            # Quick check: is this a tree (connected forest)?
            ne = len(edges)
            verts = set()
            for u, w in edges:
                verts.add(u)
                verts.add(w)
            nv = len(verts)
            if ne != nv - 1:
                n_forests_skipped += 1
                continue
            
            n_trees_total += 1
            key = canon_key_bruteforce_bitset(edges, n)
            if key not in buckets:
                buckets[key] = {"edges": edges, "freq": 0}
            buckets[key]["freq"] += 1
        
        # Extract tree coefficients from buckets
        n_tree_classes = 0
        for key, b in buckets.items():
            edges = b["edges"]
            tree_key = classify_tree(edges, n)
            
            if tree_key is None:
                continue
            
            n_tree_classes += 1
            ne = len(edges)
            nv = len(set(u for e in edges for u in e))
            
            aut = aut_size_via_color_classes(edges, n)
            orbit = math.factorial(n) // aut
            freq = b["freq"]
            coeff = freq // orbit
            
            assert freq % orbit == 0, f"Non-integer coeff: {freq}/{orbit}"
            
            coeffs[(k, tree_key)] = coeff
            
            if tree_key not in tree_info:
                name = tree_name(tree_key[0], nv)
                existing = [tk for tk in tree_info if tree_info[tk][0] == name]
                if existing:
                    name = f"{name}_{chr(ord('a') + len(existing))}"
                tree_info[tree_key] = (name, ne, nv)
        
        dt = time.time() - t0
        if verbose:
            print(f"    {n_trees_total} trees, {n_forests_skipped} non-tree forests skipped, "
                  f"{n_tree_classes} iso classes ({dt:.1f}s)")
            tree_coeffs = [(tree_info[tk][0], coeffs[(k, tk)]) 
                           for tk in sorted(tree_info.keys(), key=lambda x: tree_info[x][0])
                           if (k, tk) in coeffs and coeffs[(k, tk)] > 0]
            for name, val in tree_coeffs:
                print(f"      {name}: {val}")
    
    if verbose:
        print(f"\n  Total time: {time.time()-t_total:.1f}s")
    
    return coeffs, tree_info


# ─────────────────────────────────────────────
# Rank analysis
# ─────────────────────────────────────────────

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


def analyze_independence(coeffs, tree_info):
    if not tree_info:
        return
    
    trees_by_edges = defaultdict(list)
    for tk, (name, ne, nv) in tree_info.items():
        trees_by_edges[ne].append(tk)
    for ne in trees_by_edges:
        trees_by_edges[ne].sort(key=lambda tk: tree_info[tk][0])
    
    all_grades = sorted(set(k for (k, tk) in coeffs.keys()))
    
    # Per edge-count analysis
    for ne in sorted(trees_by_edges.keys()):
        tkeys = trees_by_edges[ne]
        n_trees = len(tkeys)
        if n_trees <= 1:
            print(f"\n  Trees with {ne} edges: 1 type (trivially independent)")
            continue
        
        names = [tree_info[tk][0] for tk in tkeys]
        grades = [k for k in all_grades if k >= ne]
        
        M = []
        for k in grades:
            row = [coeffs.get((k, tk), 0) for tk in tkeys]
            M.append(row)
        n_rows = len(M)
        
        print(f"\n  {'='*60}")
        print(f"  Trees with {ne} edges ({n_trees} types): {names}")
        print(f"  {'='*60}")
        
        header = "  grade " + "".join(f"{name:>14}" for name in names)
        print(header)
        for ki, k in enumerate(grades):
            if any(M[ki][ti] != 0 for ti in range(n_trees)):
                row_str = f"  k={k:2d}  " + "".join(f"{M[ki][ti]:>14}" for ti in range(n_trees))
                print(row_str)
        
        rank = exact_rank(M, n_rows, n_trees)
        print(f"\n  Rank: {rank} / {n_trees}", end="")
        if rank == n_trees:
            print(" — FULL RANK ✓")
        else:
            print(f" — need {n_trees - rank} more rows")
        
        if n_trees > 1:
            print("  Cumulative rank:")
            for nk in range(1, min(n_rows + 1, n_trees + 3)):
                r = exact_rank(M[:nk], nk, n_trees)
                k_end = grades[nk - 1]
                print(f"    Through grade {k_end}: rank {r}")
                if r == n_trees:
                    print(f"    → Full rank achieved at grade {k_end}")
                    break
    
    # All trees together
    all_tkeys = sorted(tree_info.keys(), key=lambda tk: (tree_info[tk][1], tree_info[tk][0]))
    n_all = len(all_tkeys)
    all_names = [tree_info[tk][0] for tk in all_tkeys]
    min_e = min(tree_info[tk][1] for tk in all_tkeys)
    grades = [k for k in all_grades if k >= min_e]
    
    M = []
    for k in grades:
        row = [coeffs.get((k, tk), 0) for tk in all_tkeys]
        M.append(row)
    n_rows = len(M)
    
    print(f"\n  {'='*60}")
    print(f"  ALL TREES ({n_all} types)")
    print(f"  {'='*60}")
    
    header = "  grade " + "".join(f"{name:>12}" for name in all_names)
    print(header)
    for ki, k in enumerate(grades):
        if any(M[ki][ti] != 0 for ti in range(n_all)):
            row_str = f"  k={k:2d}  " + "".join(f"{M[ki][ti]:>12}" for ti in range(n_all))
            print(row_str)
    
    rank = exact_rank(M, n_rows, n_all)
    print(f"\n  Rank: {rank} / {n_all}", end="")
    if rank == n_all:
        print(" — FULL RANK ✓")
    else:
        print(f" — need {n_all - rank} more rows")
    
    print("  Cumulative rank:")
    for nk in range(1, min(n_rows + 1, n_all + 3)):
        r = exact_rank(M[:nk], nk, n_all)
        k_end = grades[nk - 1]
        print(f"    Through grade {k_end}: rank {r}")
        if r == n_all:
            print(f"    → Full rank achieved at grade {k_end}")
            break


def main():
    parser = argparse.ArgumentParser(description="Tree coefficient rank analysis")
    parser.add_argument("--max-k", type=int, default=10,
                        help="Maximum grade to compute")
    args = parser.parse_args()
    
    max_k = args.max_k
    all_coeffs = {}
    all_tree_info = {}
    
    configs = [
        (5, max_k),               # 3,4-edge trees, many grades
        (6, min(max_k, 7)),       # adds 5-edge trees, cap at 7
        (7, 6),                   # adds 6-edge trees, cap at 6
    ]
    
    for n, mk in configs:
        coeffs, tinfo = extract_tree_coefficients(n, mk, verbose=True)
        
        for (k, tk), val in coeffs.items():
            if (k, tk) in all_coeffs:
                assert all_coeffs[(k, tk)] == val, \
                    f"Universality violation at grade {k}: " \
                    f"got {val}, expected {all_coeffs[(k, tk)]}"
            all_coeffs[(k, tk)] = val
        
        for tk, info in tinfo.items():
            if tk not in all_tree_info:
                all_tree_info[tk] = info
    
    print(f"\n\n{'#'*70}")
    print(f"# TREE COEFFICIENT LINEAR INDEPENDENCE ANALYSIS")
    print(f"{'#'*70}")
    
    analyze_independence(all_coeffs, all_tree_info)


if __name__ == "__main__":
    main()