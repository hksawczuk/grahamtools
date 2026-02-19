#!/usr/bin/env python3
"""
Tree coefficient rank analysis.

Three modes of computing / analyzing tree coefficient vectors:

  kn        Extract coefficients from L^k(K_n) (forest pruning).
  mobius    Derive coefficients via Mobius inversion over Prufer-generated trees.
  hardcoded Analyze hardcoded coefficient data with exact Gaussian elimination.

Usage:
    python tree_rank.py kn [--max-k MAX_K]
    python tree_rank.py mobius [--max-k 12] [--max-tree-edges 6] [--max-lg-edges 5000000]
    python tree_rank.py hardcoded
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from fractions import Fraction
from functools import reduce
from itertools import product
from math import gcd
from typing import Dict, List, Tuple

from grahamtools.kn import generate_levels_Kn_ids, expand_to_simple_base_edges_id
from grahamtools.kn.classify import canon_key
from grahamtools.utils.automorphisms import aut_size_edges
from grahamtools.utils.naming import tree_name, describe_graph
from grahamtools.utils.linalg import exact_rank, row_reduce_fraction
from grahamtools.utils.linegraph_edgelist import gamma_sequence_edgelist
from grahamtools.utils.subgraphs import enumerate_connected_subgraphs
from grahamtools.utils.connectivity import is_connected_edges


# ─────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────

def _classify_tree(edges, n):
    """Classify a tree by isomorphism type.  Returns a key or None."""
    m = len(edges)
    if m == 0:
        return None

    verts = set()
    for u, v in edges:
        verts.add(u)
        verts.add(v)
    nv = len(verts)

    if m != nv - 1:
        return None  # not a tree (disconnected forest)

    if not is_connected_edges(edges):
        return None

    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    deg_seq = tuple(sorted((len(adj[v]) for v in verts), reverse=True))

    # Canonical signature: sorted (degree, sorted-neighbor-degrees) per vertex
    vertex_sigs = []
    for v in sorted(verts):
        nbr_degs = tuple(sorted((len(adj[u]) for u in adj[v]), reverse=True))
        vertex_sigs.append((len(adj[v]), nbr_degs))
    vertex_sig = tuple(sorted(vertex_sigs, reverse=True))

    return (deg_seq, vertex_sig)


def _analyze_independence(coeffs, tree_info):
    """Shared rank analysis over a coefficient dict and tree_info dict."""
    if not tree_info:
        return

    trees_by_edges = defaultdict(list)
    for tk, (name, ne, nv) in tree_info.items():
        trees_by_edges[ne].append(tk)
    for ne in trees_by_edges:
        trees_by_edges[ne].sort(key=lambda tk: tree_info[tk][0])

    all_grades = sorted(set(k for (k, _) in coeffs.keys()))

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
                row_str = f"  k={k:2d}  " + "".join(
                    f"{M[ki][ti]:>14}" for ti in range(n_trees)
                )
                print(row_str)

        rank = exact_rank(M, n_rows, n_trees)
        print(f"\n  Rank: {rank} / {n_trees}", end="")
        if rank == n_trees:
            print(" -- FULL RANK")
        else:
            print(f" -- need {n_trees - rank} more rows")

        if n_trees > 1:
            print("  Cumulative rank:")
            for nk in range(1, min(n_rows + 1, n_trees + 3)):
                r = exact_rank(M[:nk], nk, n_trees)
                k_end = grades[nk - 1]
                print(f"    Through grade {k_end}: rank {r}")
                if r == n_trees:
                    print(f"    -> Full rank achieved at grade {k_end}")
                    break

    # All trees together
    all_tkeys = sorted(
        tree_info.keys(), key=lambda tk: (tree_info[tk][1], tree_info[tk][0])
    )
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
            row_str = f"  k={k:2d}  " + "".join(
                f"{M[ki][ti]:>12}" for ti in range(n_all)
            )
            print(row_str)

    rank = exact_rank(M, n_rows, n_all)
    print(f"\n  Rank: {rank} / {n_all}", end="")
    if rank == n_all:
        print(" -- FULL RANK")
    else:
        print(f" -- need {n_all - rank} more rows")

    print("  Cumulative rank:")
    for nk in range(1, min(n_rows + 1, n_all + 3)):
        r = exact_rank(M[:nk], nk, n_all)
        k_end = grades[nk - 1]
        print(f"    Through grade {k_end}: rank {r}")
        if r == n_all:
            print(f"    -> Full rank achieved at grade {k_end}")
            break


# ─────────────────────────────────────────────────────────────
#  Subcommand: kn
# ─────────────────────────────────────────────────────────────

def _extract_tree_coefficients(n, max_k, verbose=True):
    """Build L^k(K_n) with forest pruning, extract tree coefficients."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Computing tree coefficients in K_{n}, grades 1..{max_k}")
        print(f"{'='*60}")

    t_total = time.time()

    if verbose:
        print(f"  Building L^k(K_{n}) with prune_cycles=True...", flush=True)

    t0 = time.time()
    V_by_level, endpoints_by_level = generate_levels_Kn_ids(
        n, max_k, prune_cycles=True
    )
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

        # Classify each forest vertex -- only canonicalize trees
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
            key = canon_key(edges, n)
            if key not in buckets:
                buckets[key] = {"edges": edges, "freq": 0}
            buckets[key]["freq"] += 1

        # Extract tree coefficients from buckets
        n_tree_classes = 0
        for key, b in buckets.items():
            edges = b["edges"]
            tree_key = _classify_tree(edges, n)

            if tree_key is None:
                continue

            n_tree_classes += 1
            ne = len(edges)
            nv = len(set(u for e in edges for u in e))

            aut = aut_size_edges(edges, n)
            orbit = math.factorial(n) // aut
            freq = b["freq"]
            coeff = freq // orbit

            assert freq % orbit == 0, f"Non-integer coeff: {freq}/{orbit}"

            coeffs[(k, tree_key)] = coeff

            if tree_key not in tree_info:
                name = tree_name(edges, n)
                existing = [tk for tk in tree_info if tree_info[tk][0] == name]
                if existing:
                    name = f"{name}_{chr(ord('a') + len(existing))}"
                tree_info[tree_key] = (name, ne, nv)

        dt = time.time() - t0
        if verbose:
            print(
                f"    {n_trees_total} trees, {n_forests_skipped} non-tree "
                f"forests skipped, {n_tree_classes} iso classes ({dt:.1f}s)"
            )
            tree_coeffs = [
                (tree_info[tk][0], coeffs[(k, tk)])
                for tk in sorted(tree_info.keys(), key=lambda x: tree_info[x][0])
                if (k, tk) in coeffs and coeffs[(k, tk)] > 0
            ]
            for name, val in tree_coeffs:
                print(f"      {name}: {val}")

    if verbose:
        print(f"\n  Total time: {time.time()-t_total:.1f}s")

    return coeffs, tree_info


def cmd_kn(args):
    """Subcommand: extract tree coefficients from L^k(K_n) and analyze rank."""
    max_k = args.max_k
    all_coeffs = {}
    all_tree_info = {}

    configs = [
        (5, max_k),                # 3,4-edge trees, many grades
        (6, min(max_k, 7)),        # adds 5-edge trees, cap at 7
        (7, 6),                    # adds 6-edge trees, cap at 6
    ]

    for n, mk in configs:
        coeffs, tinfo = _extract_tree_coefficients(n, mk, verbose=True)

        for (k, tk), val in coeffs.items():
            if (k, tk) in all_coeffs:
                assert all_coeffs[(k, tk)] == val, (
                    f"Universality violation at grade {k}: "
                    f"got {val}, expected {all_coeffs[(k, tk)]}"
                )
            all_coeffs[(k, tk)] = val

        for tk, info in tinfo.items():
            if tk not in all_tree_info:
                all_tree_info[tk] = info

    print(f"\n\n{'#'*70}")
    print(f"# TREE COEFFICIENT LINEAR INDEPENDENCE ANALYSIS")
    print(f"{'#'*70}")

    _analyze_independence(all_coeffs, all_tree_info)


# ─────────────────────────────────────────────────────────────
#  Subcommand: mobius
# ─────────────────────────────────────────────────────────────

def _prufer_to_edges(seq):
    """Convert Prufer sequence to edge list."""
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
    last = [u for u in range(n) if degree[u] == 1]
    edges.append((last[0], last[1]))
    return edges


def _deg_seq(edges):
    """Degree sequence (descending) from an edge list."""
    d = defaultdict(int)
    for u, v in edges:
        d[u] += 1
        d[v] += 1
    return tuple(sorted(d.values(), reverse=True))


def _generate_trees(max_edges):
    """Generate all non-isomorphic trees with 1..max_edges edges.

    Uses Prufer sequences for exhaustive generation + canonical dedup.
    Returns list of (canon, edges, name, n_verts, n_edges).
    """
    from grahamtools.utils.canonical import (
        canonical_graph_bruteforce,
        canonical_graph_nauty,
    )
    from grahamtools.external.nauty import nauty_available

    def _canonical(edges):
        if not edges:
            return ()
        if nauty_available():
            return canonical_graph_nauty(edges)
        return canonical_graph_bruteforce(edges)

    all_trees = []
    seen_canons = set()

    for ne in range(1, max_edges + 1):
        nv = ne + 1
        count = 0

        if nv == 2:
            edges = [(0, 1)]
            canon = _canonical(edges)
            if canon not in seen_canons:
                seen_canons.add(canon)
                all_trees.append((canon, edges, "K2", nv, ne))
                count += 1
        elif nv == 3:
            edges = [(0, 1), (1, 2)]
            canon = _canonical(edges)
            if canon not in seen_canons:
                seen_canons.add(canon)
                all_trees.append((canon, edges, "P3", nv, ne))
                count += 1
        else:
            new_this_size = []
            for seq in product(range(nv), repeat=nv - 2):
                edges = _prufer_to_edges(list(seq))
                canon = _canonical(edges)
                if canon not in seen_canons:
                    seen_canons.add(canon)
                    new_this_size.append((canon, edges, nv, ne))
                    count += 1

            new_this_size.sort(key=lambda t: _deg_seq(t[1]))

            for canon, edges, nv2, ne2 in new_this_size:
                ds = _deg_seq(edges)
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
                existing = [
                    n for _, _, n, _, _ in all_trees
                    if n == name or n.startswith(name + "_")
                ]
                if existing:
                    name = f"{name}_{chr(ord('a') + len(existing))}"
                all_trees.append((canon, edges, name, nv2, ne2))

        print(f"  {ne}-edge trees: {count} types")

    return all_trees


def cmd_mobius(args):
    """Subcommand: derive coefficients via Mobius inversion."""
    max_k = args.max_k
    max_tree_edges = args.max_tree_edges
    max_lg_edges = args.max_lg_edges

    print(f"Generating trees with up to {max_tree_edges} edges...")
    trees = _generate_trees(max_tree_edges)
    print(f"  {len(trees)} trees total\n")

    trees.sort(key=lambda t: (t[4], t[2]))

    print(f"Enumerating connected subgraphs of all trees...")
    t0 = time.time()

    all_types = {}       # canon -> (example_edges, n_verts, n_edges)
    tree_subtypes = {}   # tree_canon -> {sub_canon: count}

    for canon, edges, name, nv, ne in trees:
        subs = enumerate_connected_subgraphs(edges, max_size=ne)
        tree_subtypes[canon] = {}
        for sub_canon, (count, sub_edges, sub_nv, sub_ne) in subs.items():
            tree_subtypes[canon][sub_canon] = count
            if sub_canon not in all_types:
                all_types[sub_canon] = (sub_edges, sub_nv, sub_ne)

    print(f"  {len(all_types)} connected subgraph types ({time.time()-t0:.1f}s)\n")

    sorted_types = sorted(all_types.keys(), key=lambda c: (all_types[c][2], c))

    print(f"Computing gamma sequences (max_k={max_k})...")
    gammas = {}
    for canon in sorted_types:
        sub_edges, sub_nv, sub_ne = all_types[canon]
        desc = describe_graph(sub_edges)
        t0 = time.time()
        gamma = gamma_sequence_edgelist(sub_edges, max_k, max_edges=max_lg_edges)
        dt = time.time() - t0
        gammas[canon] = gamma
        max_valid = max(
            (k for k in range(len(gamma)) if gamma[k] is not None), default=0
        )
        if dt > 0.1:
            print(
                f"  {desc} ({sub_ne}e): gamma computed through k={max_valid} "
                f"({dt:.1f}s)"
            )

    print()

    # Mobius inversion: process in order of increasing edge count
    print(f"Computing coefficients via Mobius inversion...")
    coeffs = {}  # canon -> [coeff_0, coeff_1, ..., coeff_{max_k}]

    for canon in sorted_types:
        sub_edges, sub_nv, sub_ne = all_types[canon]
        gamma = gammas[canon]
        coeff = list(gamma)

        proper_subs = (
            enumerate_connected_subgraphs(sub_edges, max_size=sub_ne - 1)
            if sub_ne > 1
            else {}
        )

        for sub_canon, (count, _, _, _) in proper_subs.items():
            if sub_canon in coeffs:
                for k in range(min(len(coeff), len(coeffs[sub_canon]))):
                    if coeff[k] is not None and coeffs[sub_canon][k] is not None:
                        coeff[k] -= coeffs[sub_canon][k] * count

        coeffs[canon] = coeff

    print()

    # Display coefficient matrix
    print(f"{'#'*70}")
    print(f"# TREE COEFFICIENT MATRIX")
    print(f"{'#'*70}")

    tree_canons = [canon for canon, edges, name, nv, ne in trees]
    tree_names = {canon: name for canon, edges, name, nv, ne in trees}
    tree_nedges = {canon: ne for canon, edges, name, nv, ne in trees}

    print(f"\nCoefficient sequences:")
    for canon, edges, name, nv, ne in trees:
        coeff = coeffs.get(canon, [])
        vals = []
        for k in range(max_k + 1):
            if k < len(coeff) and coeff[k] is not None:
                vals.append(str(coeff[k]))
            else:
                vals.append("-")
        print(f"  {name:>10s} ({ne}e): {', '.join(vals[1:])}")

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

    names_list = [tree_names[c] for c in tree_canons]
    col_w = max(max(len(n) for n in names_list), 10) + 2

    print(f"\nFull matrix:")
    header = "  grade" + "".join(f"{n:>{col_w}}" for n in names_list)
    print(header)

    for k in valid_grades:
        vals = []
        for canon in tree_canons:
            c = coeffs.get(canon, [])
            if k < len(c) and c[k] is not None:
                vals.append(f"{c[k]:>{col_w}}")
            else:
                vals.append(f"{'-':>{col_w}}")
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
            print(
                f"\n  {ne}-edge trees: 1 type ({tnames[0]}) "
                f"-- trivially independent"
            )
            continue

        grades = [k for k in valid_grades if k >= ne]
        M = []
        for k in grades:
            row = []
            for canon in tkeys:
                c = coeffs.get(canon, [])
                if k < len(c) and c[k] is not None:
                    row.append(c[k])
                else:
                    row.append(0)
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
                    row_str += f"{'-':>{col_w}}"
            print(row_str)

        rank = exact_rank(M, n_rows, n_trees)
        print(f"\n  Rank: {rank} / {n_trees}", end="")
        if rank == n_trees:
            print(" -- FULL RANK")
        else:
            print(f" -- need {n_trees - rank} more")

        print("  Cumulative rank:")
        for nk in range(1, min(n_rows + 1, n_trees + 3)):
            r = exact_rank(M[:nk], nk, n_trees)
            print(f"    Through grade {grades[nk-1]}: rank {r}")
            if r == n_trees:
                print(f"    -> Full rank at grade {grades[nk-1]}")
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
        print(" -- FULL RANK")
    else:
        print(f" -- need {n_all - rank} more")

    print("  Cumulative rank:")
    for nk in range(1, min(n_rows + 1, n_all + 3)):
        r = exact_rank(M[:nk], nk, n_all)
        print(f"    Through grade {grades[nk-1]}: rank {r}")
        if r == n_all:
            print(f"    -> Full rank at grade {grades[nk-1]}")
            break


# ─────────────────────────────────────────────────────────────
#  Subcommand: hardcoded
# ─────────────────────────────────────────────────────────────

# Coefficient data from k=8 computation (acyclic types only).
# Format: (label, n_edges, [c1, c2, c3, c4, c5, c6, c7, c8])

_HARDCODED_TREE_TYPES = [
    ("K2",                    1, [1, 0, 0, 0, 0, 0, 0, 0]),
    ("P3",                    2, [0, 1, 0, 0, 0, 0, 0, 0]),
    ("P4",                    3, [0, 0, 1, 0, 0, 0, 0, 0]),
    ("K_{1,3}",               3, [0, 0, 3, 3, 3, 3, 3, 3]),
    ("P5",                    4, [0, 0, 0, 1, 0, 0, 0, 0]),
    ("T_{3,2,1,1,1} (cat)",   4, [0, 0, 0, 5, 15, 61, 393, 4549]),
    ("P6",                    5, [0, 0, 0, 0, 1, 0, 0, 0]),
    ("T_{3,2,2,1,1,1} #1",   5, [0, 0, 0, 0, 7, 39, 364, 6058]),
    ("T_{3,2,2,1,1,1} #2",   5, [0, 0, 0, 0, 11, 114, 1615, 37576]),
    ("T_{3,3,1,1,1,1}",      5, [0, 0, 0, 0, 48, 656, 12120, 386864]),
]


def cmd_hardcoded(args):
    """Subcommand: analyze hardcoded coefficient data."""
    N = len(_HARDCODED_TREE_TYPES)
    K = 8

    labels = [t[0] for t in _HARDCODED_TREE_TYPES]
    cvecs = [t[2] for t in _HARDCODED_TREE_TYPES]

    print("=" * 70)
    print("  Rank Analysis: Acyclic (Tree) Types Only")
    print("=" * 70)

    print(f"\n  {N} tree types, {K} grades")

    # Print the coefficient matrix
    print(f"\n  {'Type':>30s}", end="")
    for k in range(1, K + 1):
        print(f" {'c'+str(k):>8s}", end="")
    print()
    print(f"  {'-' * (30 + 9*K)}")
    for i in range(N):
        print(f"  {labels[i]:>30s}", end="")
        for k in range(K):
            if cvecs[i][k] != 0:
                print(f" {cvecs[i][k]:>8d}", end="")
            else:
                print(f" {'':>8s}", end="")
        print()

    # Build exact rational matrix M (K x N) and row-reduce
    M = []
    for k in range(K):
        row = [Fraction(cvecs[j][k]) for j in range(N)]
        M.append(row)

    mat, pivot_cols, rank = row_reduce_fraction(M, K, N)

    free_cols = [j for j in range(N) if j not in pivot_cols]

    print(f"\n  Rank: {rank}")
    print(f"  Null space dimension: {N - rank}")

    print(f"\n  Pivot types ({rank}):")
    for pc in pivot_cols:
        print(f"    {labels[pc]}")

    if free_cols:
        print(f"\n  Free types ({len(free_cols)}):")
        for fc in free_cols:
            print(f"    {labels[fc]}")

        def lcm(a, b):
            return a * b // gcd(a, b)

        print(f"\n  Null space basis (integer-scaled):")
        for fi, fc in enumerate(free_cols):
            null_vec = [Fraction(0)] * N
            null_vec[fc] = Fraction(1)
            for pi, pc in enumerate(pivot_cols):
                null_vec[pc] = -mat[pi][fc]

            # Scale to integers
            denoms = [abs(v.denominator) for v in null_vec if v != 0]
            if denoms:
                lcd_val = reduce(lcm, denoms)
                scaled = [int(v * lcd_val) for v in null_vec]
                nums = [abs(s) for s in scaled if s != 0]
                common = reduce(gcd, nums) if nums else 1
                scaled = [s // common for s in scaled]
            else:
                scaled = [0] * N

            # Verify
            check = all(
                sum(M[k][j] * null_vec[j] for j in range(N)) == 0
                for k in range(K)
            )

            print(f"\n    v_{fi+1} (free: {labels[fc]}), verified={check}:")
            for j in range(N):
                if scaled[j] != 0:
                    print(f"      {labels[j]:>30s}: {scaled[j]:+d}")
    else:
        print(f"\n  *** FULL RANK: no null vectors among tree types! ***")

    # Rank growth grade by grade
    print(f"\n  {'='*60}")
    print(f"  Rank growth by grade")
    print(f"  {'='*60}")

    for max_grade in range(1, K + 1):
        M_sub = []
        for k in range(max_grade):
            row = [Fraction(cvecs[j][k]) for j in range(N)]
            M_sub.append(row)

        _, pivots_sub, r = row_reduce_fraction(M_sub, max_grade, N)
        pivot_names = [labels[p] for p in pivots_sub]
        print(f"    K={max_grade}: rank={r}, pivots={pivot_names}")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tree coefficient rank analysis",
    )
    subparsers = parser.add_subparsers(dest="command")

    # kn
    kn_p = subparsers.add_parser(
        "kn",
        help="Extract tree coefficients from L^k(K_n) and analyze rank",
    )
    kn_p.add_argument(
        "--max-k", type=int, default=10, help="Maximum grade to compute"
    )

    # mobius
    mob_p = subparsers.add_parser(
        "mobius",
        help="Derive coefficients via Mobius inversion over Prufer trees",
    )
    mob_p.add_argument("--max-k", type=int, default=12)
    mob_p.add_argument("--max-tree-edges", type=int, default=6)
    mob_p.add_argument("--max-lg-edges", type=int, default=5_000_000)

    # hardcoded
    subparsers.add_parser(
        "hardcoded",
        help="Analyze hardcoded k=8 coefficient data",
    )

    args = parser.parse_args()

    if args.command is None:
        # Default to kn
        args.command = "kn"
        args.max_k = 10

    if args.command == "kn":
        cmd_kn(args)
    elif args.command == "mobius":
        cmd_mobius(args)
    elif args.command == "hardcoded":
        cmd_hardcoded(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
