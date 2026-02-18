#!/usr/bin/env python3
"""
Combined script: runs the full fiber analysis, then immediately
performs the linear dependence analysis on the exact computed data.
No manual transcription needed.

Usage: python3 wl1_combined_analysis.py [--max-k K]
"""

import sys
import time
from collections import defaultdict
from itertools import combinations, permutations
from fractions import Fraction
from math import gcd
from functools import reduce


# ============================================================
#  Graph utilities (same as wl1_fiber_analysis.py)
# ============================================================

def canonical_graph(edges, vertices=None):
    if not edges:
        n = len(vertices) if vertices else 0
        return ("empty", n)
    if vertices is None:
        vertices = set()
        for u, v in edges:
            vertices.add(u)
            vertices.add(v)
    vlist = sorted(vertices)
    n = len(vlist)
    if n > 10:
        return _canon_large(edges, vlist)
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

def _canon_large(edges, vlist):
    n = len(vlist)
    v_map = {v: i for i, v in enumerate(vlist)}
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[v_map[u]].add(v_map[v])
        adj[v_map[v]].add(v_map[u])
    deg_seq = tuple(sorted([len(adj[i]) for i in range(n)], reverse=True))
    edge_tup = tuple(sorted(
        (min(v_map[u], v_map[v]), max(v_map[u], v_map[v]))
        for u, v in edges
    ))
    return (deg_seq, edge_tup)

def is_connected(edges, vertices=None):
    if not edges:
        return vertices is not None and len(vertices) <= 1
    adj = defaultdict(set)
    verts = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        verts.add(u)
        verts.add(v)
    if vertices is not None:
        verts = set(vertices)
    start = next(iter(verts))
    visited = set()
    stack = [start]
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        for u in adj[v]:
            if u in verts and u not in visited:
                stack.append(u)
    return len(visited) == len(verts)

def enumerate_connected_subgraphs(edges, max_size=None):
    n_edges = len(edges)
    if max_size is None:
        max_size = n_edges
    counts = defaultdict(lambda: [0, None, 0, 0])
    for size in range(1, min(max_size, n_edges) + 1):
        for subset in combinations(range(n_edges), size):
            sub_edges = [edges[i] for i in subset]
            if not is_connected(sub_edges):
                continue
            canon = canonical_graph(sub_edges)
            entry = counts[canon]
            entry[0] += 1
            if entry[1] is None:
                verts = set()
                for u, v in sub_edges:
                    verts.add(u)
                    verts.add(v)
                entry[1] = sub_edges
                entry[2] = len(verts)
                entry[3] = size
    return {k: tuple(v) for k, v in counts.items()}

def describe_graph(edges, n_vertices):
    m = len(edges)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    verts = set()
    for u, v in edges:
        verts.add(u)
        verts.add(v)
    n = len(verts)
    deg_seq = sorted([len(adj[v]) for v in verts], reverse=True)
    if m == 1:
        return "K2"
    if m == n - 1:
        if all(d <= 2 for d in deg_seq):
            return f"P{n}"
        if deg_seq.count(1) == n - 1:
            return f"K1,{n-1}"
        return f"Tree({n}v,ds={deg_seq})"
    if all(d == 2 for d in deg_seq) and m == n:
        return f"C{n}"
    if m == n:
        return f"Unicyclic({n}v,{m}e,ds={deg_seq})"
    return f"Graph({n}v,{m}e,ds={deg_seq})"

def line_graph(edges, n_vertices):
    m = len(edges)
    if m == 0:
        return [], 0
    incident = defaultdict(list)
    for idx, (u, v) in enumerate(edges):
        incident[u].append(idx)
        incident[v].append(idx)
    new_edges = set()
    for v in range(n_vertices):
        inc = incident.get(v, [])
        for i in range(len(inc)):
            for j in range(i + 1, len(inc)):
                a, b = inc[i], inc[j]
                if a > b: a, b = b, a
                new_edges.add((a, b))
    return sorted(new_edges), m

def gamma_sequence(edges, max_k, max_edges=2_000_000):
    if not edges:
        return [0] * (max_k + 1)
    verts = set()
    for u, v in edges:
        verts.add(u)
        verts.add(v)
    n = len(verts)
    v_map = {v: i for i, v in enumerate(sorted(verts))}
    current_edges = [(v_map[u], v_map[v]) for u, v in edges]
    current_n = n
    seq = [current_n]
    for k in range(1, max_k + 1):
        new_edges, new_n = line_graph(current_edges, current_n)
        seq.append(new_n)
        if new_n == 0 or len(new_edges) > max_edges:
            while len(seq) <= max_k:
                seq.append(None)
            break
        current_edges = new_edges
        current_n = new_n
    return seq


# ============================================================
#  Main
# ============================================================

def main():
    max_k = 7
    max_edges = 2_000_000
    
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--max-k" and i + 1 < len(args):
            max_k = int(args[i + 1]); i += 2
        elif args[i] == "--max-edges" and i + 1 < len(args):
            max_edges = int(args[i + 1]); i += 2
        else:
            i += 1
    
    # Define graphs
    dumbbell = [(0,1), (1,2), (0,2), (2,3), (3,4), (4,5), (3,5)]
    chorded_c6 = [(0,1), (1,2), (2,3), (3,4), (4,5), (0,5), (0,3)]
    
    # Enumerate subgraph types
    print("Enumerating subgraph types...", flush=True)
    types_d = enumerate_connected_subgraphs(dumbbell)
    types_c = enumerate_connected_subgraphs(chorded_c6)
    
    all_types = {}
    for canon, (count, edges, nv, ne) in types_d.items():
        all_types[canon] = (count, edges, nv, ne)
    for canon, (count, edges, nv, ne) in types_c.items():
        if canon not in all_types:
            all_types[canon] = (count, edges, nv, ne)
    
    all_canons = sorted(all_types.keys(), key=lambda c: (all_types[c][3], c))
    
    print(f"  {len(all_types)} types total\n")
    
    # Compute coefficients by bootstrapping
    print(f"Computing fiber coefficients (max_k={max_k})...\n", flush=True)
    
    sorted_types = sorted(all_types.items(), key=lambda x: x[1][3])
    gammas = {}
    coeffs = {}   # canon -> list of int/None
    subtypes_map = {}
    
    for idx, (canon, (count, edges, nv, ne)) in enumerate(sorted_types):
        desc = describe_graph(edges, nv)
        gamma = gamma_sequence(edges, max_k, max_edges)
        gammas[canon] = gamma
        
        if ne > 1:
            sub_counts = enumerate_connected_subgraphs(edges, max_size=ne - 1)
        else:
            sub_counts = {}
        subtypes_map[canon] = {sc: sv[0] for sc, sv in sub_counts.items()}
        
        coeff = list(gamma)
        for sub_canon, sub_count in subtypes_map[canon].items():
            if sub_canon in coeffs:
                for k in range(min(len(coeff), len(coeffs[sub_canon]))):
                    if coeff[k] is not None and coeffs[sub_canon][k] is not None:
                        coeff[k] -= coeffs[sub_canon][k] * sub_count
        
        coeffs[canon] = coeff
        
        nonzero = [(k, c) for k, c in enumerate(coeff) if c is not None and c != 0 and k > 0]
        if nonzero:
            nz_str = ", ".join(f"c_{k}={c}" for k, c in nonzero[:6])
            print(f"  [{idx+1}/{len(sorted_types)}] {desc} ({ne}e): {nz_str}")
        else:
            print(f"  [{idx+1}/{len(sorted_types)}] {desc} ({ne}e): all zero")
    
    # Verify decomposition
    print(f"\nVerifying decomposition...", flush=True)
    for name, graph_edges, graph_types, graph_seq_fn in [
        ("Dumbbell", dumbbell, types_d, lambda: gamma_sequence(dumbbell, max_k, max_edges)),
        ("Chorded C6", chorded_c6, types_c, lambda: gamma_sequence(chorded_c6, max_k, max_edges)),
    ]:
        graph_seq = graph_seq_fn()
        ok = True
        for k in range(1, max_k + 1):
            if graph_seq[k] is None:
                break
            total = 0
            for canon in all_canons:
                count = graph_types.get(canon, (0,))[0]
                if count == 0:
                    continue
                ck = coeffs.get(canon, [])
                if k < len(ck) and ck[k] is not None:
                    total += ck[k] * count
            if total != graph_seq[k]:
                print(f"  {name} k={k}: MISMATCH Σ={total} vs γ={graph_seq[k]}")
                ok = False
        if ok:
            print(f"  {name}: all grades verified ✓")
    
    # ============================================================
    #  LINEAR DEPENDENCE ANALYSIS
    # ============================================================
    
    print(f"\n{'=' * 70}")
    print(f"  Linear Dependence Analysis")
    print(f"{'=' * 70}")
    
    # Build data structures
    type_info = []  # list of (canon, label, n_edges, coeff_vec, count_D, count_C, delta)
    
    for canon in all_canons:
        _, edges, nv, ne = all_types[canon]
        label = describe_graph(edges, nv)
        
        # Make labels unique by appending canonical form hash
        cd = types_d.get(canon, (0,))[0]
        cc = types_c.get(canon, (0,))[0]
        delta = cd - cc
        
        coeff_vec = []
        for k in range(1, max_k + 1):
            ck = coeffs.get(canon, [])
            if k < len(ck) and ck[k] is not None:
                coeff_vec.append(ck[k])
            else:
                coeff_vec.append(0)
        
        type_info.append((canon, label, ne, coeff_vec, cd, cc, delta))
    
    # Make labels unique
    label_counts = defaultdict(int)
    for ti in type_info:
        label_counts[ti[1]] += 1
    
    label_seen = defaultdict(int)
    unique_labels = []
    for ti in type_info:
        label = ti[1]
        if label_counts[label] > 1:
            label_seen[label] += 1
            unique_labels.append(f"{label} #{label_seen[label]}")
        else:
            unique_labels.append(label)
    
    # Print the full data table
    print(f"\n  {'#':>3s} {'Label':>45s} {'|E|':>4s} {'cnt_D':>6s} {'cnt_C':>6s} {'Δ':>4s}", end="")
    for k in range(1, max_k + 1):
        print(f" {'c_'+str(k):>8s}", end="")
    print()
    print(f"  {'-' * (70 + 9*max_k)}")
    
    for i, (canon, label, ne, cvec, cd, cc, delta) in enumerate(type_info):
        ulabel = unique_labels[i]
        print(f"  {i+1:>3d} {ulabel:>45s} {ne:>4d} {cd:>6d} {cc:>6d} {delta:>+4d}", end="")
        for k in range(max_k):
            if cvec[k] != 0:
                print(f" {cvec[k]:>8d}", end="")
            else:
                print(f" {'·':>8s}", end="")
        print()
    
    # Restrict to types with nonzero Delta
    nz_indices = [i for i, ti in enumerate(type_info) if ti[6] != 0]
    n_nz = len(nz_indices)
    
    print(f"\n  Types with Δ ≠ 0: {n_nz}")
    
    # Build coefficient matrix M (max_k x n_nz) with exact rationals
    M = []
    for k in range(max_k):
        row = []
        for j in nz_indices:
            row.append(Fraction(type_info[j][3][k]))
        M.append(row)
    
    Delta_vec = [Fraction(type_info[j][6]) for j in nz_indices]
    nz_labels = [unique_labels[j] for j in nz_indices]
    
    # Verify M @ Delta = 0
    print(f"\n  Exact verification M · Δ:")
    all_zero = True
    for k in range(max_k):
        val = sum(M[k][j] * Delta_vec[j] for j in range(n_nz))
        status = "✓" if val == 0 else f"✗ ({val})"
        print(f"    k={k+1}: {status}")
        if val != 0:
            all_zero = False
    
    if not all_zero:
        print(f"\n  *** Δ is NOT in the null space! ***")
        print(f"  This indicates a data issue. Stopping here.")
        return
    
    print(f"\n  Δ is confirmed in the null space of M.")
    
    # Row reduce to find rank and null space
    mat = [row[:] for row in M]  # copy
    n_rows = max_k
    n_cols = n_nz
    
    pivot_cols = []
    current_row = 0
    for col in range(n_cols):
        pivot = None
        for row in range(current_row, n_rows):
            if mat[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            continue
        mat[current_row], mat[pivot] = mat[pivot], mat[current_row]
        pivot_cols.append(col)
        scale = mat[current_row][col]
        for j in range(n_cols):
            mat[current_row][j] /= scale
        for row in range(n_rows):
            if row == current_row:
                continue
            factor = mat[row][col]
            if factor != 0:
                for j in range(n_cols):
                    mat[row][j] -= factor * mat[current_row][j]
        current_row += 1
    
    rank = len(pivot_cols)
    free_cols = [j for j in range(n_cols) if j not in pivot_cols]
    
    print(f"\n  Rank of M (restricted): {rank}")
    print(f"  Null space dimension: {n_nz - rank}")
    print(f"\n  Pivot types ({rank}):")
    for pc in pivot_cols:
        print(f"    {nz_labels[pc]}")
    print(f"\n  Free types ({len(free_cols)}):")
    for fc in free_cols:
        print(f"    {nz_labels[fc]}")
    
    # Extract null space basis
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    print(f"\n  {'=' * 60}")
    print(f"  Null Space Basis (exact, integer-scaled)")
    print(f"  {'=' * 60}")
    
    null_basis = []
    for fi, fc in enumerate(free_cols):
        null_vec = [Fraction(0)] * n_nz
        null_vec[fc] = Fraction(1)
        for pi, pc in enumerate(pivot_cols):
            null_vec[pc] = -mat[pi][fc]
        
        null_basis.append(null_vec)
        
        # Scale to integers
        denoms = [abs(v.denominator) for v in null_vec if v != 0]
        if denoms:
            lcd_val = reduce(lcm, denoms)
            scaled = [int(v * lcd_val) for v in null_vec]
            nums = [abs(s) for s in scaled if s != 0]
            common = reduce(gcd, nums) if nums else 1
            scaled = [s // common for s in scaled]
        else:
            scaled = [0] * n_nz
        
        # Verify
        check = all(
            sum(M[k][j] * null_vec[j] for j in range(n_nz)) == 0
            for k in range(max_k)
        )
        
        print(f"\n  Null vector {fi + 1} (free: {nz_labels[fc]}), verified={check}:")
        for j in range(n_nz):
            if scaled[j] != 0:
                print(f"    {nz_labels[j]:>45s}: {scaled[j]:+d}")
    
    # Express Delta as combination of null vectors
    print(f"\n  {'=' * 60}")
    print(f"  Δ as linear combination of null basis")
    print(f"  {'=' * 60}")
    
    # Delta[free_col] gives the coefficient of that null vector
    print(f"\n  Δ = ", end="")
    terms = []
    for fi, fc in enumerate(free_cols):
        coeff_val = Delta_vec[fc]
        if coeff_val != 0:
            terms.append(f"({coeff_val}) · v_{fi+1}")
    print(" + ".join(terms))
    
    # Verify reconstruction
    reconstructed = [Fraction(0)] * n_nz
    for fi, fc in enumerate(free_cols):
        for j in range(n_nz):
            reconstructed[j] += Delta_vec[fc] * null_basis[fi][j]
    
    match = all(reconstructed[j] == Delta_vec[j] for j in range(n_nz))
    print(f"  Reconstruction matches: {match}")
    
    if match:
        print(f"\n  Expanded:")
        for j in range(n_nz):
            if Delta_vec[j] != 0:
                print(f"    {nz_labels[j]:>45s}: Δ = {Delta_vec[j]:>+3}, reconstructed = {reconstructed[j]:>+3}  ✓")
    
    # ============================================================
    #  Summary: which coefficient vectors are linearly dependent?
    # ============================================================
    
    print(f"\n  {'=' * 60}")
    print(f"  Summary of Linear Dependencies")
    print(f"  {'=' * 60}")
    
    # Check for identical coefficient vectors
    print(f"\n  Types with identical coefficient vectors:")
    for i in range(n_nz):
        for j in range(i + 1, n_nz):
            cvec_i = [M[k][i] for k in range(max_k)]
            cvec_j = [M[k][j] for k in range(max_k)]
            if cvec_i == cvec_j:
                print(f"    {nz_labels[i]}  =  {nz_labels[j]}")
                print(f"      vector: {[int(v) for v in cvec_i]}")
    
    # Check for proportional coefficient vectors
    print(f"\n  Types with proportional coefficient vectors:")
    for i in range(n_nz):
        for j in range(i + 1, n_nz):
            cvec_i = [M[k][i] for k in range(max_k)]
            cvec_j = [M[k][j] for k in range(max_k)]
            # Find first nonzero in both
            ratio = None
            proportional = True
            for k in range(max_k):
                if cvec_i[k] == 0 and cvec_j[k] == 0:
                    continue
                if cvec_i[k] == 0 or cvec_j[k] == 0:
                    proportional = False
                    break
                r = cvec_i[k] / cvec_j[k]
                if ratio is None:
                    ratio = r
                elif r != ratio:
                    proportional = False
                    break
            if proportional and ratio is not None and ratio != 1:
                print(f"    {nz_labels[i]} = ({ratio}) × {nz_labels[j]}")


if __name__ == "__main__":
    main()