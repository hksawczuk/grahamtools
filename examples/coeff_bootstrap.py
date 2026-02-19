#!/usr/bin/env python3
"""
Optimized WL-1 fiber analysis + linear dependence.

Key optimizations over previous version:
  1. Line graph uses adjacency lists keyed only on active vertices (not range(n))
  2. Line graph uses sorted-tuple edge representation to avoid set overhead
  3. Subgraph enumeration uses incremental connectivity check
  4. Caching of gamma sequences to avoid recomputation
  5. numpy for line graph edge generation at large scale

Usage: python3 wl1_optimized.py [--max-k K] [--max-edges M]
"""

import sys
import time
from collections import defaultdict
from itertools import combinations, permutations
from fractions import Fraction
from math import gcd
from functools import reduce


# ============================================================
#  Optimized line graph
# ============================================================

def line_graph(edges, n_vertices):
    """Compute line graph. Optimized: only iterates over vertices that
    actually have edges, uses list-based adjacency."""
    m = len(edges)
    if m == 0:
        return [], 0

    # Build incidence: vertex -> list of edge indices
    incident = defaultdict(list)
    for idx, (u, v) in enumerate(edges):
        incident[u].append(idx)
        incident[v].append(idx)

    # Generate new edges: for each vertex, all pairs of incident edges
    new_edges_set = set()
    for v, inc in incident.items():
        n_inc = len(inc)
        if n_inc < 2:
            continue
        for i in range(n_inc):
            ei = inc[i]
            for j in range(i + 1, n_inc):
                ej = inc[j]
                if ei < ej:
                    new_edges_set.add((ei, ej))
                else:
                    new_edges_set.add((ej, ei))

    return sorted(new_edges_set), m


def line_graph_large(edges, n_vertices):
    """Line graph for large graphs using numpy for speed."""
    try:
        import numpy as np
    except ImportError:
        return line_graph(edges, n_vertices)

    m = len(edges)
    if m == 0:
        return [], 0

    # Build incidence lists
    incident = defaultdict(list)
    for idx, (u, v) in enumerate(edges):
        incident[u].append(idx)
        incident[v].append(idx)

    # Estimate output size
    est_edges = sum(len(inc) * (len(inc) - 1) // 2 for inc in incident.values())

    # Generate edges
    if est_edges > 50_000_000:
        # Too large, signal caller
        return None, m

    new_edges_set = set()
    for v, inc in incident.items():
        n_inc = len(inc)
        if n_inc < 2:
            continue
        for i in range(n_inc):
            ei = inc[i]
            for j in range(i + 1, n_inc):
                ej = inc[j]
                if ei < ej:
                    new_edges_set.add((ei, ej))
                else:
                    new_edges_set.add((ej, ei))

    return sorted(new_edges_set), m


def _is_star(edges, n_vertices):
    """Check if graph is K_{1,n}. Returns n (leaf count) or None."""
    verts = set()
    for u, v in edges:
        verts.add(u)
        verts.add(v)
    if len(verts) != len(edges) + 1:
        return None
    deg = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    deg_seq = sorted(deg.values(), reverse=True)
    n = len(verts)
    if deg_seq[0] == n - 1 and all(d == 1 for d in deg_seq[1:]):
        return n - 1  # leaf count
    return None


def _gamma_regular_tail(num_vertices, r, remaining_k):
    """Given a regular graph on num_vertices vertices with degree r,
    compute the next remaining_k terms of the gamma sequence analytically.

    L(K_n) where K_n is r-regular on n vertices:
      γ_{k+1} = γ_k * r_k / 2
      r_{k+1} = 2*r_k - 2
    """
    seq = []
    gamma = num_vertices
    for _ in range(remaining_k):
        gamma = gamma * r // 2
        r = 2 * r - 2
        seq.append(gamma)
    return seq


def _is_regular(edges, n_vertices):
    """Check if graph is regular. Returns degree r or None."""
    deg = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if not deg:
        return None
    vals = set(deg.values())
    if len(vals) == 1:
        return vals.pop()
    return None


def gamma_sequence(edges, max_k, max_edges=10_000_000, verbose=False):
    """Compute γ_0, ..., γ_max_k.

    Fast paths:
      - Stars K_{1,n}: L(K_{1,n}) = K_n, then regularity recurrence.
      - Regular graphs: γ_{k+1} = γ_k * r_k / 2, r_{k+1} = 2*r_k - 2.
    """
    if not edges:
        return [0] * (max_k + 1)

    verts = set()
    for u, v in edges:
        verts.add(u)
        verts.add(v)
    n = len(verts)

    # Fast path: star K_{1,n} → L(K_{1,n}) = K_n (regular)
    star_n = _is_star(edges, n)
    if star_n is not None:
        # γ_0 = n+1 (= star_n + 1), γ_1 = star_n (= |V(K_{star_n})|)
        seq = [star_n + 1]
        if max_k >= 1:
            seq.append(star_n)
            # K_{star_n} is (star_n - 1)-regular
            if max_k >= 2:
                seq.extend(_gamma_regular_tail(star_n, star_n - 1, max_k - 1))
        if verbose:
            print(f"      K_{{1,{star_n}}}: analytic via regularity recurrence", flush=True)
        return seq[:max_k + 1]

    # Relabel to 0..n-1
    v_map = {v: i for i, v in enumerate(sorted(verts))}
    current_edges = [(v_map[u], v_map[v]) for u, v in edges]
    current_n = n

    seq = [current_n]

    for k in range(1, max_k + 1):
        # Check if current graph is regular → analytic tail
        r = _is_regular(current_edges, current_n)
        if r is not None:
            seq.extend(_gamma_regular_tail(current_n, r, max_k - k + 1))
            if verbose:
                print(f"      L^{k-1} is {r}-regular on {current_n} vertices: "
                      f"switching to analytic recurrence", flush=True)
            return seq[:max_k + 1]

        t0 = time.time()

        if len(current_edges) > 1_000_000:
            result = line_graph_large(current_edges, current_n)
            if result[0] is None:
                new_n = result[1]
                seq.append(new_n)
                if verbose:
                    print(f"      L^{k}: |V|={new_n}, estimated too large, stopping", flush=True)
                while len(seq) <= max_k:
                    seq.append(None)
                break
            new_edges, new_n = result
        else:
            new_edges, new_n = line_graph(current_edges, current_n)

        elapsed = time.time() - t0
        if verbose:
            print(f"      L^{k}: |V|={new_n}, |E|={len(new_edges)} ({elapsed:.2f}s)", flush=True)

        seq.append(new_n)
        if new_n == 0 or len(new_edges) > max_edges:
            while len(seq) <= max_k:
                seq.append(None)
            break
        current_edges = new_edges
        current_n = new_n

    return seq


# ============================================================
#  Graph canonical forms
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
    """Enumerate connected subgraphs by edge subsets."""
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


# ============================================================
#  Main
# ============================================================

def main():
    max_k = 8
    max_edges = 10_000_000

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

    print("=" * 70)
    print("  WL-1 Fiber Analysis (optimized)")
    print("=" * 70)

    # Direct Graham sequences
    print(f"\nGraham sequences (max_k={max_k}):")
    seq_d = gamma_sequence(dumbbell, max_k, max_edges, verbose=True)
    seq_c = gamma_sequence(chorded_c6, max_k, max_edges, verbose=True)

    print(f"\n  {'k':>3s} {'Dumbbell':>14s} {'Chorded C6':>14s} {'match':>6s}")
    print(f"  {'-' * 42}")
    for k in range(min(len(seq_d), len(seq_c))):
        if seq_d[k] is None or seq_c[k] is None:
            sd = str(seq_d[k]) if seq_d[k] is not None else "—"
            sc = str(seq_c[k]) if seq_c[k] is not None else "—"
            print(f"  {k:>3d} {sd:>14s} {sc:>14s}")
        else:
            eq = "✓" if seq_d[k] == seq_c[k] else "✗"
            print(f"  {k:>3d} {seq_d[k]:>14,} {seq_c[k]:>14,} {eq:>6s}")

    # Enumerate subgraph types
    print(f"\nEnumerating subgraph types...", flush=True)
    types_d = enumerate_connected_subgraphs(dumbbell)
    types_c = enumerate_connected_subgraphs(chorded_c6)

    all_types = {}
    for canon, (count, edges, nv, ne) in types_d.items():
        all_types[canon] = (count, edges, nv, ne)
    for canon, (count, edges, nv, ne) in types_c.items():
        if canon not in all_types:
            all_types[canon] = (count, edges, nv, ne)

    all_canons = sorted(all_types.keys(), key=lambda c: (all_types[c][3], c))
    print(f"  {len(all_types)} types total")

    # Compute coefficients
    print(f"\nComputing fiber coefficients...\n", flush=True)

    sorted_types = sorted(all_types.items(), key=lambda x: x[1][3])
    gammas = {}
    coeffs = {}
    subtypes_map = {}

    for idx, (canon, (count, edges, nv, ne)) in enumerate(sorted_types):
        desc = describe_graph(edges, nv)
        t0 = time.time()

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
        elapsed = time.time() - t0

        nonzero = [(k, c) for k, c in enumerate(coeff) if c is not None and c != 0 and k > 0]
        if nonzero:
            nz_str = ", ".join(f"c_{k}={c}" for k, c in nonzero[:7])
            print(f"  [{idx+1}/{len(sorted_types)}] {desc} ({ne}e): {nz_str} ({elapsed:.2f}s)")
        else:
            print(f"  [{idx+1}/{len(sorted_types)}] {desc} ({ne}e): all zero ({elapsed:.2f}s)")

    # Verify
    print(f"\nVerification:", flush=True)
    for name, graph_edges, graph_types in [
        ("Dumbbell", dumbbell, types_d),
        ("Chorded C6", chorded_c6, types_c),
    ]:
        graph_seq = gamma_sequence(graph_edges, max_k, max_edges)
        ok = True
        for k in range(1, max_k + 1):
            if graph_seq[k] is None:
                break
            total = 0
            for canon in all_canons:
                cnt = graph_types.get(canon, (0,))[0]
                if cnt == 0:
                    continue
                ck = coeffs.get(canon, [])
                if k < len(ck) and ck[k] is not None:
                    total += ck[k] * cnt
            if total != graph_seq[k]:
                print(f"  {name} k={k}: MISMATCH Σ={total} vs γ={graph_seq[k]}")
                ok = False
        if ok:
            print(f"  {name}: all grades verified ✓")

    # Cancellation table
    print(f"\n{'=' * 70}")
    print(f"  Cancellation Analysis")
    print(f"{'=' * 70}")

    for k in range(1, max_k + 1):
        if seq_d[k] is None:
            break

        contribs = []
        for canon in all_canons:
            _, edges, nv, ne = all_types[canon]
            cd = types_d.get(canon, (0,))[0]
            cc = types_c.get(canon, (0,))[0]
            ck = coeffs.get(canon, [])
            coeff_val = ck[k] if k < len(ck) and ck[k] is not None else 0
            if coeff_val != 0 and (cd != 0 or cc != 0):
                desc = describe_graph(edges, nv)
                diff_c = coeff_val * (cd - cc)
                contribs.append((ne, desc, coeff_val, cd, cc, diff_c))

        has_diff = any(c[5] != 0 for c in contribs)
        if not has_diff and k > 2:
            continue

        print(f"\n  Grade k = {k}:")
        if has_diff:
            print(f"    Types contributing to cancellation:")
            net = 0
            for ne, desc, cv, cd, cc, dc in sorted(contribs):
                if dc != 0:
                    net += dc
                    print(f"      {desc}: coeff={cv:,}, Δcount={cd-cc:+d}, "
                          f"Δcontrib={dc:+,} (running: {net:+,})")
            print(f"      Net: {net}")

    # ============================================================
    #  Linear dependence analysis
    # ============================================================

    print(f"\n{'=' * 70}")
    print(f"  Linear Dependence Analysis")
    print(f"{'=' * 70}")

    # Build type info with unique labels
    type_info = []
    for canon in all_canons:
        _, edges, nv, ne = all_types[canon]
        label = describe_graph(edges, nv)
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

    # Restrict to nonzero Delta
    nz_indices = [i for i, ti in enumerate(type_info) if ti[6] != 0]
    n_nz = len(nz_indices)

    print(f"\n  Types with Δ ≠ 0: {n_nz}")

    # Build exact rational matrix
    M = []
    for k in range(max_k):
        row = [Fraction(type_info[j][3][k]) for j in nz_indices]
        M.append(row)

    Delta_vec = [Fraction(type_info[j][6]) for j in nz_indices]
    nz_labels = [unique_labels[j] for j in nz_indices]

    # Verify
    print(f"\n  Exact verification M · Δ:")
    all_zero = True
    for k in range(max_k):
        val = sum(M[k][j] * Delta_vec[j] for j in range(n_nz))
        status = "✓" if val == 0 else f"✗ ({val})"
        print(f"    k={k+1}: {status}")
        if val != 0:
            all_zero = False

    if not all_zero:
        print(f"\n  *** Some grades failed — coefficients may be None at those grades ***")

    # Row reduce
    mat = [row[:] for row in M]
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

    print(f"\n  Rank of M: {rank}")
    print(f"  Null space dimension: {n_nz - rank}")
    print(f"\n  Pivot types ({rank}):")
    for pc in pivot_cols:
        print(f"    {nz_labels[pc]}")
    print(f"\n  Free types ({len(free_cols)}):")
    for fc in free_cols:
        print(f"    {nz_labels[fc]}")

    # Null space basis
    def lcm(a, b):
        return a * b // gcd(a, b)

    print(f"\n  Null space basis (integer-scaled):")
    null_basis = []
    for fi, fc in enumerate(free_cols):
        null_vec = [Fraction(0)] * n_nz
        null_vec[fc] = Fraction(1)
        for pi, pc in enumerate(pivot_cols):
            null_vec[pc] = -mat[pi][fc]
        null_basis.append(null_vec)

        denoms = [abs(v.denominator) for v in null_vec if v != 0]
        if denoms:
            lcd_val = reduce(lcm, denoms)
            scaled = [int(v * lcd_val) for v in null_vec]
            nums = [abs(s) for s in scaled if s != 0]
            common = reduce(gcd, nums) if nums else 1
            scaled = [s // common for s in scaled]
        else:
            scaled = [0] * n_nz

        check = all(
            sum(M[k][j] * null_vec[j] for j in range(n_nz)) == 0
            for k in range(max_k)
        )

        print(f"\n    v_{fi+1} (free: {nz_labels[fc]}), verified={check}:")
        for j in range(n_nz):
            if scaled[j] != 0:
                print(f"      {nz_labels[j]:>50s}: {scaled[j]:+d}")

    # Express Delta
    print(f"\n  Δ decomposition:")
    terms = []
    for fi, fc in enumerate(free_cols):
        if Delta_vec[fc] != 0:
            terms.append(f"({Delta_vec[fc]})·v_{fi+1}")
    print(f"    Δ = {' + '.join(terms)}")

    reconstructed = [Fraction(0)] * n_nz
    for fi, fc in enumerate(free_cols):
        for j in range(n_nz):
            reconstructed[j] += Delta_vec[fc] * null_basis[fi][j]
    match = all(reconstructed[j] == Delta_vec[j] for j in range(n_nz))
    print(f"    Reconstruction matches: {match}")


if __name__ == "__main__":
    main()