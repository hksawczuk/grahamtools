#!/usr/bin/env python3
"""
Coefficient recovery via K_{n,m} host graphs.

γ_k(K_{n,m}) = Σ_τ count(τ, K_{n,m}) · coeff_k(τ)

where τ ranges over all connected bipartite graph types with ≤ k edges.
count(τ, K_{n,m}) = (n^↓a · m^↓b + n^↓b · m^↓a) / |Aut(τ)|

Uses nauty (geng + dreadnaut) for graph enumeration and automorphism computation.

Usage: python3 coeff_knm.py [--max-k 7] [--max-nm 9]
"""

import sys
import re
import subprocess
import argparse
from fractions import Fraction
from collections import defaultdict
import networkx as nx


# ============================================================
#  |Aut| via dreadnaut
# ============================================================

def aut_size_nauty(G):
    """Compute |Aut(G)| using nauty's dreadnaut."""
    n = G.number_of_nodes()
    if n <= 1:
        return 1

    adj = defaultdict(list)
    for u, v in G.edges():
        adj[u].append(v)
        adj[v].append(u)

    lines = [f"n={n} g"]
    for v in range(n):
        nbrs = sorted(adj[v])
        if nbrs:
            lines.append(" ".join(str(u) for u in nbrs) + ";")
        else:
            lines.append(";")
    lines.append("x")
    lines.append("q")

    dreadnaut_input = "\n".join(lines) + "\n"

    for cmd in ["dreadnaut", "nauty-dreadnaut"]:
        try:
            result = subprocess.run(
                [cmd], input=dreadnaut_input,
                capture_output=True, text=True, timeout=10
            )
            output = result.stdout + result.stderr
            for line in output.split('\n'):
                if 'grpsize' in line:
                    m2 = re.search(r'grpsize=(\d+)\*10\^(\d+)', line)
                    if m2:
                        return int(m2.group(1)) * (10 ** int(m2.group(2)))
                    m1 = re.search(r'grpsize=(\d+)', line)
                    if m1:
                        return int(m1.group(1))
        except FileNotFoundError:
            continue

    # Fallback: brute force
    from itertools import permutations
    edge_set = set()
    for u, v in G.edges():
        edge_set.add((u, v))
        edge_set.add((v, u))
    count = 0
    for perm in permutations(range(n)):
        if all((perm[u], perm[v]) in edge_set for u, v in G.edges()):
            count += 1
    return count


# ============================================================
#  Enumerate connected bipartite graphs via geng
# ============================================================

def enumerate_bipartite(max_edges):
    """Use geng -cb to enumerate all connected bipartite graphs up to max_edges edges.
    Returns list of (g6, G, nv, ne, bipart, aut, name, is_tree)."""

    all_graphs = []

    for ne in range(1, max_edges + 1):
        count = 0
        for nv in range(2, ne + 2):
            result = None
            for cmd in ["geng", "nauty-geng"]:
                try:
                    result = subprocess.run(
                        [cmd, "-cb", "-q", str(nv), f"{ne}:{ne}"],
                        capture_output=True, text=True, timeout=120
                    )
                    break
                except FileNotFoundError:
                    continue

            if result is None:
                print("ERROR: geng not found")
                sys.exit(1)

            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                G = nx.from_graph6_bytes(line.encode())

                if not nx.is_bipartite(G):
                    continue
                parts = nx.bipartite.sets(G)
                a, b = len(parts[0]), len(parts[1])
                if a > b:
                    a, b = b, a

                aut = aut_size_nauty(G)
                is_tree = (ne == nv - 1)

                # Name
                deg_seq = sorted([G.degree(v) for v in G.nodes()], reverse=True)
                if is_tree:
                    if all(d <= 2 for d in deg_seq):
                        name = f"P{nv}"
                    elif deg_seq[0] == ne:
                        name = f"K1_{ne}"
                    else:
                        ds = "".join(str(d) for d in deg_seq)
                        name = f"T{nv}({ds})"
                elif ne == nv and all(d == 2 for d in deg_seq):
                    name = f"C{nv}"
                elif ne == a * b:
                    name = f"K{a},{b}"
                else:
                    ds = "".join(str(d) for d in deg_seq)
                    name = f"B{nv}({ds})"

                # Deduplicate names within same edge count
                existing = [g for g in all_graphs if g[6] == name and g[3] == ne]
                if existing:
                    name = f"{name}_{chr(ord('a') + len(existing))}"

                all_graphs.append((line, G, nv, ne, (a, b), aut, name, is_tree))
                count += 1

        n_trees = sum(1 for g in all_graphs if g[3] == ne and g[7])
        print(f"  {ne}-edge: {count} types ({n_trees} trees, {count - n_trees} non-trees)")

    return all_graphs


# ============================================================
#  Analytical gamma for K_{n,m}
# ============================================================

def gamma_Knm(n, m, max_k):
    """γ_k(K_{n,m}). L(K_{n,m}) is (n+m-2)-regular on nm vertices."""
    if n == 0 or m == 0:
        return [n + m] + [0] * max_k
    seq = [n + m]
    v = n * m
    d = n + m - 2
    for k in range(1, max_k + 1):
        seq.append(v)
        if v == 0:
            while len(seq) <= max_k:
                seq.append(0)
            break
        e = v * d // 2
        v = e
        d = 2 * d - 2
    return seq


def falling(n, k):
    """n^{↓k} = n!/(n-k)!"""
    if k < 0 or k > n:
        return 0
    r = 1
    for i in range(k):
        r *= (n - i)
    return r


def count_in_Knm(bipart, aut, n, m):
    """count(τ, K_{n,m}) = (n^↓a · m^↓b + n^↓b · m^↓a) / |Aut(τ)|"""
    a, b = bipart
    total = falling(n, a) * falling(m, b) + falling(n, b) * falling(m, a)
    return Fraction(total, aut)


# ============================================================
#  Linear system solver
# ============================================================

def solve_system(M, b, n_vars):
    """Gaussian elimination with exact Fraction arithmetic."""
    n_eqs = len(b)
    aug = [[M[i][j] for j in range(n_vars)] + [b[i]] for i in range(n_eqs)]

    pivot_row = 0
    pivot_cols = []
    for col in range(n_vars):
        piv = None
        for r in range(pivot_row, n_eqs):
            if aug[r][col] != 0:
                piv = r
                break
        if piv is None:
            continue
        aug[pivot_row], aug[piv] = aug[piv], aug[pivot_row]
        pivot_cols.append(col)
        scale = aug[pivot_row][col]
        for j in range(n_vars + 1):
            aug[pivot_row][j] /= scale
        for r in range(n_eqs):
            if r != pivot_row and aug[r][col] != 0:
                f = aug[r][col]
                for j in range(n_vars + 1):
                    aug[r][j] -= f * aug[pivot_row][j]
        pivot_row += 1

    rank = len(pivot_cols)
    x = [Fraction(0)] * n_vars
    for i, col in enumerate(pivot_cols):
        x[col] = aug[i][n_vars]

    return x, rank, pivot_cols


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-k", type=int, default=7)
    parser.add_argument("--max-nm", type=int, default=12,
                        help="Max value of n,m for K_{n,m} host graphs")
    args = parser.parse_args()
    max_k = args.max_k
    max_nm = args.max_nm

    print("=" * 70)
    print("  Enumerating connected bipartite graphs")
    print("=" * 70)
    all_graphs = enumerate_bipartite(max_k)
    all_graphs.sort(key=lambda g: (g[3], g[2], g[6]))
    print(f"\n  Total: {len(all_graphs)} types")

    # Show signature collisions
    sig_groups = defaultdict(list)
    for g in all_graphs:
        sig = (g[4], g[5])
        sig_groups[sig].append(g)

    print(f"\n  Signature collisions (bipart, |Aut|):")
    for sig, gs in sorted(sig_groups.items()):
        if len(gs) > 1:
            names = [g[6] for g in gs]
            print(f"    {sig}: {names}")

    # Host graphs
    host_params = []
    for n in range(1, max_nm + 1):
        for m in range(n, max_nm + 1):
            host_params.append((n, m))
    print(f"\n  Host graphs: {len(host_params)} K_{{n,m}} pairs (max n,m={max_nm})")

    # Solve grade by grade
    print(f"\n{'='*70}")
    print(f"  Solving for coefficients grade by grade")
    print(f"{'='*70}")

    all_coeffs = {}

    for k in range(1, max_k + 1):
        active = [g for g in all_graphs if g[3] <= k]
        n_active = len(active)
        active_names = [g[6] for g in active]

        print(f"\n  Grade k={k}: {n_active} unknowns")

        # Build equations
        equations = []
        for n, m in host_params:
            gamma = gamma_Knm(n, m, k)
            if k >= len(gamma):
                continue
            gamma_k = Fraction(gamma[k])

            counts = [count_in_Knm(g[4], g[5], n, m) for g in active]

            if all(c == 0 for c in counts):
                continue
            equations.append((f"K_{{{n},{m}}}", counts, gamma_k))

        M_mat = [eq[1] for eq in equations]
        b_vec = [eq[2] for eq in equations]

        x, rank, pivots = solve_system(M_mat, b_vec, n_active)

        free = [j for j in range(n_active) if j not in pivots]

        print(f"    {len(equations)} equations, rank {rank}/{n_active}")
        if free:
            free_names = [active_names[j] for j in free]
            print(f"    Free variables ({len(free)}): {free_names}")

        # Store
        for j, g in enumerate(active):
            all_coeffs[(k, g[6])] = x[j]

        # Show nonzero tree coefficients
        tree_coeffs = [(g[6], x[j]) for j, g in enumerate(active)
                       if g[7] and x[j] != 0]
        if tree_coeffs:
            tc_str = ", ".join(f"{n}={v}" for n, v in tree_coeffs)
            print(f"    Tree coefficients: {tc_str}")

    # ── Tree coefficient matrix ──
    trees = [g for g in all_graphs if g[7]]
    print(f"\n{'='*70}")
    print(f"  Tree coefficient submatrix ({len(trees)} trees)")
    print(f"{'='*70}")

    tree_names = [g[6] for g in trees]
    col_w = max(max(len(n) for n in tree_names), 8) + 2
    header = "  grade" + "".join(f"{n:>{col_w}}" for n in tree_names)
    print(header)

    for k in range(1, max_k + 1):
        vals = []
        for g in trees:
            v = all_coeffs.get((k, g[6]), Fraction(0))
            if v.denominator == 1:
                vals.append(f"{int(v):>{col_w}}")
            else:
                vals.append(f"{str(v):>{col_w}}")
        print(f"  k={k:<3d}" + "".join(vals))

    # ── All coefficients ──
    print(f"\n{'='*70}")
    print(f"  All bipartite type coefficients")
    print(f"{'='*70}")

    for ne in range(1, max_k + 1):
        types_ne = [g for g in all_graphs if g[3] == ne]
        if not types_ne:
            continue
        print(f"\n  {ne}-edge types:")
        for g in types_ne:
            tree_tag = " (tree)" if g[7] else ""
            sig = f"bipart={g[4]}, |Aut|={g[5]}"
            coeff_str = []
            for k in range(ne, max_k + 1):
                v = all_coeffs.get((k, g[6]), Fraction(0))
                if v.denominator == 1:
                    coeff_str.append(str(int(v)))
                else:
                    coeff_str.append(str(v))
            print(f"    {g[6]:>16s}{tree_tag:>8s}  [{sig}]  k={ne}..{max_k}: {', '.join(coeff_str)}")

    # ── Comparison with Möbius values ──
    print(f"\n{'='*70}")
    print(f"  Comparison with Möbius inversion values")
    print(f"{'='*70}")

    known = {
        (1, "K2"): 1, (1, "P2"): 1,
        (2, "P3"): 1,
        (3, "K1_3"): 3, (3, "P4"): 1,
        (4, "K1_3"): 3, (4, "K1_4"): 24, (4, "P5"): 1, (4, "fork"): 5,
        (5, "K1_3"): 3, (5, "K1_4"): 168, (5, "P5"): 0, (5, "fork"): 15,
        (5, "K1_5"): 480, (5, "P6"): 1,
        (6, "K1_3"): 3, (6, "K1_4"): 1608, (6, "fork"): 61,
        (6, "K1_5"): 14880, (6, "K1_6"): 23040,
        (7, "K1_3"): 3, (7, "K1_4"): 27528, (7, "fork"): 393,
        (7, "K1_5"): 619680, (7, "K1_6"): 2557440,
    }

    mismatches = 0
    matches = 0
    for (k, name), expected in sorted(known.items()):
        got = all_coeffs.get((k, name), None)
        if got is None:
            continue
        if got != Fraction(expected):
            print(f"  MISMATCH: coeff_{k}({name}) = {got}, expected {expected}")
            mismatches += 1
        else:
            matches += 1

    if mismatches == 0:
        print(f"  All {matches} checked values match ✓")
    else:
        print(f"  {matches} match, {mismatches} mismatch")


if __name__ == "__main__":
    main()