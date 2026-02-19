#!/usr/bin/env python3
"""
Fiber coefficient computation and analysis for iterated line graphs.

Subcommands
-----------
kn (default)
    Compute fiber coefficients from L^k(K_n) using the grahamtools
    level-generation / canonical-classification machinery.

knm
    Recover coefficients analytically via K_{n,m} host graphs and
    nauty-based enumeration of connected bipartite graph types.

solve
    Enumerate all connected graph types up to a given edge count, then
    solve the linear system  gamma_k(K_n) = sum_tau coeff_k(tau) sub(tau, K_n)
    using exact Gaussian elimination.

matrix
    Hardcoded coefficient-matrix analysis for the 6 tree types on 6
    vertices in K_6 (bitmask-based fiber counting through iterated
    line graphs).

Usage examples
--------------
  python3 fiber_coefficients.py kn --n 5 --max-k 7
  python3 fiber_coefficients.py knm --max-k 7 --max-nm 12
  python3 fiber_coefficients.py solve 5 16
  python3 fiber_coefficients.py matrix
"""

import argparse
import math
import sys
import time
from collections import defaultdict
from fractions import Fraction
from functools import reduce
from itertools import combinations
from math import gcd

import networkx as nx

# ---------------------------------------------------------------------------
# grahamtools imports — replaces all locally reimplemented helpers
# ---------------------------------------------------------------------------
from grahamtools.kn import generate_levels_Kn_ids, expand_to_simple_base_edges_id
from grahamtools.kn.classify import canon_key
from grahamtools.utils.automorphisms import aut_size_edges, orbit_size_under_Sn
from grahamtools.utils.naming import tree_name, describe_graph
from grahamtools.utils.linalg import exact_rank, row_reduce_fraction
from grahamtools.utils.linegraph_edgelist import gamma_sequence_edgelist
from grahamtools.utils.subgraphs import enumerate_connected_subgraphs
from grahamtools.utils.connectivity import is_connected_edges
from grahamtools.external.nauty import geng_g6, aut_size_g6, nauty_available


# ===================================================================
#  Subcommand: kn  (fiber coefficients from L^k(K_n))
# ===================================================================

def cmd_kn(args: argparse.Namespace) -> None:
    """Compute fiber coefficients by classifying vertices of L^k(K_n)."""
    n = args.n
    max_k = args.max_k

    # ---- generate levels ----
    print(f"Generating L^k(K_{n}) for k=0..{max_k} (trees only)...")
    t0 = time.time()
    V_by_level, ep = generate_levels_Kn_ids(n, max_k, prune_cycles=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    for k in range(max_k + 1):
        print(f"  Grade {k}: {len(V_by_level.get(k, []))} tree vertices")

    # ---- classify fibers per grade ----
    all_fibers: dict[object, dict] = {}

    for k in range(1, max_k + 1):
        Vk = V_by_level.get(k, [])
        if not Vk:
            continue

        t0 = time.time()

        buckets: dict[object, dict] = {}
        for v in Vk:
            edges = expand_to_simple_base_edges_id(v, k, ep)
            key = canon_key(edges, n)
            if key not in buckets:
                buckets[key] = {"edges": edges, "freq": 0}
            buckets[key]["freq"] += 1

        elapsed = time.time() - t0

        for key, bucket in buckets.items():
            edges = bucket["edges"]
            freq = bucket["freq"]
            aut = aut_size_edges(edges, n)
            orbit = orbit_size_under_Sn(edges, n)
            coeff = freq // orbit

            if key not in all_fibers:
                name = tree_name(edges, n)
                all_fibers[key] = {
                    "name": name,
                    "edges": edges,
                    "aut": aut,
                    "orbit_sz": orbit,
                    "count_in_Kn": orbit,
                    "nedges": len(edges),
                    "coeff": {},
                    "fiber_sz": {},
                    "n_orbits": {},
                }

            all_fibers[key]["coeff"][k] = coeff
            all_fibers[key]["fiber_sz"][k] = freq
            all_fibers[key]["n_orbits"][k] = coeff

        if elapsed > 0.5:
            print(
                f"    Grade {k}: classified {len(Vk)} vertices into "
                f"{len(buckets)} fibers in {elapsed:.1f}s"
            )

    _print_kn_results(all_fibers, n, max_k)


def _print_kn_results(all_fibers: dict, n: int, max_k: int) -> None:
    """Print detailed results for the kn subcommand."""
    sorted_fibers = sorted(
        all_fibers.values(),
        key=lambda f: (f["nedges"], -max(f["fiber_sz"].values(), default=0)),
    )

    # ---- per-fiber detail ----
    print(f"\n{'=' * 80}")
    print(f"  Fiber coefficients for L^k(K_{n})")
    print(f"{'=' * 80}")

    for fiber in sorted_fibers:
        name = fiber["name"]
        ne = fiber["nedges"]
        aut = fiber["aut"]
        orbit = fiber["orbit_sz"]

        print(
            f"\n  {name}  ({ne} edges, |Aut|={aut}, orbit_sz={orbit}, "
            f"count_in_K{n}={orbit})"
        )
        print(
            f"    {'k':>4s} {'coeff_k':>12s} {'fiber_sz':>12s} "
            f"{'#orbits':>10s} {'ratio':>10s}"
        )

        prev_coeff = None
        for k in sorted(fiber["coeff"]):
            c = fiber["coeff"][k]
            fs = fiber["fiber_sz"][k]
            no = fiber["n_orbits"][k]
            ratio = c / prev_coeff if prev_coeff and prev_coeff > 0 else None
            print(
                f"    {k:>4d} {c:>12d} {fs:>12d} {no:>10d} "
                f"{f'{ratio:.2f}' if ratio else '---':>10s}"
            )
            prev_coeff = c

    # ---- summary table ----
    print(f"\n{'=' * 80}")
    print(f"  Summary: coeff_k(tau) for all fibers")
    print(f"{'=' * 80}")

    all_grades: set[int] = set()
    for f in sorted_fibers:
        all_grades.update(f["coeff"].keys())
    grades = sorted(all_grades)

    hdr = f"  {'Tree':>16s} {'|e|':>4s} {'|Aut|':>5s}"
    for k in grades:
        hdr += f" {'k=' + str(k):>10s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for fiber in sorted_fibers:
        row = f"  {fiber['name']:>16s} {fiber['nedges']:>4d} {fiber['aut']:>5d}"
        for k in grades:
            c = fiber["coeff"].get(k)
            row += f" {c:>10d}" if c is not None else f" {'':>10s}"
        print(row)

    # ---- growth rates ----
    print(f"\n{'=' * 80}")
    print(f"  Growth rates: coeff_k / coeff_{{k-1}}")
    print(f"{'=' * 80}")

    hdr = f"  {'Tree':>16s}"
    for k in grades[1:]:
        hdr += f" {'k=' + str(k):>10s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for fiber in sorted_fibers:
        row = f"  {fiber['name']:>16s}"
        for k in grades[1:]:
            c = fiber["coeff"].get(k)
            cp = fiber["coeff"].get(k - 1)
            if c is not None and cp is not None and cp > 0:
                row += f" {c / cp:>10.2f}"
            else:
                row += f" {'':>10s}"
        print(row)

    # ---- fiber fractions ----
    print(f"\n{'=' * 80}")
    print(f"  Fiber fractions: fiber_sz / |V(L^k)|")
    print(f"{'=' * 80}")

    hdr = f"  {'Tree':>16s}"
    for k in grades:
        hdr += f" {'k=' + str(k):>10s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    totals = {}
    for k in grades:
        totals[k] = sum(f["fiber_sz"].get(k, 0) for f in sorted_fibers)

    row = f"  {'TOTAL':>16s}"
    for k in grades:
        row += f" {totals[k]:>10d}"
    print(row)
    print("  " + "-" * (len(hdr) - 2))

    for fiber in sorted_fibers:
        row = f"  {fiber['name']:>16s}"
        for k in grades:
            fs = fiber["fiber_sz"].get(k, 0)
            if totals[k] > 0:
                row += f" {fs / totals[k]:>10.4f}"
            else:
                row += f" {'':>10s}"
        print(row)


# ===================================================================
#  Subcommand: knm  (K_{n,m} host graph approach)
# ===================================================================

def _enumerate_bipartite(max_edges: int) -> list[tuple]:
    """Enumerate connected bipartite graphs via nauty's geng -cb.

    Returns list of (g6, G, nv, ne, bipart, aut, name, is_tree).
    """
    all_graphs: list[tuple] = []

    for ne in range(1, max_edges + 1):
        count = 0
        for nv in range(2, ne + 2):
            for g6 in geng_g6(
                nv,
                connected=True,
                bipartite=True,
                min_edges=ne,
                max_edges=ne,
            ):
                G = nx.from_graph6_bytes(g6.encode())
                if not nx.is_bipartite(G):
                    continue
                parts = nx.bipartite.sets(G)
                a, b = len(parts[0]), len(parts[1])
                if a > b:
                    a, b = b, a

                aut = aut_size_g6(g6)
                is_tree = ne == nv - 1

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

                all_graphs.append((g6, G, nv, ne, (a, b), aut, name, is_tree))
                count += 1

        n_trees = sum(1 for g in all_graphs if g[3] == ne and g[7])
        print(f"  {ne}-edge: {count} types ({n_trees} trees, {count - n_trees} non-trees)")

    return all_graphs


def _gamma_Knm(n: int, m: int, max_k: int) -> list[int]:
    """Analytical gamma_k(K_{n,m})."""
    if n == 0 or m == 0:
        return [n + m] + [0] * max_k
    seq = [n + m]
    v = n * m
    d = n + m - 2
    for _ in range(1, max_k + 1):
        seq.append(v)
        if v == 0:
            while len(seq) <= max_k:
                seq.append(0)
            break
        e = v * d // 2
        v = e
        d = 2 * d - 2
    return seq


def _falling(n: int, k: int) -> int:
    """Falling factorial n^{down-k}."""
    if k < 0 or k > n:
        return 0
    r = 1
    for i in range(k):
        r *= n - i
    return r


def _count_in_Knm(
    bipart: tuple[int, int], aut: int, n: int, m: int
) -> Fraction:
    """count(tau, K_{n,m}) for bipartite tau with partition sizes (a, b)."""
    a, b = bipart
    total = _falling(n, a) * _falling(m, b) + _falling(n, b) * _falling(m, a)
    return Fraction(total, aut)


def _solve_system(
    M: list[list[Fraction]], b: list[Fraction], n_vars: int
) -> tuple[list[Fraction], int, list[int]]:
    """Gaussian elimination with exact Fraction arithmetic."""
    n_eqs = len(b)
    aug = [M[i][:] + [b[i]] for i in range(n_eqs)]

    pivot_row = 0
    pivot_cols: list[int] = []
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


def cmd_knm(args: argparse.Namespace) -> None:
    """Recover coefficients via K_{n,m} host graphs."""
    max_k = args.max_k
    max_nm = args.max_nm

    if not nauty_available():
        print("ERROR: nauty (geng, shortg) required for 'knm' subcommand")
        sys.exit(1)

    print("=" * 70)
    print("  Enumerating connected bipartite graphs")
    print("=" * 70)
    all_graphs = _enumerate_bipartite(max_k)
    all_graphs.sort(key=lambda g: (g[3], g[2], g[6]))
    print(f"\n  Total: {len(all_graphs)} types")

    # Signature collisions
    sig_groups: dict[tuple, list] = defaultdict(list)
    for g in all_graphs:
        sig_groups[(g[4], g[5])].append(g)

    print(f"\n  Signature collisions (bipart, |Aut|):")
    for sig, gs in sorted(sig_groups.items()):
        if len(gs) > 1:
            names = [g[6] for g in gs]
            print(f"    {sig}: {names}")

    # Host graphs
    host_params = [
        (n, m) for n in range(1, max_nm + 1) for m in range(n, max_nm + 1)
    ]
    print(f"\n  Host graphs: {len(host_params)} K_{{n,m}} pairs (max n,m={max_nm})")

    # Solve grade by grade
    print(f"\n{'=' * 70}")
    print(f"  Solving for coefficients grade by grade")
    print(f"{'=' * 70}")

    all_coeffs: dict[tuple[int, str], Fraction] = {}

    for k in range(1, max_k + 1):
        active = [g for g in all_graphs if g[3] <= k]
        n_active = len(active)
        active_names = [g[6] for g in active]

        print(f"\n  Grade k={k}: {n_active} unknowns")

        equations: list[tuple[str, list[Fraction], Fraction]] = []
        for n, m in host_params:
            gamma = _gamma_Knm(n, m, k)
            if k >= len(gamma):
                continue
            gamma_k = Fraction(gamma[k])
            counts = [_count_in_Knm(g[4], g[5], n, m) for g in active]
            if all(c == 0 for c in counts):
                continue
            equations.append((f"K_{{{n},{m}}}", counts, gamma_k))

        M_mat = [eq[1] for eq in equations]
        b_vec = [eq[2] for eq in equations]
        x, rank, pivots = _solve_system(M_mat, b_vec, n_active)

        free = [j for j in range(n_active) if j not in pivots]
        print(f"    {len(equations)} equations, rank {rank}/{n_active}")
        if free:
            free_names = [active_names[j] for j in free]
            print(f"    Free variables ({len(free)}): {free_names}")

        for j, g in enumerate(active):
            all_coeffs[(k, g[6])] = x[j]

        tree_coeffs = [
            (g[6], x[j])
            for j, g in enumerate(active)
            if g[7] and x[j] != 0
        ]
        if tree_coeffs:
            tc_str = ", ".join(f"{nm}={v}" for nm, v in tree_coeffs)
            print(f"    Tree coefficients: {tc_str}")

    # Tree coefficient matrix
    trees = [g for g in all_graphs if g[7]]
    print(f"\n{'=' * 70}")
    print(f"  Tree coefficient submatrix ({len(trees)} trees)")
    print(f"{'=' * 70}")

    tree_names_list = [g[6] for g in trees]
    col_w = max(max(len(nm) for nm in tree_names_list), 8) + 2
    header = "  grade" + "".join(f"{nm:>{col_w}}" for nm in tree_names_list)
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

    # All coefficients
    print(f"\n{'=' * 70}")
    print(f"  All bipartite type coefficients")
    print(f"{'=' * 70}")

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
            print(
                f"    {g[6]:>16s}{tree_tag:>8s}  [{sig}]  "
                f"k={ne}..{max_k}: {', '.join(coeff_str)}"
            )

    # Comparison with Moebius values
    print(f"\n{'=' * 70}")
    print(f"  Comparison with Moebius inversion values")
    print(f"{'=' * 70}")

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

    mismatches = matches = 0
    for (k, name), expected in sorted(known.items()):
        got = all_coeffs.get((k, name))
        if got is None:
            continue
        if got != Fraction(expected):
            print(f"  MISMATCH: coeff_{k}({name}) = {got}, expected {expected}")
            mismatches += 1
        else:
            matches += 1

    if mismatches == 0:
        print(f"  All {matches} checked values match")
    else:
        print(f"  {matches} match, {mismatches} mismatch")


# ===================================================================
#  Subcommand: solve  (coefficient system via Gaussian elimination)
# ===================================================================

def _enumerate_connected_graphs_nx(max_edges: int) -> list[tuple]:
    """Enumerate all connected graphs with 1..max_edges edges.

    Returns list of (G, n_vertices, n_edges, aut_size).
    """
    all_types: list[tuple] = []

    for e in range(1, max_edges + 1):
        types_this_e: list[tuple] = []
        for v in range(2, e + 2):
            if e > v * (v - 1) // 2 or e < v - 1:
                continue
            all_possible = list(combinations(range(v), 2))
            for edge_combo in combinations(all_possible, e):
                G = nx.Graph()
                G.add_nodes_from(range(v))
                G.add_edges_from(edge_combo)
                if not nx.is_connected(G):
                    continue
                is_new = True
                for G2, v2, e2, _ in types_this_e:
                    if v2 == v and nx.is_isomorphic(G, G2):
                        is_new = False
                        break
                if is_new:
                    aut = _count_automorphisms_nx(G)
                    types_this_e.append((G, v, e, aut))
        all_types.extend(types_this_e)
        print(
            f"  e={e}: {len(types_this_e)} types "
            f"(cumulative: {len(all_types)})",
            flush=True,
        )
    return all_types


def _count_automorphisms_nx(G: nx.Graph) -> int:
    """Brute-force automorphism count for a networkx graph."""
    from itertools import permutations as _perms

    nodes = sorted(G.nodes())
    edge_set = set(frozenset(e) for e in G.edges())
    count = 0
    for perm in _perms(nodes):
        mapping = dict(zip(nodes, perm))
        if set(frozenset((mapping[u], mapping[v])) for u, v in G.edges()) == edge_set:
            count += 1
    return count


def _classify_type_nx(G: nx.Graph) -> str:
    """Classify a small graph by degree sequence / structure."""
    v = G.number_of_nodes()
    e = G.number_of_edges()
    deg_seq = tuple(sorted(dict(G.degree()).values()))

    if e == v - 1:  # tree
        if e == 1:
            return "K2"
        if e == 2:
            return "P3"
        if e == 3:
            if max(deg_seq) == 3:
                return "K1_3"
            return "P4"
        if e == 4:
            if max(deg_seq) == 4:
                return "K1_4"
            if max(deg_seq) == 3:
                return "fork"
            return "P5"
        if e == 5:
            if max(deg_seq) == 5:
                return "K1_5"
            if deg_seq == (1, 1, 2, 2, 2, 2):
                return "P6"
            if deg_seq == (1, 1, 1, 1, 3, 3):
                return "dblstar"
            if deg_seq == (1, 1, 1, 1, 2, 4):
                return "spider"
            if deg_seq == (1, 1, 1, 2, 2, 3):
                v3 = [nd for nd in G.nodes() if G.degree(nd) == 3][0]
                nbr_degs = sorted(G.degree(u) for u in G.neighbors(v3))
                return "catA" if nbr_degs == [1, 2, 2] else "catB"
        return f"tree_{v}v_{deg_seq}"
    return f"g{v}v{e}e_{deg_seq}"


def _gamma_k_Kn(k: int, n: int) -> int:
    """gamma_k(K_n) = |V(L^k(K_n))| computed analytically."""
    v = n * (n - 1) // 2
    d = 2 * (n - 2)
    for _ in range(2, k + 1):
        v = v * d // 2
        d = 2 * d - 2
    return v * d // 2


def _sub_tau_Kn(v_tau: int, aut_size: int, n: int) -> int:
    """sub(tau, K_n) = n^{down v_tau} / |Aut(tau)|."""
    if n < v_tau:
        return 0
    ff = 1
    for i in range(v_tau):
        ff *= n - i
    return ff // aut_size


def _solve_grade(
    k: int,
    active_indices: list[int],
    all_types: list[tuple],
    n_values: list[int],
) -> tuple[dict[int, int] | None, object]:
    """Solve for coeff_k(tau) for active types."""
    n_unknowns = len(active_indices)
    n_eqs = len(n_values)

    S = []
    gamma = []
    for n_val in n_values:
        row = []
        for j in active_indices:
            _, v, e, aut = all_types[j]
            row.append(Fraction(_sub_tau_Kn(v, aut, n_val)))
        S.append(row)
        gamma.append(Fraction(_gamma_k_Kn(k, n_val)))

    aug = [S[i][:] + [gamma[i]] for i in range(n_eqs)]

    # Gaussian elimination
    for col in range(n_unknowns):
        pivot = None
        for row in range(col, min(n_unknowns, n_eqs)):
            if aug[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            return None, f"SINGULAR at column {col} ({active_indices[col]})"
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        for row in range(n_eqs):
            if row != col and aug[row][col] != 0:
                factor = aug[row][col] / aug[col][col]
                for j in range(n_unknowns + 1):
                    aug[row][j] -= factor * aug[col][j]

    solution: dict[int, int] = {}
    for ci, j in enumerate(active_indices):
        val = aug[ci][n_unknowns] / aug[ci][ci]
        if val.denominator != 1:
            return None, f"Non-integer coeff for type {j}: {val}"
        solution[j] = int(val)

    max_residual = Fraction(0)
    for i in range(n_unknowns, n_eqs):
        max_residual = max(max_residual, abs(aug[i][n_unknowns]))

    return solution, max_residual


def cmd_solve(args: argparse.Namespace) -> None:
    """Solve coefficient system using enumerated graph types."""
    max_edges = args.max_edges
    max_grade = args.max_grade

    print(f"Parameters: max_edges={max_edges}, max_grade={max_grade}\n")

    print("Enumerating connected graph types...")
    t0 = time.time()
    all_types = _enumerate_connected_graphs_nx(max_edges)
    print(f"Total types: {len(all_types)} ({time.time() - t0:.1f}s)\n")

    type_names = [_classify_type_nx(G) for G, _, _, _ in all_types]
    type_is_tree = [e == v - 1 for _, v, e, _ in all_types]
    type_edges = [e for _, _, e, _ in all_types]

    for e in range(1, max_edges + 1):
        types_e = [
            (i, type_names[i], all_types[i][1], all_types[i][3])
            for i in range(len(all_types))
            if type_edges[i] == e
        ]
        trees_e = [x for x in types_e if type_is_tree[x[0]]]
        print(f"  e={e}: {len(types_e)} types ({len(trees_e)} trees)")
        for idx, name, v, aut in types_e:
            tree_mark = " [TREE]" if type_is_tree[idx] else ""
            print(f"    {name}: {v}v, |Aut|={aut}{tree_mark}")

    v_max = max(v for _, v, _, _ in all_types)
    all_coeffs: dict[int, dict[int, int]] = {}

    print(f"\nComputing coefficients grade by grade...\n")

    for k in range(1, max_grade + 1):
        t0 = time.time()
        active = [i for i in range(len(all_types)) if type_edges[i] <= k]
        n_active = len(active)
        n_extra = 5
        n_values = list(range(v_max, v_max + n_active + n_extra))

        solution, info = _solve_grade(k, active, all_types, n_values)
        elapsed = time.time() - t0

        if solution is None:
            print(f"  Grade {k}: FAILED -- {info} ({elapsed:.2f}s)")
            continue

        all_coeffs[k] = solution
        nonzero = sum(1 for v in solution.values() if v != 0)
        tree_coeffs = [
            (type_names[j], solution[j])
            for j in active
            if type_is_tree[j] and solution[j] != 0
        ]
        print(
            f"  Grade {k}: {n_active} unknowns, {nonzero} nonzero, "
            f"residual={info} ({elapsed:.2f}s)"
        )
        if tree_coeffs:
            tc_str = ", ".join(f"{nm}={val}" for nm, val in tree_coeffs)
            print(f"    Trees: {tc_str}")

    # Tree independence analysis
    print(f"\n{'=' * 70}")
    print(f"TREE COEFFICIENT INDEPENDENCE ANALYSIS")
    print(f"{'=' * 70}")

    for target_e in range(1, max_edges + 1):
        tree_indices = [
            i
            for i in range(len(all_types))
            if type_is_tree[i] and type_edges[i] == target_e
        ]
        n_trees = len(tree_indices)
        if n_trees <= 1:
            continue

        tree_labels = [type_names[i] for i in tree_indices]
        min_grade = target_e
        grades = list(range(min_grade, max_grade + 1))
        n_rows = len(grades)

        M: list[list[int]] = []
        for k in grades:
            row = [all_coeffs.get(k, {}).get(j, 0) for j in tree_indices]
            M.append(row)

        print(f"\nTrees with {target_e} edges ({n_trees} types): {tree_labels}")
        header = "  grade " + "".join(f"{nm:>14}" for nm in tree_labels)
        print(header)
        for ki, k in enumerate(grades):
            row_str = f"  k={k:2d}  " + "".join(
                f"{M[ki][ti]:>14}" for ti in range(n_trees)
            )
            print(row_str)

        rank = exact_rank(M, n_rows, n_trees)
        status = "FULL RANK" if rank == n_trees else "RANK DEFICIENT"
        print(f"\n  Rank: {rank} / {n_trees} -- {status}")

        print("  Cumulative rank:")
        for nk in range(1, min(n_rows + 1, n_trees + 3)):
            r = exact_rank(M[:nk], nk, n_trees)
            print(f"    Grades {min_grade}..{min_grade + nk - 1}: rank {r}")

    # All trees together
    all_tree_indices = [i for i in range(len(all_types)) if type_is_tree[i]]
    n_all_trees = len(all_tree_indices)
    all_tree_labels = [type_names[i] for i in all_tree_indices]
    min_e = min(type_edges[i] for i in all_tree_indices)
    grades = list(range(min_e, max_grade + 1))
    n_rows = len(grades)

    M = []
    for k in grades:
        row = [all_coeffs.get(k, {}).get(j, 0) for j in all_tree_indices]
        M.append(row)

    print(f"\n{'=' * 70}")
    print(f"ALL TREES ({n_all_trees} types): {all_tree_labels}")
    print(f"{'=' * 70}")
    header = "  grade " + "".join(f"{nm:>14}" for nm in all_tree_labels)
    print(header)
    for ki, k in enumerate(grades):
        row_str = f"  k={k:2d}  " + "".join(
            f"{M[ki][ti]:>14}" for ti in range(n_all_trees)
        )
        print(row_str)

    rank = exact_rank(M, n_rows, n_all_trees)
    status = "FULL RANK" if rank == n_all_trees else "RANK DEFICIENT"
    print(f"\n  Rank: {rank} / {n_all_trees} -- {status}")

    print("  Cumulative rank:")
    for nk in range(1, min(n_rows + 1, n_all_trees + 3)):
        r = exact_rank(M[:nk], nk, n_all_trees)
        print(f"    Grades {min_e}..{min_e + nk - 1}: rank {r}")


# ===================================================================
#  Subcommand: matrix  (hardcoded n=6 bitmask-based analysis)
# ===================================================================

def _edges_of_kn(n: int) -> list[tuple[int, int]]:
    """All edges of K_n as (i, j) with i < j."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _edge_index(i: int, j: int, n: int) -> int:
    """Map edge (i,j) with i<j to its index in K_n."""
    if i > j:
        i, j = j, i
    return i * n - i * (i + 1) // 2 + (j - i - 1)


def _base_to_edge_list(
    base_mask: int, edge_list: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Convert bitmask to list of edges."""
    return [edge_list[i] for i in range(len(edge_list)) if base_mask & (1 << i)]


def _classify_tree_from_mask(
    base_mask: int, edge_list: list[tuple[int, int]], n_vertices: int
) -> int:
    """Classify a tree type from its base edge bitmask.

    Returns type index 0-5 (for the 6 trees on 6 vertices) or -1.
    """
    edges = _base_to_edge_list(base_mask, edge_list)
    ne = len(edges)
    if ne != 5:
        return -1

    verts: set[int] = set()
    for u, v in edges:
        verts.add(u)
        verts.add(v)
    if len(verts) != 6:
        return -1

    adj: dict[int, set[int]] = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    start = next(iter(verts))
    visited = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for u in adj[node]:
            if u not in visited:
                visited.add(u)
                stack.append(u)
    if len(visited) != len(verts):
        return -1

    deg_seq = tuple(sorted(len(adj[v]) for v in verts))

    if deg_seq == (1, 1, 2, 2, 2, 2):
        return 0  # P_6
    if deg_seq == (1, 1, 1, 1, 1, 5):
        return 5  # K_{1,5}
    if deg_seq == (1, 1, 1, 1, 3, 3):
        return 4  # Double star
    if deg_seq == (1, 1, 1, 1, 2, 4):
        return 3  # Spider
    if deg_seq == (1, 1, 1, 2, 2, 3):
        v3 = [v for v in verts if len(adj[v]) == 3][0]
        nbr_degs = sorted(len(adj[u]) for u in adj[v3])
        if nbr_degs == [1, 2, 2]:
            return 1  # Caterpillar A
        if nbr_degs == [1, 1, 2]:
            return 2  # Caterpillar B
    return -1


def _count_labeled_copies_n6(n: int) -> list[int]:
    """Count sub(tau, K_6) for each of the 6 tree types on 6 vertices."""
    edge_list = _edges_of_kn(n)
    ne = len(edge_list)
    counts = [0] * 6
    for combo in combinations(range(ne), 5):
        mask = 0
        for i in combo:
            mask |= 1 << i
        t = _classify_tree_from_mask(mask, edge_list, n)
        if t >= 0:
            counts[t] += 1
    return counts


def _build_line_graph_level(
    n_prev: int,
    adj_prev: list[set[int]],
    base_prev: list[int],
) -> tuple[int, list[set[int]], list[int]]:
    """Build next line graph level using bitmask base tracking."""
    new_edges: list[tuple[int, int]] = []
    for v in range(n_prev):
        for u in adj_prev[v]:
            if u > v:
                new_edges.append((v, u))

    n_new = len(new_edges)
    base_new = [base_prev[v] | base_prev[u] for v, u in new_edges]

    incident: list[list[int]] = [[] for _ in range(n_prev)]
    for idx, (v, u) in enumerate(new_edges):
        incident[v].append(idx)
        incident[u].append(idx)

    adj_new: list[set[int]] = [set() for _ in range(n_new)]
    for v_prev in range(n_prev):
        inc = incident[v_prev]
        for i in range(len(inc)):
            for j in range(i + 1, len(inc)):
                a, b = inc[i], inc[j]
                adj_new[a].add(b)
                adj_new[b].add(a)

    return n_new, adj_new, base_new


def cmd_matrix(args: argparse.Namespace) -> None:
    """Hardcoded coefficient matrix analysis for trees on 6 vertices in K_6."""
    import numpy as np

    n = 6
    edge_list = _edges_of_kn(n)
    ne = len(edge_list)  # 15
    tree_names_list = ["P_6", "Cat_A", "Cat_B", "Spider", "DblStar", "K_{1,5}"]
    n_types = 6

    print(f"=== Coefficient matrix for trees on {n} vertices in K_{n} ===\n")

    print("Counting labeled copies of each tree type in K_6...")
    t0 = time.time()
    sub_counts = _count_labeled_copies_n6(n)
    print(
        f"  sub(tau, K_6): {list(zip(tree_names_list, sub_counts))} "
        f"({time.time() - t0:.1f}s)"
    )
    print(f"  Total tree edge-subsets: {sum(sub_counts)}")

    # Build L^1(K_6)
    print(f"\nBuilding iterated line graphs...")

    adj1: list[set[int]] = [set() for _ in range(ne)]
    base1 = [1 << i for i in range(ne)]
    for i in range(ne):
        for j in range(i + 1, ne):
            ei, ej = edge_list[i], edge_list[j]
            if ei[0] in ej or ei[1] in ej:
                adj1[i].add(j)
                adj1[j].add(i)

    cur_n = ne
    cur_adj = adj1
    cur_base = base1
    print(f"  L^1(K_{n}): {cur_n} vertices")

    max_grade = 10
    coeffs = np.zeros((n_types, max_grade - 4), dtype=np.int64)

    for grade in range(2, max_grade + 1):
        t0 = time.time()
        cur_n, cur_adj, cur_base = _build_line_graph_level(cur_n, cur_adj, cur_base)
        elapsed = time.time() - t0
        print(f"  L^{grade}(K_{n}): {cur_n} vertices ({elapsed:.1f}s)", flush=True)

        if grade >= 5:
            type_counts = [0] * n_types
            for v in range(cur_n):
                t = _classify_tree_from_mask(cur_base[v], edge_list, n)
                if t >= 0:
                    type_counts[t] += 1

            col = grade - 5
            for t in range(n_types):
                if sub_counts[t] > 0:
                    assert type_counts[t] % sub_counts[t] == 0, (
                        f"Non-integer coefficient at grade {grade}, "
                        f"type {tree_names_list[t]}: {type_counts[t]}/{sub_counts[t]}"
                    )
                    coeffs[t][col] = type_counts[t] // sub_counts[t]

            print(f"    Fiber counts: {list(zip(tree_names_list, type_counts))}")
            print(
                f"    Coefficients: "
                f"{list(zip(tree_names_list, coeffs[:, col].tolist()))}"
            )

        mem_mb = sys.getsizeof(cur_adj) / 1e6
        print(f"    (approx adj mem: {mem_mb:.0f} MB)", flush=True)

        if cur_n > 5_000_000:
            print(f"    Too large, stopping at grade {grade}")
            break

    # Print coefficient matrix
    n_cols = coeffs.shape[1]
    print(
        f"\n=== Coefficient Matrix "
        f"(rows=types, cols=grades 5..{4 + n_cols}) ==="
    )
    header = "".ljust(12) + "".join(
        f"k={k}".rjust(12) for k in range(5, 5 + n_cols)
    )
    print(header)
    for t in range(n_types):
        row = tree_names_list[t].ljust(12) + "".join(
            f"{coeffs[t][c]}".rjust(12) for c in range(n_cols)
        )
        print(row)

    # Rank analysis
    M = coeffs.T  # shape (n_grades, n_types)
    print(f"\n=== Rank analysis ===")
    print(f"Matrix shape: {M.shape} (grades x types)")

    Mf = M.astype(np.float64)
    rank = np.linalg.matrix_rank(Mf)
    print(f"Rank: {rank} (out of {n_types} types)")

    if rank == n_types:
        print("FULL RANK -- coefficient vectors are linearly independent!")
    else:
        print(f"RANK DEFICIENT -- rank {rank} < {n_types}")

    sv = np.linalg.svd(Mf, compute_uv=False)
    print(f"Singular values: {sv}")
    if sv[-1] > 0:
        print(f"Condition number: {sv[0] / sv[-1]:.2e}")
    else:
        print("Infinite condition number")

    print(f"\nCumulative rank by grade:")
    for k in range(1, n_cols + 1):
        r = np.linalg.matrix_rank(M[:k, :].astype(np.float64))
        print(f"  Grades 5..{4 + k}: rank {r}")


# ===================================================================
#  CLI entry point
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fiber coefficient computation and analysis for iterated line graphs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- kn --
    p_kn = subparsers.add_parser(
        "kn",
        help="Compute fiber coefficients from L^k(K_n)",
    )
    p_kn.add_argument("--n", type=int, default=5, help="Complete graph K_n (default: 5)")
    p_kn.add_argument("--max-k", type=int, default=7, help="Maximum grade (default: 7)")

    # -- knm --
    p_knm = subparsers.add_parser(
        "knm",
        help="K_{n,m} bipartite host graph analysis",
    )
    p_knm.add_argument("--max-k", type=int, default=7, help="Maximum grade (default: 7)")
    p_knm.add_argument(
        "--max-nm", type=int, default=12,
        help="Max value of n, m for K_{n,m} host graphs (default: 12)",
    )

    # -- solve --
    p_solve = subparsers.add_parser(
        "solve",
        help="Solve coefficient system via enumerated graph types",
    )
    p_solve.add_argument("max_edges", type=int, nargs="?", default=5, help="Max edges (default: 5)")
    p_solve.add_argument("max_grade", type=int, nargs="?", default=16, help="Max grade (default: 16)")

    # -- matrix --
    subparsers.add_parser(
        "matrix",
        help="Hardcoded coefficient matrix analysis for n=6 trees",
    )

    args = parser.parse_args()

    if args.command is None or args.command == "kn":
        if args.command is None:
            # Set defaults for the kn command when invoked with no subcommand
            if not hasattr(args, "n"):
                args.n = 5
            if not hasattr(args, "max_k"):
                args.max_k = 7
        cmd_kn(args)
    elif args.command == "knm":
        cmd_knm(args)
    elif args.command == "solve":
        cmd_solve(args)
    elif args.command == "matrix":
        cmd_matrix(args)


if __name__ == "__main__":
    main()
