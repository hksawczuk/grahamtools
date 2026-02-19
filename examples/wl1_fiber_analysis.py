#!/usr/bin/env python3
"""
Fiber decomposition and linear dependence analysis for WL-1 equivalent graphs:
  - Dumbbell: two triangles connected by a bridge (6 vertices, 7 edges)
  - Chorded C6: 6-cycle with antipodal chord (6 vertices, 7 edges)

These graphs are WL-1 equivalent, which implies identical Graham sequences.
We decompose gamma_k(G) = sum_tau coeff_k(tau) * count(tau, G) and analyse how
differing subgraph counts produce identical weighted sums ("fiber analysis"),
then compute the null space of the coefficient matrix to study which linear
dependencies among coefficient vectors allow the cancellation ("linear
dependence analysis").

Usage:
    python3 wl1_fiber_analysis.py [--max-k K]
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from fractions import Fraction
from functools import reduce
from math import gcd

from grahamtools.utils.connectivity import is_connected_edges
from grahamtools.utils.linegraph_edgelist import (
    line_graph_edgelist,
    gamma_sequence_edgelist,
)
from grahamtools.utils.canonical import (
    canonical_graph_nauty,
    canonical_graph_bruteforce,
)
from grahamtools.utils.subgraphs import enumerate_connected_subgraphs
from grahamtools.utils.naming import describe_graph
from grahamtools.utils.linalg import row_reduce_fraction
from grahamtools.external.nauty import nauty_available


# ============================================================
#  Coefficient extraction via Mobius inversion
# ============================================================

def compute_all_coefficients(
    all_types: dict[object, tuple[int, list[tuple[int, int]], int, int]],
    max_k: int,
    *,
    max_edges: int = 5_000_000,
) -> tuple[
    dict[object, list[int | None]],
    dict[object, list[int | None]],
    dict[object, dict[object, int]],
]:
    """Compute coeff_k(tau) for all types by bootstrapping.

    all_types: dict of canon -> (count, edges, n_verts, n_edges)

    Process types in order of increasing edge count.
    For each tau:
      1. Compute gamma_k(tau) by line graph iteration
      2. Enumerate connected subgraphs of tau
      3. coeff_k(tau) = gamma_k(tau) - sum_{sigma subset tau} coeff_k(sigma) * count(sigma, tau)

    Returns:
      gammas:  dict canon -> [gamma_0, ..., gamma_max_k]
      coeffs:  dict canon -> [c_0, ..., c_max_k]
      subtypes: dict canon -> dict of sub_canon -> count
    """
    sorted_types = sorted(all_types.items(), key=lambda x: x[1][3])

    gammas: dict[object, list[int | None]] = {}
    coeffs: dict[object, list[int | None]] = {}
    subtypes: dict[object, dict[object, int]] = {}

    total = len(sorted_types)

    for idx, (canon, (_count, edges, nv, ne)) in enumerate(sorted_types):
        desc = describe_graph(edges)
        print(
            f"  [{idx + 1}/{total}] {desc} ({ne} edges)...",
            end="",
            flush=True,
        )

        t0 = time.time()

        # Step 1: compute gamma_k(tau)
        gamma = gamma_sequence_edgelist(edges, max_k, max_edges=max_edges)
        gammas[canon] = gamma

        # Step 2: enumerate proper connected subgraphs of tau
        if ne > 1:
            sub_counts = enumerate_connected_subgraphs(edges, max_size=ne - 1)
        else:
            sub_counts = {}
        subtypes[canon] = {sc: sv[0] for sc, sv in sub_counts.items()}

        # Step 3: Mobius extraction
        coeff: list[int | None] = list(gamma)
        for sub_canon, sub_count in subtypes[canon].items():
            if sub_canon in coeffs:
                for k in range(min(len(coeff), len(coeffs[sub_canon]))):
                    if coeff[k] is not None and coeffs[sub_canon][k] is not None:
                        coeff[k] -= coeffs[sub_canon][k] * sub_count

        coeffs[canon] = coeff

        elapsed = time.time() - t0

        nonzero = [
            (k, c)
            for k, c in enumerate(coeff)
            if c is not None and c != 0 and k > 0
        ]
        if nonzero:
            nz_str = ", ".join(f"c_{k}={c}" for k, c in nonzero[:6])
            print(f" {nz_str} ({elapsed:.2f}s)")
        else:
            print(f" all zero ({elapsed:.2f}s)")

    return gammas, coeffs, subtypes


# ============================================================
#  Fiber analysis
# ============================================================

def run_fiber_analysis(
    dumbbell: list[tuple[int, int]],
    chorded_c6: list[tuple[int, int]],
    max_k: int,
    max_edges: int,
) -> tuple[
    dict[object, tuple[int, list[tuple[int, int]], int, int]],
    dict[object, tuple[int, list[tuple[int, int]], int, int]],
    dict[object, tuple[int, list[tuple[int, int]], int, int]],
    list[object],
    dict[object, list[int | None]],
    list[int | None],
    list[int | None],
]:
    """Run the fiber decomposition analysis and return computed data."""

    print("=" * 70)
    print("  WL-1 Equivalent Pair: Dumbbell vs Chorded C6")
    print("=" * 70)

    for name, edges in [("Dumbbell", dumbbell), ("Chorded C6", chorded_c6)]:
        degs = [0] * 6
        for u, v in edges:
            degs[u] += 1
            degs[v] += 1
        print(f"\n  {name}:")
        print(f"    Edges: {edges}")
        print(f"    Degree seq: {sorted(degs, reverse=True)}")

    # ---- Compute Graham sequences directly ----

    print(f"\n{'=' * 70}")
    print("  Graham Sequences (direct line graph iteration)")
    print(f"{'=' * 70}")

    seq_d = gamma_sequence_edgelist(dumbbell, max_k, max_edges=max_edges)
    seq_c = gamma_sequence_edgelist(chorded_c6, max_k, max_edges=max_edges)

    print(f"\n  {'k':>3s} {'Dumbbell':>14s} {'Chorded C6':>14s} {'diff':>10s}")
    print(f"  {'-' * 45}")
    for k in range(min(len(seq_d), len(seq_c))):
        if seq_d[k] is None or seq_c[k] is None:
            print(f"  {k:>3d} {'--':>14s} {'--':>14s}")
        else:
            d = seq_d[k] - seq_c[k]
            eq = "OK" if d == 0 else f"DIFF (diff={d})"
            print(f"  {k:>3d} {seq_d[k]:>14,} {seq_c[k]:>14,} {eq:>10s}")

    # ---- Enumerate subgraph types ----

    print(f"\n{'=' * 70}")
    print("  Enumerating connected subgraph types")
    print(f"{'=' * 70}")

    print("\n  Dumbbell subgraphs:")
    t0 = time.time()
    types_d = enumerate_connected_subgraphs(dumbbell)
    print(f"    {len(types_d)} types found ({time.time() - t0:.3f}s)")

    print("\n  Chorded C6 subgraphs:")
    t0 = time.time()
    types_c = enumerate_connected_subgraphs(chorded_c6)
    print(f"    {len(types_c)} types found ({time.time() - t0:.3f}s)")

    # Merge all types
    all_types: dict[object, tuple[int, list[tuple[int, int]], int, int]] = {}
    for canon, (count, edges, nv, ne) in types_d.items():
        all_types[canon] = (count, edges, nv, ne)
    for canon, (count, edges, nv, ne) in types_c.items():
        if canon not in all_types:
            all_types[canon] = (count, edges, nv, ne)

    all_canons = sorted(all_types.keys(), key=lambda c: (all_types[c][3], str(c)))

    print(f"\n  Total distinct types across both graphs: {len(all_types)}")

    # Show counts comparison
    print(
        f"\n  {'Type':>40s} {'#edges':>6s} {'Dumb':>6s} {'Chord':>6s} {'diff':>6s}"
    )
    print(f"  {'-' * 68}")

    for canon in all_canons:
        _, edges, nv, ne = all_types[canon]
        cd = types_d.get(canon, (0, None, 0, 0))[0]
        cc = types_c.get(canon, (0, None, 0, 0))[0]
        diff = cd - cc
        desc = describe_graph(edges)
        mark = " <-" if diff != 0 else ""
        print(f"  {desc:>40s} {ne:>6d} {cd:>6d} {cc:>6d} {diff:>+6d}{mark}")

    # ---- Compute coefficients ----

    print(f"\n{'=' * 70}")
    print("  Computing fiber coefficients (bootstrap)")
    print(f"{'=' * 70}\n")

    _gammas, coeffs, _subtypes_map = compute_all_coefficients(
        all_types, max_k, max_edges=max_edges
    )

    # ---- Verify decomposition ----

    print(f"\n{'=' * 70}")
    print("  Verification: gamma_k = sum coeff_k(tau) * count(tau, G)")
    print(f"{'=' * 70}")

    for name, _graph_edges, graph_types, graph_seq in [
        ("Dumbbell", dumbbell, types_d, seq_d),
        ("Chorded C6", chorded_c6, types_c, seq_c),
    ]:
        print(f"\n  {name}:")
        for k in range(1, max_k + 1):
            if k >= len(graph_seq) or graph_seq[k] is None:
                break
            total = 0
            for canon in all_canons:
                count = graph_types.get(canon, (0,))[0]
                if count == 0:
                    continue
                ck = coeffs.get(canon, [])
                if k < len(ck) and ck[k] is not None:
                    total += ck[k] * count

            match = "OK" if total == graph_seq[k] else f"MISMATCH (got {total})"
            print(f"    k={k}: sum = {total:,}  vs  gamma_{k} = {graph_seq[k]:,}  {match}")

    # ---- Analyse cancellation for WL-1 equivalence ----

    print(f"\n{'=' * 70}")
    print("  Cancellation Analysis: how differing counts produce equal gamma_k")
    print(f"{'=' * 70}")

    for k in range(1, max_k + 1):
        if k >= len(seq_d) or seq_d[k] is None:
            break

        contribs = []
        for canon in all_canons:
            _, edges, nv, ne = all_types[canon]
            cd = types_d.get(canon, (0,))[0]
            cc = types_c.get(canon, (0,))[0]

            ck = coeffs.get(canon, [])
            coeff_val = ck[k] if k < len(ck) and ck[k] is not None else 0

            if coeff_val != 0 and (cd != 0 or cc != 0):
                contrib_d = coeff_val * cd
                contrib_c = coeff_val * cc
                diff_contrib = contrib_d - contrib_c
                desc = describe_graph(edges)
                contribs.append(
                    (ne, desc, coeff_val, cd, cc, contrib_d, contrib_c, diff_contrib)
                )

        if not contribs:
            continue

        has_diff = any(c[7] != 0 for c in contribs)

        print(f"\n  Grade k = {k}:")
        print(
            f"    {'Type':>35s} {'coeff':>8s} {'cnt_D':>6s} {'cnt_C':>6s} "
            f"{'ctrb_D':>10s} {'ctrb_C':>10s} {'diff':>10s}"
        )
        print(f"    {'-' * 90}")

        total_d = 0
        total_c = 0

        for ne, desc, cv, cd, cc, ctd, ctc, dc in sorted(contribs):
            total_d += ctd
            total_c += ctc
            mark = " <-" if dc != 0 else ""
            print(
                f"    {desc:>35s} {cv:>8,} {cd:>6d} {cc:>6d} "
                f"{ctd:>10,} {ctc:>10,} {dc:>+10,}{mark}"
            )

        print(
            f"    {'':>35s} {'':>8s} {'':>6s} {'':>6s} "
            f"{total_d:>10,} {total_c:>10,} {total_d - total_c:>+10,}"
        )

        if has_diff:
            print(f"\n    Types contributing to cancellation at k={k}:")
            net = 0
            for ne, desc, cv, cd, cc, ctd, ctc, dc in sorted(contribs):
                if dc != 0:
                    net += dc
                    print(
                        f"      {desc}: coeff={cv}, count_diff={cd - cc}, "
                        f"contribution_diff={dc:+d} (running: {net:+d})"
                    )
            print(f"      Net difference: {net}")

    return all_types, types_d, types_c, all_canons, coeffs, seq_d, seq_c


# ============================================================
#  Linear dependence analysis
# ============================================================

def run_linear_dependence_analysis(
    all_types: dict[object, tuple[int, list[tuple[int, int]], int, int]],
    types_d: dict[object, tuple[int, list[tuple[int, int]], int, int]],
    types_c: dict[object, tuple[int, list[tuple[int, int]], int, int]],
    all_canons: list[object],
    coeffs: dict[object, list[int | None]],
    max_k: int,
) -> None:
    """Analyse linear dependencies among coefficient vectors."""

    print(f"\n{'=' * 70}")
    print("  Linear Dependence Analysis")
    print(f"{'=' * 70}")

    # Build data structures
    type_info: list[
        tuple[object, str, int, list[int], int, int, int]
    ] = []  # (canon, label, n_edges, coeff_vec, count_D, count_C, delta)

    for canon in all_canons:
        _, edges, nv, ne = all_types[canon]
        label = describe_graph(edges)

        cd = types_d.get(canon, (0,))[0]
        cc = types_c.get(canon, (0,))[0]
        delta = cd - cc

        coeff_vec: list[int] = []
        for k in range(1, max_k + 1):
            ck = coeffs.get(canon, [])
            if k < len(ck) and ck[k] is not None:
                coeff_vec.append(ck[k])
            else:
                coeff_vec.append(0)

        type_info.append((canon, label, ne, coeff_vec, cd, cc, delta))

    # Make labels unique
    label_counts: dict[str, int] = defaultdict(int)
    for ti in type_info:
        label_counts[ti[1]] += 1

    label_seen: dict[str, int] = defaultdict(int)
    unique_labels: list[str] = []
    for ti in type_info:
        label = ti[1]
        if label_counts[label] > 1:
            label_seen[label] += 1
            unique_labels.append(f"{label} #{label_seen[label]}")
        else:
            unique_labels.append(label)

    # Print the full data table
    print(
        f"\n  {'#':>3s} {'Label':>45s} {'|E|':>4s} {'cnt_D':>6s} {'cnt_C':>6s} {'D':>4s}",
        end="",
    )
    for k in range(1, max_k + 1):
        print(f" {'c_' + str(k):>8s}", end="")
    print()
    print(f"  {'-' * (70 + 9 * max_k)}")

    for i, (canon, label, ne, cvec, cd, cc, delta) in enumerate(type_info):
        ulabel = unique_labels[i]
        print(
            f"  {i + 1:>3d} {ulabel:>45s} {ne:>4d} {cd:>6d} {cc:>6d} {delta:>+4d}",
            end="",
        )
        for k in range(max_k):
            if cvec[k] != 0:
                print(f" {cvec[k]:>8d}", end="")
            else:
                print(f" {'*':>8s}", end="")
        print()

    # Restrict to types with nonzero Delta
    nz_indices = [i for i, ti in enumerate(type_info) if ti[6] != 0]
    n_nz = len(nz_indices)

    print(f"\n  Types with D != 0: {n_nz}")

    if n_nz == 0:
        print("  No types with nonzero Delta -- nothing to analyse.")
        return

    # Build coefficient matrix M (max_k x n_nz) with exact rationals
    M: list[list[Fraction]] = []
    for k in range(max_k):
        row: list[Fraction] = []
        for j in nz_indices:
            row.append(Fraction(type_info[j][3][k]))
        M.append(row)

    Delta_vec = [Fraction(type_info[j][6]) for j in nz_indices]
    nz_labels = [unique_labels[j] for j in nz_indices]

    # Verify M @ Delta = 0
    print(f"\n  Exact verification M * Delta:")
    all_zero = True
    for k in range(max_k):
        val = sum(M[k][j] * Delta_vec[j] for j in range(n_nz))
        status = "OK" if val == 0 else f"FAIL ({val})"
        print(f"    k={k + 1}: {status}")
        if val != 0:
            all_zero = False

    if not all_zero:
        print(f"\n  *** Delta is NOT in the null space! ***")
        print("  This indicates a data issue. Stopping here.")
        return

    print(f"\n  Delta is confirmed in the null space of M.")

    # Row reduce to find rank and null space
    mat, pivot_cols, rank = row_reduce_fraction(M, max_k, n_nz)
    free_cols = [j for j in range(n_nz) if j not in pivot_cols]

    print(f"\n  Rank of M (restricted): {rank}")
    print(f"  Null space dimension: {n_nz - rank}")
    print(f"\n  Pivot types ({rank}):")
    for pc in pivot_cols:
        print(f"    {nz_labels[pc]}")
    print(f"\n  Free types ({len(free_cols)}):")
    for fc in free_cols:
        print(f"    {nz_labels[fc]}")

    # Extract null space basis
    def lcm(a: int, b: int) -> int:
        return a * b // gcd(a, b)

    print(f"\n  {'=' * 60}")
    print("  Null Space Basis (exact, integer-scaled)")
    print(f"  {'=' * 60}")

    null_basis: list[list[Fraction]] = []
    for fi, fc in enumerate(free_cols):
        null_vec: list[Fraction] = [Fraction(0)] * n_nz
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
    print("  Delta as linear combination of null basis")
    print(f"  {'=' * 60}")

    print(f"\n  Delta = ", end="")
    terms: list[str] = []
    for fi, fc in enumerate(free_cols):
        coeff_val = Delta_vec[fc]
        if coeff_val != 0:
            terms.append(f"({coeff_val}) * v_{fi + 1}")
    print(" + ".join(terms) if terms else "0")

    # Verify reconstruction
    reconstructed: list[Fraction] = [Fraction(0)] * n_nz
    for fi, fc in enumerate(free_cols):
        for j in range(n_nz):
            reconstructed[j] += Delta_vec[fc] * null_basis[fi][j]

    match = all(reconstructed[j] == Delta_vec[j] for j in range(n_nz))
    print(f"  Reconstruction matches: {match}")

    if match:
        print(f"\n  Expanded:")
        for j in range(n_nz):
            if Delta_vec[j] != 0:
                print(
                    f"    {nz_labels[j]:>45s}: "
                    f"D = {Delta_vec[j]:>+3}, "
                    f"reconstructed = {reconstructed[j]:>+3}  OK"
                )

    # ============================================================
    #  Summary: which coefficient vectors are linearly dependent?
    # ============================================================

    print(f"\n  {'=' * 60}")
    print("  Summary of Linear Dependencies")
    print(f"  {'=' * 60}")

    # Check for identical coefficient vectors
    print(f"\n  Types with identical coefficient vectors:")
    found_identical = False
    for i in range(n_nz):
        for j in range(i + 1, n_nz):
            cvec_i = [M[k][i] for k in range(max_k)]
            cvec_j = [M[k][j] for k in range(max_k)]
            if cvec_i == cvec_j:
                found_identical = True
                print(f"    {nz_labels[i]}  =  {nz_labels[j]}")
                print(f"      vector: {[int(v) for v in cvec_i]}")
    if not found_identical:
        print("    (none)")

    # Check for proportional coefficient vectors
    print(f"\n  Types with proportional coefficient vectors:")
    found_proportional = False
    for i in range(n_nz):
        for j in range(i + 1, n_nz):
            cvec_i = [M[k][i] for k in range(max_k)]
            cvec_j = [M[k][j] for k in range(max_k)]
            ratio: Fraction | None = None
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
                found_proportional = True
                print(f"    {nz_labels[i]} = ({ratio}) x {nz_labels[j]}")
    if not found_proportional:
        print("    (none)")


# ============================================================
#  Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WL-1 fiber decomposition and linear dependence analysis"
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=7,
        help="maximum line-graph iteration depth (default: 7)",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=2_000_000,
        help="abort line-graph iteration when edge count exceeds this (default: 2000000)",
    )
    args = parser.parse_args()

    max_k: int = args.max_k
    max_edges: int = args.max_edges

    # ---- Define the graphs ----

    # Dumbbell: triangle 0-1-2, triangle 3-4-5, bridge 2-3
    dumbbell: list[tuple[int, int]] = [
        (0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (3, 5),
    ]

    # Chorded C6: cycle 0-1-2-3-4-5-0 plus chord 0-3
    chorded_c6: list[tuple[int, int]] = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (0, 3),
    ]

    # ---- Part 1: Fiber analysis ----

    all_types, types_d, types_c, all_canons, coeffs, seq_d, seq_c = (
        run_fiber_analysis(dumbbell, chorded_c6, max_k, max_edges)
    )

    # ---- Part 2: Linear dependence analysis ----

    run_linear_dependence_analysis(
        all_types, types_d, types_c, all_canons, coeffs, max_k
    )


if __name__ == "__main__":
    main()
