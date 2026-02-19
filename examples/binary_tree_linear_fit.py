"""
Exact linear fit of γ_k(T_{2,d}) = |V(L^k(T_{2,d}))| to the model α_k · 2^d + β_k.

For complete binary trees T_{2,d}:
  1. Compute γ_k for d = 2..15 (where feasible) and k = 0..6.
  2. Fit γ_k = α_k · 2^d + β_k using exact rational arithmetic (fractions.Fraction).
  3. Verify the fit against all computed data points.
  4. If the fit is not exact, investigate correction terms at small d.
"""
from __future__ import annotations

import sys
import time
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx

sys.path.insert(
    0, "/Users/hamiltonsawczuk/grahamtools/.claude/worktrees/amazing-kilby/src"
)

from grahamtools.linegraph.adjlist import edges_from_adj, line_graph_adj

VERTEX_CUTOFF = 5_000_000


# ---------------------------------------------------------------------------
# Tree construction
# ---------------------------------------------------------------------------
def complete_binary_tree_adj(depth: int) -> List[List[int]]:
    """Build a complete binary tree of given depth as an adjacency list."""
    G = nx.balanced_tree(r=2, h=depth)
    n = G.number_of_nodes()
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in G.edges():
        adj[u].append(v)
        adj[v].append(u)
    return [sorted(neigh) for neigh in adj]


def edge_count(adj: Sequence[Sequence[int]]) -> int:
    return len(edges_from_adj(adj))


# ---------------------------------------------------------------------------
# Compute gamma table
# ---------------------------------------------------------------------------
def compute_gamma_table(
    depths: List[int], k_max: int
) -> Dict[int, Dict[int, int]]:
    """gamma[k][d] = |V(L^k(T_{2,d}))|"""
    gamma: Dict[int, Dict[int, int]] = {}

    for depth in depths:
        print(f"  Computing depth d={depth} ...", end="", flush=True)
        t_start = time.perf_counter()
        adj = complete_binary_tree_adj(depth)
        cur = adj

        for k in range(k_max + 1):
            V = len(cur)
            E = edge_count(cur)
            gamma.setdefault(k, {})[depth] = V

            if V == 0 or E == 0:
                break

            # Safety check for next iterate
            next_V = E
            if next_V > VERTEX_CUTOFF:
                break

            if k < k_max:
                cur = line_graph_adj(cur)

        dt = time.perf_counter() - t_start
        ks_done = sorted(k2 for k2 in range(k_max + 1) if depth in gamma.get(k2, {}))
        print(f" done in {dt:.2f}s  (k=0..{max(ks_done)})")

    return gamma


# ---------------------------------------------------------------------------
# Exact rational fitting: γ = α · 2^d + β
# ---------------------------------------------------------------------------
def fit_linear_2d(
    data: Dict[int, int], d1: Optional[int] = None, d2: Optional[int] = None
) -> Tuple[Fraction, Fraction, Dict[int, Fraction]]:
    """
    Fit γ = α · 2^d + β using two specified depths (or the two largest by default).
    Returns (α, β, residuals_dict).
    """
    sorted_d = sorted(data.keys())
    if d1 is None or d2 is None:
        d2, d1 = sorted_d[-1], sorted_d[-2]
    g2, g1 = Fraction(data[d2]), Fraction(data[d1])
    pow2_d2, pow2_d1 = Fraction(2 ** d2), Fraction(2 ** d1)

    alpha = (g2 - g1) / (pow2_d2 - pow2_d1)
    beta = g1 - alpha * pow2_d1

    residuals: Dict[int, Fraction] = {}
    for d, g in data.items():
        predicted = alpha * Fraction(2 ** d) + beta
        residuals[d] = Fraction(g) - predicted

    return alpha, beta, residuals


# ---------------------------------------------------------------------------
# General n-parameter solver using Gaussian elimination
# ---------------------------------------------------------------------------
def solve_exact(
    basis_funcs,  # list of callables f(d) -> Fraction
    data: Dict[int, int],
    fit_points: Optional[List[int]] = None,  # which depths to use for fitting
) -> Tuple[List[Fraction], Dict[int, Fraction]]:
    """
    Fit γ = c_0 * f_0(d) + c_1 * f_1(d) + ... using exact arithmetic.
    Returns (coefficients, residuals).
    """
    n = len(basis_funcs)
    sorted_d = sorted(data.keys())
    if fit_points is None:
        fit_points = sorted_d[-n:]
    assert len(fit_points) == n

    # Build augmented matrix
    A = []
    for di in fit_points:
        row = [f(di) for f in basis_funcs] + [Fraction(data[di])]
        A.append(row)

    # Gaussian elimination with partial pivoting
    for col in range(n):
        pivot = None
        for row in range(col, n):
            if A[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            raise ValueError(f"Singular system at column {col}")
        A[col], A[pivot] = A[pivot], A[col]
        for row in range(n):
            if row == col:
                continue
            if A[row][col] == 0:
                continue
            factor = A[row][col] / A[col][col]
            for j in range(n + 1):
                A[row][j] -= factor * A[col][j]

    coeffs = [A[i][n] / A[i][i] for i in range(n)]

    residuals: Dict[int, Fraction] = {}
    for d, g in data.items():
        predicted = sum(c * f(d) for c, f in zip(coeffs, basis_funcs))
        residuals[d] = Fraction(g) - predicted

    return coeffs, residuals


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------
def frac_str(f: Fraction) -> str:
    """Pretty print a fraction."""
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"


def print_verification_table(
    data: Dict[int, int], residuals: Dict[int, Fraction], model_name: str
) -> bool:
    print(f"\n  Verification table ({model_name}):")
    print(f"  {'d':>4}  {'2^d':>10}  {'actual':>14}  {'predicted':>14}  {'residual':>12}")
    print(f"  {'----':>4}  {'----------':>10}  {'--------------':>14}  {'--------------':>14}  {'------------':>12}")
    all_exact = True
    for d in sorted(data.keys()):
        actual = data[d]
        res = residuals[d]
        predicted = actual - int(res)
        if res != 0:
            all_exact = False
        res_str = str(int(res)) if res.denominator == 1 else frac_str(res)
        print(
            f"  {d:>4}  {2**d:>10}  {actual:>14,}  {predicted:>14,}  {res_str:>12}"
        )
    return all_exact


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    k_max = 6
    depths = list(range(2, 16))  # d=2..15

    print("=" * 78)
    print("Computing gamma_k(T_{2,d}) = |V(L^k(T_{2,d}))| for complete binary trees")
    print(f"Depths: d = {depths[0]}..{depths[-1]},  k = 0..{k_max}")
    print(f"Vertex cutoff: {VERTEX_CUTOFF:,}")
    print("=" * 78)
    print()

    gamma = compute_gamma_table(depths, k_max)

    # Print the raw data table
    print()
    print("=" * 78)
    print("Raw data: gamma_k(T_{2,d})")
    print("=" * 78)

    all_depths_with_data = set()
    for k in range(k_max + 1):
        all_depths_with_data |= set(gamma.get(k, {}).keys())
    all_depths_sorted = sorted(all_depths_with_data)

    header = f"{'k':>4}" + "".join(f"{d:>14}" for d in all_depths_sorted)
    print(header)
    print("-" * len(header))
    for k in range(k_max + 1):
        row = f"{k:>4}"
        for d in all_depths_sorted:
            v = gamma.get(k, {}).get(d)
            row += f"{v:>14,}" if v is not None else f"{'--':>14}"
        print(row)

    # -----------------------------------------------------------------------
    # Fitting: gamma_k = alpha_k * 2^d + beta_k
    # -----------------------------------------------------------------------
    print()
    print("=" * 78)
    print("FITTING gamma_k = alpha_k * 2^d + beta_k  (exact rational arithmetic)")
    print("=" * 78)

    results = {}  # store (alpha, beta, exact_from_d, residuals)

    for k in range(k_max + 1):
        data = gamma.get(k, {})
        if len(data) < 2:
            print(f"\nk={k}: insufficient data ({len(data)} points)")
            continue

        print(f"\n{'=' * 78}")
        print(f"k = {k}  ({len(data)} data points: d = {sorted(data.keys())})")
        print(f"{'=' * 78}")

        # Fit using two largest depths
        alpha, beta, residuals = fit_linear_2d(data)
        print(f"\n  Model: gamma_{k}(T_{{2,d}}) = alpha * 2^d + beta")
        print(f"  alpha_{k} = {frac_str(alpha)}")
        print(f"  beta_{k}  = {frac_str(beta)}")

        all_exact = print_verification_table(data, residuals, "alpha*2^d + beta")

        # Find the minimum d from which the fit is exact
        nonzero_residuals = {d: r for d, r in residuals.items() if r != 0}
        if all_exact:
            print(f"\n  >>> FIT IS EXACT for all {len(data)} data points (d >= {min(data.keys())}) <<<")
            results[k] = (alpha, beta, min(data.keys()), residuals)
        else:
            max_bad_d = max(nonzero_residuals.keys())
            exact_from = max_bad_d + 1
            print(f"\n  Fit is EXACT for d >= {exact_from}, but has residuals at d = {sorted(nonzero_residuals.keys())}")
            results[k] = (alpha, beta, exact_from, residuals)

            # Analyze the residual structure
            print(f"\n  Residual analysis (values where fit breaks down):")
            for d in sorted(nonzero_residuals.keys()):
                r = nonzero_residuals[d]
                print(f"    d={d}: residual = {frac_str(r)}")

            # Check if residuals form a geometric sequence (powers of some base)
            res_values = [nonzero_residuals[d] for d in sorted(nonzero_residuals.keys())]
            if len(res_values) >= 2:
                ratios = []
                for i in range(len(res_values) - 1):
                    if res_values[i + 1] != 0:
                        ratios.append(res_values[i] / res_values[i + 1])
                if ratios:
                    print(f"  Ratios of consecutive residuals (d ascending): {[frac_str(r) for r in ratios]}")

    # -----------------------------------------------------------------------
    # Try to understand the residual structure for k=4,5,6
    # -----------------------------------------------------------------------
    print()
    print("=" * 78)
    print("INVESTIGATING RESIDUAL STRUCTURE FOR k >= 4")
    print("=" * 78)

    for k in [4, 5, 6]:
        data = gamma.get(k, {})
        if not data:
            continue
        alpha, beta, exact_from, residuals = results[k]
        nonzero = {d: residuals[d] for d in sorted(residuals.keys()) if residuals[d] != 0}
        if not nonzero:
            continue

        print(f"\n--- k = {k} ---")
        print(f"  Main formula (d >= {exact_from}): gamma_{k} = {frac_str(alpha)} * 2^d + ({frac_str(beta)})")
        print(f"  Residuals: {', '.join(f'd={d}: {frac_str(r)}' for d, r in sorted(nonzero.items()))}")

        # Try to fit residuals to various models
        # The key insight: maybe a different "boundary" contributes corrections at small d
        # Let's see if we can express gamma_k exactly with additional lower-order terms
        # that vanish for large d

        # For k=4, we only have residual at d=2 (value 8), which is hard to fit alone.
        # For k=5, residuals at d=2 (204) and d=3 (16).
        # For k=6, residuals at d=2 (5956), d=3 (888), d=4 (32).

        # Try: does the full sequence satisfy gamma = alpha * 2^d + beta + c * r^d for some r < 2?
        # If so, the correction is c * r^d which vanishes for large d.

        # For two residuals, we can solve: r1 = c * r^d1, r2 = c * r^d2
        # => r = (r1/r2)^(1/(d1-d2))
        # But we need exact. Let's check: is residual[d] = c * 1^d = c (constant)?

        if len(nonzero) >= 2:
            ds = sorted(nonzero.keys())
            # Check all possible integer bases for the correction term
            for base in [1, -1]:
                # Fit c * base^d to residuals using two points
                d0, d1_val = ds[0], ds[1]
                r0, r1_res = nonzero[d0], nonzero[d1_val]
                if base ** d0 == 0 or base ** d1_val == 0:
                    continue
                c_test = r0 / Fraction(base ** d0)
                c_test2 = r1_res / Fraction(base ** d1_val)
                if c_test == c_test2:
                    # Check all
                    ok = True
                    for d, r in nonzero.items():
                        if c_test * Fraction(base ** d) != r:
                            ok = False
                            break
                    if ok:
                        print(f"  Correction: residual = {frac_str(c_test)} * {base}^d (exact!)")

        # Let's also try: does it fit alpha * 2^d + beta + c * r^d for rational r?
        # Check if residuals fit c * (something)^d
        if len(nonzero) >= 2:
            ds = sorted(nonzero.keys())
            # Try to find ratio: res[d+1] / res[d]
            for i in range(len(ds) - 1):
                if ds[i+1] == ds[i] + 1 and nonzero[ds[i]] != 0:
                    ratio = nonzero[ds[i+1]] / nonzero[ds[i]]
                    print(f"  res[d={ds[i+1]}] / res[d={ds[i]}] = {frac_str(ratio)}")

            # General: try res = c * r^d  with r = (res[d2]/res[d1])^(1/(d2-d1))
            d0, d1_val = ds[0], ds[-1]
            r0, r1_res = nonzero[d0], nonzero[d1_val]
            # r^(d1-d0) = r1/r0
            gap = d1_val - d0
            ratio_total = r1_res / r0
            # Check if ratio_total is a perfect power: r^gap = ratio_total
            # Try small rational r
            print(f"  res[d={d1_val}] / res[d={d0}] = {frac_str(ratio_total)} (gap = {gap})")

    # -----------------------------------------------------------------------
    # Try fitting with a 3^d correction term for k=4,5,6
    # -----------------------------------------------------------------------
    print()
    print("=" * 78)
    print("TRYING RICHER EXACT MODELS")
    print("=" * 78)

    # Basis functions to try
    model_specs = [
        ("alpha*2^d + beta*3^d + gamma_const",
         [lambda d: Fraction(2**d), lambda d: Fraction(3**d), lambda d: Fraction(1)]),
        ("alpha*2^d + beta*4^d + gamma_const",
         [lambda d: Fraction(2**d), lambda d: Fraction(4**d), lambda d: Fraction(1)]),
        ("alpha*2^d + beta*(-1)^d + gamma_const",
         [lambda d: Fraction(2**d), lambda d: Fraction((-1)**d), lambda d: Fraction(1)]),
        ("alpha*2^d + beta*(-2)^d + gamma_const",
         [lambda d: Fraction(2**d), lambda d: Fraction((-2)**d), lambda d: Fraction(1)]),
    ]

    for k in range(k_max + 1):
        data = gamma.get(k, {})
        if len(data) < 3:
            continue

        alpha_lin, beta_lin, residuals_lin = fit_linear_2d(data)
        if all(r == 0 for r in residuals_lin.values()):
            continue  # already exact with linear model

        print(f"\n--- k = {k} ---")

        for model_name, basis in model_specs:
            try:
                coeffs, residuals = solve_exact(basis, data)
                all_exact = all(r == 0 for r in residuals.values())
                if all_exact:
                    coeff_strs = [frac_str(c) for c in coeffs]
                    print(f"  {model_name}: coeffs = {coeff_strs}")
                    print(f"  >>> EXACT! <<<")
                    print_verification_table(data, residuals, model_name)
                    break
            except Exception:
                pass

        else:
            # None of the simple models worked. Try more complex ones.
            # Try: alpha*2^d + beta*d*2^d + gamma*4^d + delta
            more_models = [
                ("a*2^d + b*d*2^d + c*4^d + d_const",
                 [lambda d: Fraction(2**d), lambda d: Fraction(d * 2**d),
                  lambda d: Fraction(4**d), lambda d: Fraction(1)]),
                ("a*2^d + b*d + c*d^2 + d_const",
                 [lambda d: Fraction(2**d), lambda d: Fraction(d),
                  lambda d: Fraction(d**2), lambda d: Fraction(1)]),
            ]
            for model_name, basis in more_models:
                if len(data) < len(basis):
                    continue
                try:
                    coeffs, residuals = solve_exact(basis, data)
                    all_exact = all(r == 0 for r in residuals.values())
                    if all_exact:
                        coeff_strs = [frac_str(c) for c in coeffs]
                        print(f"  {model_name}: coeffs = {coeff_strs}")
                        print(f"  >>> EXACT! <<<")
                        print_verification_table(data, residuals, model_name)
                        break
                except Exception:
                    pass
            else:
                print(f"  No simple exact model found. Showing best approximation residuals.")
                # Let's at least dump the residual pattern
                alpha_lin, beta_lin, res_lin = fit_linear_2d(data)
                nonzero = {d: res_lin[d] for d in sorted(res_lin.keys()) if res_lin[d] != 0}
                for d, r in sorted(nonzero.items()):
                    # Factor out powers of 2
                    ri = int(r)
                    twos = 0
                    temp = abs(ri)
                    while temp > 0 and temp % 2 == 0:
                        twos += 1
                        temp //= 2
                    print(f"    d={d}: residual = {ri} = {ri // (2**twos)} * 2^{twos}")

    # -----------------------------------------------------------------------
    # Deeper investigation: try to see if full formula involves multiple
    # exponential terms. Use all data points for an overdetermined system.
    # -----------------------------------------------------------------------
    print()
    print("=" * 78)
    print("SYSTEMATIC SEARCH: gamma_k = sum_i c_i * r_i^d + const")
    print("Try all combinations of small integer bases.")
    print("=" * 78)

    # For each k with non-exact linear fit, try gamma = c1*b1^d + c2*b2^d + c3
    # for all pairs (b1, b2) with b1, b2 in {-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8}
    # (but b1 != b2)
    candidate_bases = list(range(-4, 9))

    for k in [4, 5, 6]:
        data = gamma.get(k, {})
        if len(data) < 3:
            continue

        print(f"\n--- k = {k} ---")
        found = False

        for b1 in candidate_bases:
            if found:
                break
            for b2 in candidate_bases:
                if b2 <= b1:
                    continue
                if found:
                    break

                basis = [
                    (lambda d, b=b1: Fraction(b**d)),
                    (lambda d, b=b2: Fraction(b**d)),
                    (lambda d: Fraction(1)),
                ]
                try:
                    coeffs, residuals = solve_exact(basis, data)
                    all_exact = all(r == 0 for r in residuals.values())
                    if all_exact and any(c != 0 for c in coeffs):
                        print(f"  EXACT: gamma_{k} = ({frac_str(coeffs[0])}) * {b1}^d + ({frac_str(coeffs[1])}) * {b2}^d + ({frac_str(coeffs[2])})")
                        found = True
                except Exception:
                    pass

        if not found:
            # Try with 4 terms: c1*b1^d + c2*b2^d + c3*b3^d + const
            print(f"  No 2-base model found. Trying 3 bases...")
            for b1 in candidate_bases:
                if found:
                    break
                for b2 in candidate_bases:
                    if b2 <= b1 or found:
                        continue
                    for b3 in candidate_bases:
                        if b3 <= b2 or found:
                            continue
                        if len(data) < 4:
                            continue
                        basis = [
                            (lambda d, b=b1: Fraction(b**d)),
                            (lambda d, b=b2: Fraction(b**d)),
                            (lambda d, b=b3: Fraction(b**d)),
                            (lambda d: Fraction(1)),
                        ]
                        try:
                            coeffs, residuals = solve_exact(basis, data)
                            all_exact = all(r == 0 for r in residuals.values())
                            if all_exact and any(c != 0 for c in coeffs):
                                print(f"  EXACT: gamma_{k} = ({frac_str(coeffs[0])}) * {b1}^d + ({frac_str(coeffs[1])}) * {b2}^d + ({frac_str(coeffs[2])}) * {b3}^d + ({frac_str(coeffs[3])})")
                                print_verification_table(data, residuals, f"3-base model")
                                found = True
                        except Exception:
                            pass

        if not found:
            print(f"  No simple multi-base model found up to 3 bases with |base| <= 8.")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    print()
    print("For complete binary trees T_{2,d}, gamma_k = |V(L^k(T_{2,d}))|:")
    print()
    for k in range(k_max + 1):
        data = gamma.get(k, {})
        if len(data) < 2:
            continue
        alpha, beta, residuals = fit_linear_2d(data)
        nonzero = {d: r for d, r in residuals.items() if r != 0}
        if not nonzero:
            print(f"  k={k}:  gamma_{k} = {frac_str(alpha)} * 2^d + ({frac_str(beta)})       [EXACT for all d >= {min(data.keys())}]")
        else:
            max_bad = max(nonzero.keys())
            print(f"  k={k}:  gamma_{k} = {frac_str(alpha)} * 2^d + ({frac_str(beta)})       [EXACT for d >= {max_bad + 1}, corrections at d <= {max_bad}]")

    print()
    print("Sequence of alpha_k coefficients: ", end="")
    alphas = []
    for k in range(k_max + 1):
        data = gamma.get(k, {})
        if len(data) >= 2:
            alpha, _, _ = fit_linear_2d(data)
            alphas.append(alpha)
            print(f"{frac_str(alpha)}", end="  ")
    print()

    print("Sequence of beta_k coefficients:  ", end="")
    for k in range(k_max + 1):
        data = gamma.get(k, {})
        if len(data) >= 2:
            _, beta, _ = fit_linear_2d(data)
            print(f"{frac_str(beta)}", end="  ")
    print()

    # Check ratios of consecutive alpha values
    print()
    print("Ratios alpha_{k+1} / alpha_k:")
    for i in range(len(alphas) - 1):
        if alphas[i] != 0:
            ratio = alphas[i + 1] / alphas[i]
            print(f"  alpha_{i+1} / alpha_{i} = {frac_str(alphas[i+1])} / {frac_str(alphas[i])} = {float(ratio):.6f}")


if __name__ == "__main__":
    main()
