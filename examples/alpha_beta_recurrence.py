"""
Systematic search for recurrence relations in the α_k and β_k sequences
arising from iterated line graphs of complete binary trees.

We have:  γ_k(T_{2,d}) = α_k · 2^d + β_k

    α_k: 2, 2, 3,   7,   29,    227,     3469
    β_k: -1, -2, -5, -18, -104, -1048,  -19468

This script checks whether these sequences satisfy any low-order recurrence
using exact rational arithmetic (fractions.Fraction).

Key criterion: we only trust a recurrence if the number of verification
data points EXCEEDS the number of unknowns (overdetermined check).
"""
from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Tuple, Dict


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────
alpha = [Fraction(x) for x in [2, 2, 3, 7, 29, 227, 3469]]
beta  = [Fraction(x) for x in [-1, -2, -5, -18, -104, -1048, -19468]]

K = len(alpha)  # 7 terms, indices 0..6


# ──────────────────────────────────────────────────────────────────────────────
# Exact linear solver (Gaussian elimination over Fraction)
# ──────────────────────────────────────────────────────────────────────────────
def solve_exact(A: List[List[Fraction]], b: List[Fraction]) -> Optional[List[Fraction]]:
    """Solve A·x = b exactly using Gaussian elimination with partial pivoting."""
    n = len(A)
    if len(b) != n:
        return None
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for col in range(n):
        pivot = None
        for row in range(col, n):
            if M[row][col] != 0:
                pivot = row
                break
        if pivot is None:
            return None
        M[col], M[pivot] = M[pivot], M[col]
        for row in range(col + 1, n):
            if M[row][col] != 0:
                factor = M[row][col] / M[col][col]
                for j in range(col, n + 1):
                    M[row][j] -= factor * M[col][j]

    x = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        if M[i][i] == 0:
            return None
        x[i] /= M[i][i]
    return x


def try_recurrence(name: str, seq: List[Fraction],
                   build_row, min_idx: int, num_unknowns: int,
                   extra_seqs=None, quiet=False) -> Optional[Tuple[List[Fraction], int, Optional[Fraction]]]:
    """
    Generic recurrence tester.

    build_row(k, seq, extra_seqs) -> (list_of_coefficients, rhs)

    Returns (coefficients, num_verified, prediction) or None.
    num_verified = number of data points checked (including those used to solve).
    A recurrence is "trustworthy" only if num_verified > num_unknowns.
    """
    all_rows = []
    all_rhs = []
    for k in range(min_idx, len(seq) - 1):
        try:
            row, rhs = build_row(k, seq, extra_seqs)
            all_rows.append(row)
            all_rhs.append(rhs)
        except (IndexError, KeyError):
            break

    if len(all_rows) < num_unknowns:
        if not quiet:
            print(f"  [{name}] Not enough data ({len(all_rows)} eqns for {num_unknowns} unknowns)")
        return None

    A = all_rows[:num_unknowns]
    b_vec = all_rhs[:num_unknowns]
    sol = solve_exact(A, b_vec)
    if sol is None:
        if not quiet:
            print(f"  [{name}] Singular system")
        return None

    # Verify against ALL equations
    verified = 0
    for row, rhs in zip(all_rows, all_rhs):
        predicted = sum(c * s for c, s in zip(sol, row))
        if predicted != rhs:
            if not quiet:
                print(f"  [{name}] FAILED at verification point {verified + 1}/{len(all_rows)}")
            return None
        verified += 1

    excess = verified - num_unknowns  # how many extra checks passed

    # Try prediction
    pred = None
    k_last = len(seq) - 1
    try:
        # Build row for the last known term predicting the NEXT unknown term
        row_next, _ = build_row(k_last, seq + [Fraction(0)], extra_seqs)
        pred = sum(c * s for c, s in zip(sol, row_next))
    except Exception:
        pass

    # Check if coefficients are nice (integer or simple fraction)
    all_integer = all(c.denominator == 1 for c in sol)
    coeff_str = ", ".join(str(c) for c in sol)

    status = "EXACT" if excess > 0 else "FITTED (no excess)"
    marker = "***" if excess > 0 and all_integer else ""

    if not quiet:
        print(f"  [{name}] {status}: verified {verified}/{len(all_rows)}, "
              f"{num_unknowns} unknowns, excess={excess}")
        int_tag = " (all integer)" if all_integer else " (fractional)"
        print(f"    Coefficients{int_tag}: [{coeff_str}]")
        if pred is not None:
            print(f"    Prediction for next term: {pred}" +
                  (f" = {int(pred)}" if pred.denominator == 1 else f" (NOT integer!)"))
        if marker:
            print(f"    {marker} STRONG CANDIDATE {marker}")
    
    return (sol, verified, pred)


# ──────────────────────────────────────────────────────────────────────────────
# Banner
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("RECURRENCE ANALYSIS FOR alpha AND beta SEQUENCES")
print("  alpha_k: 2, 2, 3, 7, 29, 227, 3469")
print("  beta_k:  -1, -2, -5, -18, -104, -1048, -19468")
print("  From: gamma_k(T_{2,d}) = alpha_k * 2^d + beta_k")
print()
print("  Criterion: a recurrence with N unknowns needs >N data points to be trusted.")
print("  (otherwise it is just fitting, not a genuine pattern)")
print("=" * 80)
print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: LINEAR RECURRENCES
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("SECTION 1: LINEAR RECURRENCES")
print("=" * 80)
print()

for label, seq in [("alpha", alpha), ("beta", beta)]:
    # 3-term: x_{k+1} = a*x_k + b*x_{k-1} + c  (3 unknowns, need from k=1)
    def build_3(k, s, _):
        return [s[k], s[k-1], Fraction(1)], s[k+1]
    try_recurrence(f"{label}: x_{{k+1}} = a*x_k + b*x_{{k-1}} + c", seq, build_3, 1, 3)
    print()

    # 4-term: x_{k+1} = a*x_k + b*x_{k-1} + c*x_{k-2} + d  (4 unknowns, need from k=2)
    def build_4(k, s, _):
        return [s[k], s[k-1], s[k-2], Fraction(1)], s[k+1]
    try_recurrence(f"{label}: x_{{k+1}} = a*x_k + b*x_{{k-1}} + c*x_{{k-2}} + d", seq, build_4, 2, 4)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: QUADRATIC RECURRENCES (single-sequence)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("SECTION 2: QUADRATIC / PRODUCT RECURRENCES (single-sequence)")
print("=" * 80)
print()

for label, seq in [("alpha", alpha), ("beta", beta)]:
    # x_{k+1} = a*x_k^2 + b*x_k + c  (3 unknowns)
    def build_q1(k, s, _):
        return [s[k]**2, s[k], Fraction(1)], s[k+1]
    try_recurrence(f"{label}: x_{{k+1}} = a*x_k^2 + b*x_k + c", seq, build_q1, 0, 3)
    print()

    # x_{k+1} = a*x_k*x_{k-1} + b*x_k + c*x_{k-1} + d  (4 unknowns)
    def build_q2(k, s, _):
        return [s[k]*s[k-1], s[k], s[k-1], Fraction(1)], s[k+1]
    try_recurrence(f"{label}: x_{{k+1}} = a*x_k*x_{{k-1}} + b*x_k + c*x_{{k-1}} + d", seq, build_q2, 1, 4)
    print()

    # x_{k+1} = a*x_k^2 + b*x_k*x_{k-1} + c*x_k + d*x_{k-1} + e  (5 unknowns)
    def build_q3(k, s, _):
        return [s[k]**2, s[k]*s[k-1], s[k], s[k-1], Fraction(1)], s[k+1]
    try_recurrence(f"{label}: x_{{k+1}} = a*x_k^2 + b*x_k*x_{{k-1}} + c*x_k + d*x_{{k-1}} + e", seq, build_q3, 1, 5)
    print()

    # x_{k+1} = a*x_k^2 + b*x_{k-1}^2 + c*x_k + d*x_{k-1} + e  (5 unknowns)
    def build_q4(k, s, _):
        return [s[k]**2, s[k-1]**2, s[k], s[k-1], Fraction(1)], s[k+1]
    try_recurrence(f"{label}: x_{{k+1}} = a*x_k^2 + b*x_{{k-1}}^2 + c*x_k + d*x_{{k-1}} + e", seq, build_q4, 1, 5)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: SPECIAL FORMS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("SECTION 3: SPECIAL FORMS")
print("=" * 80)
print()

# x_{k+1} = x_k*(x_k - 1) + c
print("--- alpha_{k+1} = alpha_k * (alpha_k - 1) + c? ---")
for k in range(K - 1):
    delta = alpha[k+1] - alpha[k] * (alpha[k] - 1)
    print(f"  k={k}: alpha_k*(alpha_k-1) = {int(alpha[k]*(alpha[k]-1))}, "
          f"alpha_{{k+1}} = {int(alpha[k+1])}, residual = {int(delta)}")
print("  Not constant => NO\n")

# x_{k+1} = x_k^2 + a*x_k + b
for label, seq in [("alpha", alpha), ("beta", beta)]:
    def build_sp1(k, s, _):
        return [s[k], Fraction(1)], s[k+1] - s[k]**2
    try_recurrence(f"{label}: x_{{k+1}} = x_k^2 + a*x_k + b", seq, build_sp1, 0, 2)
    print()

# x_{k+1} = x_k*(x_{k-1} + a) + b
for label, seq in [("alpha", alpha), ("beta", beta)]:
    def build_sp2(k, s, _):
        return [s[k], Fraction(1)], s[k+1] - s[k]*s[k-1]
    try_recurrence(f"{label}: x_{{k+1}} = x_k*(x_{{k-1}} + a) + b", seq, build_sp2, 1, 2)
    print()

# x_{k+1} = a*x_k^2 - x_k + c
for label, seq in [("alpha", alpha), ("beta", beta)]:
    def build_sp3(k, s, _):
        return [s[k]**2, Fraction(1)], s[k+1] + s[k]
    try_recurrence(f"{label}: x_{{k+1}} = a*x_k^2 - x_k + c", seq, build_sp3, 0, 2)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: CROSS-SEQUENCE RECURRENCES
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("SECTION 4: CROSS-SEQUENCE RECURRENCES")
print("=" * 80)
print()

# alpha_{k+1} = a*alpha_k + b*beta_k + c
def build_c1(k, s, extra):
    return [s[k], extra[0][k], Fraction(1)], s[k+1]
try_recurrence("alpha_{k+1} = a*alpha_k + b*beta_k + c", alpha, build_c1, 0, 3, extra_seqs=[beta])
print()

# beta_{k+1} = a*alpha_k + b*beta_k + c
def build_c2(k, s, extra):
    return [extra[0][k], s[k], Fraction(1)], s[k+1]
try_recurrence("beta_{k+1} = a*alpha_k + b*beta_k + c", beta, build_c2, 0, 3, extra_seqs=[alpha])
print()

# alpha_{k+1} = a*alpha_k*beta_k + b*alpha_k + c*beta_k + d
def build_c3(k, s, extra):
    b = extra[0]
    return [s[k]*b[k], s[k], b[k], Fraction(1)], s[k+1]
try_recurrence("alpha_{k+1} = a*alpha_k*beta_k + b*alpha_k + c*beta_k + d", alpha, build_c3, 0, 4, extra_seqs=[beta])
print()

# beta_{k+1} = a*alpha_k*beta_k + b*alpha_k + c*beta_k + d
def build_c4(k, s, extra):
    a_seq = extra[0]
    return [a_seq[k]*s[k], a_seq[k], s[k], Fraction(1)], s[k+1]
try_recurrence("beta_{k+1} = a*alpha_k*beta_k + b*alpha_k + c*beta_k + d", beta, build_c4, 0, 4, extra_seqs=[alpha])
print()

# alpha_{k+1} = a*alpha_k^2 + b*alpha_k + c*beta_k + d
def build_c5(k, s, extra):
    return [s[k]**2, s[k], extra[0][k], Fraction(1)], s[k+1]
try_recurrence("alpha_{k+1} = a*alpha_k^2 + b*alpha_k + c*beta_k + d", alpha, build_c5, 0, 4, extra_seqs=[beta])
print()

# beta_{k+1} = a*beta_k^2 + b*beta_k + c*alpha_k + d
def build_c6(k, s, extra):
    return [s[k]**2, s[k], extra[0][k], Fraction(1)], s[k+1]
try_recurrence("beta_{k+1} = a*beta_k^2 + b*beta_k + c*alpha_k + d", beta, build_c6, 0, 4, extra_seqs=[alpha])
print()

# alpha_{k+1} = a*alpha_k^2 + b*alpha_k*beta_k + c*alpha_k + d*beta_k + e
def build_c7(k, s, extra):
    b = extra[0]
    return [s[k]**2, s[k]*b[k], s[k], b[k], Fraction(1)], s[k+1]
try_recurrence("alpha_{k+1} = a*a^2 + b*a*B + c*a + d*B + e", alpha, build_c7, 0, 5, extra_seqs=[beta])
print()

# beta_{k+1} = a*beta_k^2 + b*alpha_k*beta_k + c*beta_k + d*alpha_k + e
def build_c8(k, s, extra):
    a_seq = extra[0]
    return [s[k]**2, a_seq[k]*s[k], s[k], a_seq[k], Fraction(1)], s[k+1]
try_recurrence("beta_{k+1} = a*B^2 + b*a*B + c*B + d*a + e", beta, build_c8, 0, 5, extra_seqs=[alpha])
print()

# Full quadratic cross (6 unknowns, 6 equations for k=0..5 -> 0 excess but let's see)
def build_c9(k, s, extra):
    b = extra[0]
    return [s[k]**2, s[k]*b[k], b[k]**2, s[k], b[k], Fraction(1)], s[k+1]
try_recurrence("alpha_{k+1} = full quad(alpha_k, beta_k)", alpha, build_c9, 0, 6, extra_seqs=[beta])
print()

def build_c10(k, s, extra):
    a_seq = extra[0]
    return [a_seq[k]**2, a_seq[k]*s[k], s[k]**2, a_seq[k], s[k], Fraction(1)], s[k+1]
try_recurrence("beta_{k+1} = full quad(alpha_k, beta_k)", beta, build_c10, 0, 6, extra_seqs=[alpha])
print()

# Lagged cross-linear: 5 unknowns
def build_c11(k, s, extra):
    b = extra[0]
    return [s[k], s[k-1], b[k], b[k-1], Fraction(1)], s[k+1]
try_recurrence("alpha_{k+1} = a*a_k + b*a_{k-1} + c*B_k + d*B_{k-1} + e", alpha, build_c11, 1, 5, extra_seqs=[beta])
print()

def build_c12(k, s, extra):
    a_seq = extra[0]
    return [a_seq[k], a_seq[k-1], s[k], s[k-1], Fraction(1)], s[k+1]
try_recurrence("beta_{k+1} = a*a_k + b*a_{k-1} + c*B_k + d*B_{k-1} + e", beta, build_c12, 1, 5, extra_seqs=[alpha])
print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: POINTWISE RELATIONSHIPS alpha <-> beta
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("SECTION 5: POINTWISE RELATIONSHIPS alpha <-> beta")
print("=" * 80)
print()

print("  k   alpha_k   beta_k    beta/alpha   alpha+beta  alpha*beta    alpha^2+beta")
for k in range(K):
    a, b = alpha[k], beta[k]
    r = b / a if a != 0 else None
    print(f"  {k}   {int(a):>7}  {int(b):>7}   {str(r):>10}   {int(a+b):>9}  {int(a*b):>10}    {int(a**2 + b):>10}")
print()

# sigma_k = alpha_k + beta_k
sigma = [alpha[k] + beta[k] for k in range(K)]
print(f"  sigma_k = alpha_k + beta_k: {[int(x) for x in sigma]}")
print()

# gamma for small d
for d in range(0, 5):
    gamma_d = [int(alpha[k] * (2**d) + beta[k]) for k in range(K)]
    print(f"  gamma_k(d={d}): {gamma_d}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: RATIO AND GROWTH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("SECTION 6: RATIO AND GROWTH ANALYSIS")
print("=" * 80)
print()

print("  Successive ratios alpha_{k+1}/alpha_k:")
for k in range(K - 1):
    if alpha[k] != 0:
        r = alpha[k+1] / alpha[k]
        print(f"    k={k}: {r} = {float(r):.6f}")
print()

print("  Successive ratios beta_{k+1}/beta_k:")
for k in range(K - 1):
    if beta[k] != 0:
        r = beta[k+1] / beta[k]
        print(f"    k={k}: {r} = {float(r):.6f}")
print()

print("  Log2 of |alpha_k|:")
import math
for k in range(K):
    if alpha[k] > 0:
        print(f"    k={k}: log2({int(alpha[k])}) = {math.log2(float(alpha[k])):.4f}")
print()

print("  Log2 of |beta_k|:")
for k in range(K):
    if beta[k] != 0:
        print(f"    k={k}: log2({abs(int(beta[k]))}) = {math.log2(abs(float(beta[k]))):.4f}")
print()

# alpha_k * (alpha_k - 1) vs alpha_{k+1}
print("  alpha_k^2 - alpha_k vs alpha_{k+1}:")
for k in range(K - 1):
    val = alpha[k]**2 - alpha[k]
    print(f"    k={k}: alpha_k^2 - alpha_k = {int(val)}, alpha_{{k+1}} = {int(alpha[k+1])}, "
          f"ratio = {float(alpha[k+1]/val) if val else 'inf':.6f}")
print()

# Check second differences, etc.
print("  First differences Delta alpha_k = alpha_{k+1} - alpha_k:")
d1 = [alpha[k+1] - alpha[k] for k in range(K-1)]
print(f"    {[int(x) for x in d1]}")

print("  Second differences Delta^2 alpha_k:")
d2 = [d1[k+1] - d1[k] for k in range(len(d1)-1)]
print(f"    {[int(x) for x in d2]}")

print("  Ratios of first differences:")
for k in range(len(d1) - 1):
    if d1[k] != 0:
        print(f"    k={k}: {d1[k+1]}/{d1[k]} = {float(d1[k+1]/d1[k]):.4f}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: OEIS-STYLE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("SECTION 7: OEIS-STYLE COMPARISON")
print("=" * 80)
print()

# Sylvester sequence
print("  Sylvester sequence (a_{n+1} = a_n^2 - a_n + 1): 2, 3, 7, 43, 1807, ...")
sylv = [Fraction(2)]
for _ in range(7):
    sylv.append(sylv[-1]**2 - sylv[-1] + 1)
print(f"    {[int(x) for x in sylv[:8]]}")
print(f"    alpha: {[int(x) for x in alpha]}")
print("    Note: alpha matches Sylvester at positions 0,2,3 but diverges at 4")
print()

# Check if alpha satisfies a_n = product of previous + 1 (Sylvester-like)
print("  Checking partial products prod_{i<k} alpha_i:")
prod = Fraction(1)
for k in range(K):
    print(f"    k={k}: prod = {int(prod)}, alpha_k = {int(alpha[k])}, prod+1 = {int(prod+1)}")
    prod *= alpha[k]
print()

# Catalan numbers: 1, 1, 2, 5, 14, 42, 132, ...
catalan = [Fraction(1)]
for n in range(1, 10):
    catalan.append(catalan[-1] * 2 * (2*n - 1) / (n + 1))
print(f"  Catalan: {[int(x) for x in catalan[:10]]}")
print(f"  alpha:   {[int(x) for x in alpha]}")
print()

# Check alpha_k * (alpha_k - 1) + 1 pattern
print("  alpha_k^2 - alpha_k + 1:")
for k in range(K):
    val = alpha[k]**2 - alpha[k] + 1
    print(f"    k={k}: {int(val)}", end="")
    if k < K - 1:
        print(f"  (alpha_{{k+1}} = {int(alpha[k+1])}, ratio = {float(alpha[k+1])/float(val):.6f})")
    else:
        print()
print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: SYSTEMATIC BRUTE FORCE — 2-parameter families
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("SECTION 8: SYSTEMATIC BRUTE-FORCE — 2-PARAMETER FAMILIES")
print("  (2 unknowns, so we need at least 3 data points => 1+ excess)")
print("=" * 80)
print()

results_2param = []

def test_2param(name, seq, build_fn, min_idx, extra=None, label=""):
    """Test a 2-parameter recurrence, requiring at least 1 excess verification."""
    all_rows = []
    all_rhs = []
    for k in range(min_idx, len(seq) - 1):
        try:
            row, rhs = build_fn(k, seq, extra)
            all_rows.append(row)
            all_rhs.append(rhs)
        except (IndexError, KeyError):
            break
    if len(all_rows) < 3:  # need at least 3 for 2 unknowns + 1 check
        return
    sol = solve_exact(all_rows[:2], all_rhs[:2])
    if sol is None:
        return
    for row, rhs in zip(all_rows, all_rhs):
        if sum(c*s for c, s in zip(sol, row)) != rhs:
            return
    # ALL match
    pred = None
    try:
        k_last = len(seq) - 1
        row_next, _ = build_fn(k_last, seq + [Fraction(0)], extra)
        pred = sum(c*s for c, s in zip(sol, row_next))
    except Exception:
        pass
    int_coeffs = all(c.denominator == 1 for c in sol)
    results_2param.append((label + name, sol, len(all_rows), pred, int_coeffs))

# For alpha
for label, seq, extra in [("alpha", alpha, None), ("beta", beta, None)]:
    # x_{k+1} - x_k^2 = a*x_k + b
    test_2param("x_{k+1} = x_k^2 + a*x_k + b",
                seq, lambda k,s,_: ([s[k], Fraction(1)], s[k+1] - s[k]**2), 0, label=label+": ")
    
    # x_{k+1} - x_k*x_{k-1} = a*x_k + b
    test_2param("x_{k+1} = x_k*x_{k-1} + a*x_k + b",
                seq, lambda k,s,_: ([s[k], Fraction(1)], s[k+1] - s[k]*s[k-1]), 1, label=label+": ")
    
    # x_{k+1} = a*x_k^2 + b  (fix linear coeff to 0)
    test_2param("x_{k+1} = a*x_k^2 + b",
                seq, lambda k,s,_: ([s[k]**2, Fraction(1)], s[k+1]), 0, label=label+": ")
    
    # x_{k+1} = a*x_k*x_{k-1} + b
    test_2param("x_{k+1} = a*x_k*x_{k-1} + b",
                seq, lambda k,s,_: ([s[k]*s[k-1], Fraction(1)], s[k+1]), 1, label=label+": ")
    
    # x_{k+1} = a*x_k^2 - x_k + b
    test_2param("x_{k+1} = a*x_k^2 - x_k + b",
                seq, lambda k,s,_: ([s[k]**2, Fraction(1)], s[k+1] + s[k]), 0, label=label+": ")
    
    # x_{k+1} + x_k = a*x_k*x_{k-1} + b
    test_2param("x_{k+1} = a*x_k*x_{k-1} - x_k + b",
                seq, lambda k,s,_: ([s[k]*s[k-1], Fraction(1)], s[k+1] + s[k]), 1, label=label+": ")

# Cross-sequence 2-param
# alpha_{k+1} = a*beta_k + b
test_2param("alpha_{k+1} = a*beta_k + b",
            alpha, lambda k,s,e: ([e[0][k], Fraction(1)], s[k+1]), 0, extra=[beta], label="cross: ")

# beta_{k+1} = a*alpha_k + b
test_2param("beta_{k+1} = a*alpha_k + b",
            beta, lambda k,s,e: ([e[0][k], Fraction(1)], s[k+1]), 0, extra=[alpha], label="cross: ")

# alpha_{k+1} = a*alpha_k*beta_k + b
test_2param("alpha_{k+1} = a*alpha_k*beta_k + b",
            alpha, lambda k,s,e: ([s[k]*e[0][k], Fraction(1)], s[k+1]), 0, extra=[beta], label="cross: ")

# beta_{k+1} = a*alpha_k*beta_k + b
test_2param("beta_{k+1} = a*alpha_k*beta_k + b",
            beta, lambda k,s,e: ([s[k]*e[0][k], Fraction(1)], s[k+1]), 0, extra=[alpha], label="cross: ")

# alpha_{k+1} - alpha_k^2 = a*beta_k + b
test_2param("alpha_{k+1} = alpha_k^2 + a*beta_k + b",
            alpha, lambda k,s,e: ([e[0][k], Fraction(1)], s[k+1] - s[k]**2), 0, extra=[beta], label="cross: ")

# beta_{k+1} - beta_k^2 = a*alpha_k + b
test_2param("beta_{k+1} = beta_k^2 + a*alpha_k + b",
            beta, lambda k,s,e: ([e[0][k], Fraction(1)], s[k+1] - s[k]**2), 0, extra=[alpha], label="cross: ")

# alpha_{k+1} = a*(alpha_k + beta_k) + b
test_2param("alpha_{k+1} = a*(alpha_k + beta_k) + b",
            alpha, lambda k,s,e: ([s[k] + e[0][k], Fraction(1)], s[k+1]), 0, extra=[beta], label="cross: ")

# alpha_{k+1} = a*(alpha_k - beta_k) + b
test_2param("alpha_{k+1} = a*(alpha_k - beta_k) + b",
            alpha, lambda k,s,e: ([s[k] - e[0][k], Fraction(1)], s[k+1]), 0, extra=[beta], label="cross: ")

if results_2param:
    print("  MATCHES found (2-parameter families, verified on >2 points):")
    for name, sol, nv, pred, is_int in results_2param:
        tag = " *** INTEGER COEFFS ***" if is_int else ""
        print(f"    {name}")
        print(f"      coeffs = [{', '.join(str(c) for c in sol)}]{tag}")
        print(f"      verified on {nv} points")
        if pred is not None:
            frac_tag = "" if pred.denominator == 1 else " (NOT integer!)"
            print(f"      prediction: {pred}{frac_tag}")
else:
    print("  No exact matches in 2-parameter families.")
print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: SYSTEMATIC BRUTE FORCE — 3-parameter families
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("SECTION 9: SYSTEMATIC BRUTE-FORCE — 3-PARAMETER FAMILIES")
print("  (3 unknowns, need >= 4 data points => 1+ excess)")
print("=" * 80)
print()

results_3param = []

def test_3param(name, seq, build_fn, min_idx, extra=None, label=""):
    """Test a 3-parameter recurrence, requiring at least 1 excess."""
    all_rows = []
    all_rhs = []
    for k in range(min_idx, len(seq) - 1):
        try:
            row, rhs = build_fn(k, seq, extra)
            all_rows.append(row)
            all_rhs.append(rhs)
        except (IndexError, KeyError):
            break
    if len(all_rows) < 4:  # need 4 for 3 unknowns + 1 check
        return
    sol = solve_exact(all_rows[:3], all_rhs[:3])
    if sol is None:
        return
    for row, rhs in zip(all_rows, all_rhs):
        if sum(c*s for c, s in zip(sol, row)) != rhs:
            return
    pred = None
    try:
        k_last = len(seq) - 1
        row_next, _ = build_fn(k_last, seq + [Fraction(0)], extra)
        pred = sum(c*s for c, s in zip(sol, row_next))
    except Exception:
        pass
    int_coeffs = all(c.denominator == 1 for c in sol)
    results_3param.append((label + name, sol, len(all_rows), pred, int_coeffs))


for label, seq, extra_s in [("alpha", alpha, beta), ("beta", beta, alpha)]:
    other_label = "beta" if label == "alpha" else "alpha"
    other = extra_s

    # x_{k+1} = a*x_k^2 + b*x_k + c
    test_3param(f"{label}_{{k+1}} = a*{label}_k^2 + b*{label}_k + c",
                seq, lambda k,s,_: ([s[k]**2, s[k], Fraction(1)], s[k+1]), 0, label="")

    # x_{k+1} = a*x_k + b*x_{k-1} + c
    test_3param(f"{label}_{{k+1}} = a*{label}_k + b*{label}_{{k-1}} + c",
                seq, lambda k,s,_: ([s[k], s[k-1], Fraction(1)], s[k+1]), 1, label="")

    # x_{k+1} = a*x_k*x_{k-1} + b*x_k + c
    test_3param(f"{label}_{{k+1}} = a*{label}_k*{label}_{{k-1}} + b*{label}_k + c",
                seq, lambda k,s,_: ([s[k]*s[k-1], s[k], Fraction(1)], s[k+1]), 1, label="")

    # x_{k+1} = a*x_k*x_{k-1} + b*x_{k-1} + c
    test_3param(f"{label}_{{k+1}} = a*{label}_k*{label}_{{k-1}} + b*{label}_{{k-1}} + c",
                seq, lambda k,s,_: ([s[k]*s[k-1], s[k-1], Fraction(1)], s[k+1]), 1, label="")

    # Cross: x_{k+1} = a*x_k + b*other_k + c
    test_3param(f"{label}_{{k+1}} = a*{label}_k + b*{other_label}_k + c",
                seq, lambda k,s,e: ([s[k], e[0][k], Fraction(1)], s[k+1]), 0, extra=[other], label="")

    # Cross: x_{k+1} = a*x_k*other_k + b*x_k + c
    test_3param(f"{label}_{{k+1}} = a*{label}_k*{other_label}_k + b*{label}_k + c",
                seq, lambda k,s,e: ([s[k]*e[0][k], s[k], Fraction(1)], s[k+1]), 0, extra=[other], label="")

    # Cross: x_{k+1} = a*x_k*other_k + b*other_k + c
    test_3param(f"{label}_{{k+1}} = a*{label}_k*{other_label}_k + b*{other_label}_k + c",
                seq, lambda k,s,e: ([s[k]*e[0][k], e[0][k], Fraction(1)], s[k+1]), 0, extra=[other], label="")

    # x_{k+1} = a*x_k^2 + b*other_k + c
    test_3param(f"{label}_{{k+1}} = a*{label}_k^2 + b*{other_label}_k + c",
                seq, lambda k,s,e: ([s[k]**2, e[0][k], Fraction(1)], s[k+1]), 0, extra=[other], label="")

    # x_{k+1} = a*other_k^2 + b*x_k + c
    test_3param(f"{label}_{{k+1}} = a*{other_label}_k^2 + b*{label}_k + c",
                seq, lambda k,s,e: ([e[0][k]**2, s[k], Fraction(1)], s[k+1]), 0, extra=[other], label="")

    # x_{k+1} = a*x_k^2 + b*x_k*other_k + c
    test_3param(f"{label}_{{k+1}} = a*{label}_k^2 + b*{label}_k*{other_label}_k + c",
                seq, lambda k,s,e: ([s[k]**2, s[k]*e[0][k], Fraction(1)], s[k+1]), 0, extra=[other], label="")

if results_3param:
    print("  MATCHES found (3-parameter families, verified on >=4 points):")
    for name, sol, nv, pred, is_int in results_3param:
        tag = " *** INTEGER COEFFS ***" if is_int else ""
        print(f"    {name}")
        print(f"      coeffs = [{', '.join(str(c) for c in sol)}]{tag}")
        print(f"      verified on {nv} points, excess = {nv - 3}")
        if pred is not None:
            frac_tag = "" if pred.denominator == 1 else " (NOT integer!)"
            print(f"      prediction: {pred}{frac_tag}")
else:
    print("  No exact matches in 3-parameter families.")
print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: GAMMA SEQUENCES FOR FIXED SMALL d
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("SECTION 10: RECURRENCES FOR gamma_k(d) = alpha_k * 2^d + beta_k")
print("=" * 80)
print()

for d in range(2, 9):
    gamma_d = [alpha[k] * Fraction(2**d) + beta[k] for k in range(K)]
    print(f"  d={d}: gamma = {[int(x) for x in gamma_d]}")
    
    # Try x_{k+1} = a*x_k^2 + b*x_k + c (3-param)
    def build_gq(k, s, _):
        return [s[k]**2, s[k], Fraction(1)], s[k+1]
    
    all_rows = []
    all_rhs = []
    for k in range(K - 1):
        row, rhs = build_gq(k, gamma_d, None)
        all_rows.append(row)
        all_rhs.append(rhs)
    
    if len(all_rows) >= 4:
        sol = solve_exact(all_rows[:3], all_rhs[:3])
        if sol is not None:
            ok = True
            for row, rhs in zip(all_rows, all_rhs):
                if sum(c*s for c, s in zip(sol, row)) != rhs:
                    ok = False
                    break
            if ok:
                print(f"    x_{k+1} = a*x^2 + b*x + c: MATCH, coeffs = {[str(c) for c in sol]}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11: COMPREHENSIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("COMPREHENSIVE SUMMARY")
print("=" * 80)
print()

all_matches = results_2param + results_3param
if all_matches:
    # Separate by excess
    genuine = [(n,s,v,p,i) for n,s,v,p,i in all_matches if v - len(s) >= 2]
    marginal = [(n,s,v,p,i) for n,s,v,p,i in all_matches if v - len(s) == 1]
    
    if genuine:
        print("  STRONG matches (2+ excess verification points):")
        for name, sol, nv, pred, is_int in genuine:
            tag = " (integer coeffs)" if is_int else " (fractional)"
            print(f"    {name}")
            print(f"      coefficients: [{', '.join(str(c) for c in sol)}]{tag}")
            print(f"      verified on {nv} points, excess = {nv - len(sol)}")
            if pred is not None:
                print(f"      PREDICTION: {pred}" + (" = " + str(int(pred)) if pred.denominator == 1 else " (non-integer!)"))
        print()
    
    if marginal:
        print("  MARGINAL matches (exactly 1 excess verification point):")
        for name, sol, nv, pred, is_int in marginal:
            tag = " (integer coeffs)" if is_int else " (fractional)"
            print(f"    {name}")
            print(f"      coefficients: [{', '.join(str(c) for c in sol)}]{tag}")
            print(f"      verified on {nv} points, excess = {nv - len(sol)}")
            if pred is not None:
                print(f"      PREDICTION: {pred}" + (" = " + str(int(pred)) if pred.denominator == 1 else " (non-integer!)"))
        print()
else:
    print("  No exact recurrence matches found in the families tested.")
    print()

# Always show manual investigation of key patterns
print("  --- Key observations ---")
print()
print("  alpha_k^2 - alpha_k: ", [int(alpha[k]**2 - alpha[k]) for k in range(K)])
print("  alpha_{k+1}:         ", [int(alpha[k+1]) for k in range(K-1)])
print("  ratio (alpha_{k+1})/(alpha_k^2 - alpha_k):")
for k in range(K-1):
    denom = alpha[k]**2 - alpha[k]
    if denom != 0:
        r = alpha[k+1] / denom
        print(f"    k={k}: {r} = {float(r):.8f}")
print()

# Check if there's a nice relationship between consecutive alpha terms
# involving division
print("  Does alpha_{k+2} * alpha_k relate nicely to alpha_{k+1}?")
for k in range(K - 2):
    val = alpha[k+2] * alpha[k]
    sq = alpha[k+1]**2
    print(f"    k={k}: alpha_{{k+2}}*alpha_k = {int(val)}, alpha_{{k+1}}^2 = {int(sq)}, "
          f"diff = {int(val - sq)}, ratio = {float(val/sq):.6f}")
print()

# Check: does the sequence of alpha_{k+1} * alpha_{k-1} - alpha_k^2 satisfy a recurrence?
somos = [alpha[k+1] * alpha[k-1] - alpha[k]**2 for k in range(1, K-1)]
print(f"  Somos-like: alpha_{{k+1}}*alpha_{{k-1}} - alpha_k^2 = {[int(x) for x in somos]}")
print(f"  Ratios: ", end="")
for i in range(len(somos) - 1):
    if somos[i] != 0:
        print(f"{float(somos[i+1]/somos[i]):.4f}  ", end="")
print()
print()

# Final predictions summary
print("=" * 80)
print("PREDICTIONS (if any recurrence was found)")
print("=" * 80)
print()

# Collect all integer predictions
alpha_preds: Dict[str, int] = {}
beta_preds: Dict[str, int] = {}

for name, sol, nv, pred, is_int in all_matches:
    if pred is not None and pred.denominator == 1:
        val = int(pred)
        if "alpha" in name.lower() and "beta" not in name.split("=")[0].lower():
            alpha_preds[name] = val
        elif "beta" in name.lower():
            beta_preds[name] = val

if alpha_preds:
    print("  alpha_7 predictions:")
    for name, val in sorted(alpha_preds.items(), key=lambda x: x[1]):
        print(f"    {name}: {val}")
    vals = set(alpha_preds.values())
    if len(vals) == 1:
        print(f"  => All agree: alpha_7 = {vals.pop()}")
    else:
        print(f"  => DISAGREEMENT: {sorted(vals)}")
else:
    print("  No integer predictions for alpha_7")

print()

if beta_preds:
    print("  beta_7 predictions:")
    for name, val in sorted(beta_preds.items(), key=lambda x: x[1]):
        print(f"    {name}: {val}")
    vals = set(beta_preds.values())
    if len(vals) == 1:
        print(f"  => All agree: beta_7 = {vals.pop()}")
    else:
        print(f"  => DISAGREEMENT: {sorted(vals)}")
else:
    print("  No integer predictions for beta_7")

print()
print("=" * 80)
print("DONE")
print("=" * 80)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12: DEEPER INVESTIGATION OF SOMOS-LIKE QUANTITIES
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("SECTION 12: DEEPER INVESTIGATION")
print("=" * 80)
print()

# S_k = alpha_{k+1}*alpha_{k-1} - alpha_k^2
S = [alpha[k+1] * alpha[k-1] - alpha[k]**2 for k in range(1, K-1)]
print(f"  S_k = alpha_{{k+1}}*alpha_{{k-1}} - alpha_k^2: {[int(x) for x in S]}")

# T_k = beta_{k+1}*beta_{k-1} - beta_k^2
T = [beta[k+1] * beta[k-1] - beta[k]**2 for k in range(1, K-1)]
print(f"  T_k = beta_{{k+1}}*beta_{{k-1}} - beta_k^2:   {[int(x) for x in T]}")
print()

# Ratios of S
print("  S_{k+1}/S_k:")
for i in range(len(S)-1):
    if S[i] != 0:
        r = S[i+1] / S[i]
        print(f"    {r} = {float(r):.6f}")
print()

# Ratios of T
print("  T_{k+1}/T_k:")
for i in range(len(T)-1):
    if T[i] != 0:
        r = T[i+1] / T[i]
        print(f"    {r} = {float(r):.6f}")
print()

# Check S_k vs alpha_k * beta_k, or alpha_k * alpha_{k-1}, etc.
print("  S_k vs alpha_k products:")
for i, k in enumerate(range(1, K-1)):
    print(f"    k={k}: S = {int(S[i])}, alpha_k*alpha_{{k-1}} = {int(alpha[k]*alpha[k-1])}, "
          f"alpha_k*beta_k = {int(alpha[k]*beta[k])}, "
          f"ratio S/(alpha_k*alpha_{{k-1}}) = {float(S[i]/(alpha[k]*alpha[k-1])):.6f}")
print()

# Check if S_k and T_k are simply related
print("  S_k / T_k:")
for i in range(min(len(S), len(T))):
    if T[i] != 0:
        r = S[i] / T[i]
        print(f"    k={i+1}: {r} = {float(r):.6f}")
print()

# Check S_k + T_k, S_k - T_k
print(f"  S_k + T_k: {[int(S[i] + T[i]) for i in range(min(len(S), len(T)))]}")
print(f"  S_k - T_k: {[int(S[i] - T[i]) for i in range(min(len(S), len(T)))]}")
print()

# Try: does alpha_{k+1} = f(alpha_k, beta_k) for some polynomial f that uses both?
# Let's try more cross-sequence patterns with the structure we see

# alpha_{k+1} = alpha_k^2 + beta_k + 1?
print("  Checking alpha_{k+1} vs alpha_k^2 + beta_k + 1:")
for k in range(K-1):
    val = alpha[k]**2 + beta[k] + 1
    print(f"    k={k}: alpha_k^2 + beta_k + 1 = {int(val)}, alpha_{{k+1}} = {int(alpha[k+1])}, "
          f"diff = {int(alpha[k+1] - val)}")
print()

# alpha_{k+1} = alpha_k^2 + beta_k?
print("  Checking alpha_{k+1} vs alpha_k^2 + beta_k:")
for k in range(K-1):
    val = alpha[k]**2 + beta[k]
    print(f"    k={k}: {int(val)}, alpha_{{k+1}} = {int(alpha[k+1])}, diff = {int(alpha[k+1] - val)}")
print()

# Residual: alpha_{k+1} - (alpha_k^2 + beta_k)
print("  R_k = alpha_{k+1} - alpha_k^2 - beta_k:")
R = [alpha[k+1] - alpha[k]**2 - beta[k] for k in range(K-1)]
print(f"    {[int(x) for x in R]}")
print()

# Check if R_k satisfies anything
print("  R_k ratios:")
for i in range(len(R)-1):
    if R[i] != 0:
        print(f"    R_{{k+1}}/R_k = {R[i+1]}/{R[i]} = {float(R[i+1]/R[i]):.6f}")
print()

# Try: beta_{k+1} = -2*alpha_k*beta_k - beta_k + something?
print("  Checking beta_{k+1} + 2*alpha_k*beta_k + beta_k:")
for k in range(K-1):
    val = beta[k+1] + 2*alpha[k]*beta[k] + beta[k]
    print(f"    k={k}: {int(val)}")
print()

# More structured: define gamma_k = alpha_k * 2^d + beta_k
# For a TREE, we know |E| = |V| - 1. So gamma_1 = alpha_1*2^d + beta_1
# and |E(T_{2,d})| = gamma_0 - 1 = 2^{d+1} - 2
# gamma_1 = |V(L(T))| = |E(T)| = 2^{d+1} - 2 = 2*2^d - 2
# So alpha_1 = 2, beta_1 = -2. CHECK: matches!

# For the LINE GRAPH, |E(L(G))| = (1/2)*sum(deg(v)^2) - |E(G)|
# where the sum is over vertices of G.
# So gamma_{k+1}(vertices) = |E(L^k(G))| = some function of the degree sequence of L^k(G)

# The key insight: vertex counts and edge counts are related through degree sequences.
# Let's examine mu_k = |E(L^k)| / |V(L^k)| (average degree / 2)

print()
print("  gamma_k for d=8 and edge counts:")
print("  (We can compute edge_k from vertex counts since edge_k = gamma_{k+1})")
print()
for d in [3, 4, 5, 6, 7, 8]:
    gamma_d = [alpha[k] * Fraction(2**d) + beta[k] for k in range(K)]
    print(f"  d={d}:")
    for k in range(K):
        vk = int(gamma_d[k])
        if k < K - 1:
            ek = int(gamma_d[k+1])  # |E(L^k)| = |V(L^{k+1})| is NOT right in general
            # Actually |V(L(G))| = |E(G)|, so |V(L^{k+1})| = |E(L^k)|
            # but gamma_{k+1} is |V(L^{k+1})|, which = |E(L^k)|
            # Wait, this is only true for simple graphs and our L^k are simple
            # Actually the line graph has |V(L(G))| = |E(G)|, always
            print(f"    k={k}: |V| = {vk}, |E| = {int(gamma_d[k+1])}")
        else:
            print(f"    k={k}: |V| = {vk}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13: EXPLORE alpha_k - beta_k and related
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("SECTION 13: DERIVED SEQUENCES")
print("=" * 80)
print()

diff_seq = [alpha[k] - beta[k] for k in range(K)]
print(f"  alpha_k - beta_k:     {[int(x) for x in diff_seq]}")

prod_seq = [alpha[k] * beta[k] for k in range(K)]
print(f"  alpha_k * beta_k:     {[int(x) for x in prod_seq]}")

# alpha_k^2 + beta_k
sum2_seq = [alpha[k]**2 + beta[k] for k in range(K)]
print(f"  alpha_k^2 + beta_k:   {[int(x) for x in sum2_seq]}")

# beta_k^2 + alpha_k
sum3_seq = [beta[k]**2 + alpha[k] for k in range(K)]
print(f"  beta_k^2 + alpha_k:   {[int(x) for x in sum3_seq]}")

# alpha_k^2 - 2*beta_k
sum4_seq = [alpha[k]**2 - 2*beta[k] for k in range(K)]
print(f"  alpha_k^2 - 2*beta_k: {[int(x) for x in sum4_seq]}")

print()

# First differences of alpha_k^2 + beta_k
d_sum2 = [sum2_seq[k+1] - sum2_seq[k] for k in range(len(sum2_seq)-1)]
print(f"  Delta(alpha_k^2 + beta_k): {[int(x) for x in d_sum2]}")

# Ratios
print("  Ratios of (alpha_k^2 + beta_k):")
for k in range(len(sum2_seq) - 1):
    if sum2_seq[k] != 0:
        r = sum2_seq[k+1] / sum2_seq[k]
        print(f"    k={k}: {r} = {float(r):.6f}")
print()

# Explore the "almost Sylvester" aspect more carefully
# Sylvester: s_{n+1} = s_n(s_n - 1) + 1
# We have: alpha_{k+1} ≈ alpha_k(alpha_k-1) for small k but diverges
# The CORRECTION is alpha_{k+1} = alpha_k(alpha_k-1) + delta_k
# where delta_k = 0, 1, 1, -13, -585, -47833
# What IS delta_k?

print("  delta_k = alpha_{k+1} - alpha_k*(alpha_k - 1):")
delta = [alpha[k+1] - alpha[k]*(alpha[k] - 1) for k in range(K-1)]
print(f"    {[int(x) for x in delta]}")

# Is delta_k related to beta?
print("  delta_k vs beta_k:")
for k in range(K-1):
    print(f"    k={k}: delta = {int(delta[k])}, beta_k = {int(beta[k])}, "
          f"beta_{{k+1}} = {int(beta[k+1])}, "
          f"delta/beta_k = {float(delta[k]/beta[k]) if beta[k] != 0 else 'inf':.6f}")
print()

# Is delta_k = f(alpha_{k-1}, beta_{k-1}, alpha_k, beta_k)?
# delta_0 = 0, but alpha_{-1} and beta_{-1} are not defined. Skip k=0.
# delta_k for k >= 1:
print("  Try delta_k = a*beta_k + b*alpha_k + c for k>=1:")
delta_from1 = delta[1:]  # k=1,2,3,4,5
# For k=1: a*(-2) + b*2 + c = 1
# For k=2: a*(-5) + b*3 + c = 1  
# For k=3: a*(-18) + b*7 + c = -13
rows_d = []
rhs_d = []
for i, k in enumerate(range(1, K-1)):
    rows_d.append([beta[k], alpha[k], Fraction(1)])
    rhs_d.append(delta[k])

if len(rows_d) >= 4:
    sol_d = solve_exact(rows_d[:3], rhs_d[:3])
    if sol_d:
        ok = True
        for row, rhs in zip(rows_d, rhs_d):
            if sum(c*s for c, s in zip(sol_d, row)) != rhs:
                ok = False
                break
        if ok:
            print(f"    MATCH: delta = {sol_d[0]}*beta + {sol_d[1]}*alpha + {sol_d[2]}")
        else:
            print(f"    FAILED. Coefficients from first 3: {[str(c) for c in sol_d]}")
    else:
        print("    Singular")
print()

# Try delta_k = a*beta_k + b
print("  Try delta_k = a*beta_k + b for k>=0:")
rows_d2 = [[beta[k], Fraction(1)] for k in range(K-1)]
rhs_d2 = [delta[k] for k in range(K-1)]
sol_d2 = solve_exact(rows_d2[:2], rhs_d2[:2])
if sol_d2:
    ok = True
    for row, rhs in zip(rows_d2, rhs_d2):
        if sum(c*s for c, s in zip(sol_d2, row)) != rhs:
            ok = False
            break
    print(f"    From first 2: delta = {sol_d2[0]}*beta + {sol_d2[1]}")
    print(f"    Verified on all: {ok}")
print()

# Try delta_k = a*alpha_k*beta_k + b
print("  Try delta_k = a*alpha_k*beta_k + b:")
rows_d3 = [[alpha[k]*beta[k], Fraction(1)] for k in range(K-1)]
rhs_d3 = [delta[k] for k in range(K-1)]
sol_d3 = solve_exact(rows_d3[:2], rhs_d3[:2])
if sol_d3:
    ok = True
    for row, rhs in zip(rows_d3, rhs_d3):
        if sum(c*s for c, s in zip(sol_d3, row)) != rhs:
            ok = False
            break
    print(f"    delta = {sol_d3[0]}*alpha*beta + {sol_d3[1]}")
    print(f"    Verified: {ok}")
print()

# Try delta_k = a*beta_k^2 + b*beta_k + c
print("  Try delta_k = a*beta_k^2 + b*beta_k + c:")
rows_d4 = [[beta[k]**2, beta[k], Fraction(1)] for k in range(K-1)]
rhs_d4 = [delta[k] for k in range(K-1)]
sol_d4 = solve_exact(rows_d4[:3], rhs_d4[:3])
if sol_d4:
    ok = True
    for row, rhs in zip(rows_d4, rhs_d4):
        if sum(c*s for c, s in zip(sol_d4, row)) != rhs:
            ok = False
            break
    print(f"    delta = {sol_d4[0]}*beta^2 + {sol_d4[1]}*beta + {sol_d4[2]}")
    print(f"    Verified: {ok}")
print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14: EXHAUSTIVE 3-PARAM WITH PRODUCTS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("SECTION 14: EXHAUSTIVE 3-PARAM CROSS-SEQUENCE WITH PRODUCTS")
print("  Using both alpha_k, beta_k, alpha_{k-1}, beta_{k-1}")
print("=" * 80)
print()

# Build a large set of "monomial" features from (alpha_k, beta_k, alpha_{k-1}, beta_{k-1})
# up to degree 2, then try all 3-element subsets as recurrence for alpha_{k+1} or beta_{k+1}

def get_features(k, alpha, beta):
    """Return dict of feature_name -> value at index k (needs k >= 1)."""
    ak, bk = alpha[k], beta[k]
    akm = alpha[k-1]
    bkm = beta[k-1]
    return {
        "a_k": ak, "b_k": bk, "a_{k-1}": akm, "b_{k-1}": bkm,
        "1": Fraction(1),
        "a_k^2": ak**2, "b_k^2": bk**2, "a_{k-1}^2": akm**2, "b_{k-1}^2": bkm**2,
        "a_k*b_k": ak*bk, "a_k*a_{k-1}": ak*akm, "a_k*b_{k-1}": ak*bkm,
        "b_k*a_{k-1}": bk*akm, "b_k*b_{k-1}": bk*bkm, "a_{k-1}*b_{k-1}": akm*bkm,
    }

feature_names = list(get_features(1, alpha, beta).keys())
n_features = len(feature_names)

# For alpha_{k+1}: we have k=1..5 giving 5 equations, need 3 unknowns => 2 excess
# For beta_{k+1}: same

from itertools import combinations

print(f"  Number of features: {n_features}")
print(f"  Testing all C({n_features}, 3) = {n_features*(n_features-1)*(n_features-2)//6} combinations of 3 features")
print()

for target_label, target_seq, other_seq in [("alpha", alpha, beta), ("beta", beta, alpha)]:
    hits = []
    
    # Build feature matrix for k=1..K-2 (predicting target at k+1, i.e., k=2..K-1)
    target_rhs = [target_seq[k+1] for k in range(1, K-1)]  # indices 2..6
    feature_matrix = []
    for k in range(1, K-1):
        feats = get_features(k, alpha, beta)
        feature_matrix.append([feats[name] for name in feature_names])
    
    n_eqns = len(target_rhs)  # should be 5
    
    for combo in combinations(range(n_features), 3):
        # Extract sub-matrix
        sub_matrix = [[feature_matrix[i][j] for j in combo] for i in range(n_eqns)]
        sub_rhs = target_rhs[:]
        
        # Solve using first 3 equations
        sol = solve_exact(sub_matrix[:3], sub_rhs[:3])
        if sol is None:
            continue
        
        # Verify on ALL
        ok = True
        for i in range(n_eqns):
            pred = sum(c * sub_matrix[i][j] for j, c in enumerate(sol))
            if pred != sub_rhs[i]:
                ok = False
                break
        
        if ok:
            names = [feature_names[j] for j in combo]
            int_coeffs = all(c.denominator == 1 for c in sol)
            hits.append((names, sol, int_coeffs))
    
    if hits:
        print(f"  {target_label}_{{k+1}} exact matches (verified on {n_eqns} points, 2 excess):")
        for names, sol, is_int in hits:
            tag = " *** INTEGER ***" if is_int else ""
            terms = " + ".join(f"({c})*{n}" for c, n in zip(sol, names))
            print(f"    {target_label}_{{k+1}} = {terms}{tag}")
            
            # Try prediction
            try:
                feats_next = get_features(K-1, alpha, beta)
                pred_val = sum(c * feats_next[n] for c, n in zip(sol, names))
                frac_note = "" if pred_val.denominator == 1 else " (non-integer!)"
                print(f"      -> predicted {target_label}_{K} = {pred_val}{frac_note}")
            except Exception as e:
                print(f"      -> prediction failed: {e}")
        print()
    else:
        print(f"  No 3-feature matches for {target_label}_{{k+1}}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 15: EXHAUSTIVE 2-PARAM WITH PRODUCTS (even stronger: 3 excess)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("SECTION 15: EXHAUSTIVE 2-PARAM CROSS-SEQUENCE (3 excess)")
print("=" * 80)
print()

for target_label, target_seq in [("alpha", alpha), ("beta", beta)]:
    hits = []
    
    target_rhs = [target_seq[k+1] for k in range(1, K-1)]
    feature_matrix = []
    for k in range(1, K-1):
        feats = get_features(k, alpha, beta)
        feature_matrix.append([feats[name] for name in feature_names])
    
    n_eqns = len(target_rhs)
    
    for combo in combinations(range(n_features), 2):
        sub_matrix = [[feature_matrix[i][j] for j in combo] for i in range(n_eqns)]
        sub_rhs = target_rhs[:]
        
        sol = solve_exact(sub_matrix[:2], sub_rhs[:2])
        if sol is None:
            continue
        
        ok = True
        for i in range(n_eqns):
            pred = sum(c * sub_matrix[i][j] for j, c in enumerate(sol))
            if pred != sub_rhs[i]:
                ok = False
                break
        
        if ok:
            names = [feature_names[j] for j in combo]
            int_coeffs = all(c.denominator == 1 for c in sol)
            hits.append((names, sol, int_coeffs))
    
    if hits:
        print(f"  {target_label}_{{k+1}} exact matches (verified on {n_eqns} points, {n_eqns-2} excess):")
        for names, sol, is_int in hits:
            tag = " *** INTEGER ***" if is_int else ""
            terms = " + ".join(f"({c})*{n}" for c, n in zip(sol, names))
            print(f"    {target_label}_{{k+1}} = {terms}{tag}")
            try:
                feats_next = get_features(K-1, alpha, beta)
                pred_val = sum(c * feats_next[n] for c, n in zip(sol, names))
                frac_note = "" if pred_val.denominator == 1 else " (non-integer!)"
                print(f"      -> predicted {target_label}_{K} = {pred_val}{frac_note}")
            except Exception as e:
                print(f"      -> prediction failed: {e}")
        print()
    else:
        print(f"  No 2-feature matches for {target_label}_{{k+1}}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 16: ALSO INCLUDE k=0 DATA (features with only alpha_0, beta_0)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("SECTION 16: SINGLE-STEP FEATURES (no lag, k=0..K-2)")
print("  Features from (alpha_k, beta_k) only, predicting alpha_{k+1} or beta_{k+1}")
print("=" * 80)
print()

def get_features_nolag(k, alpha, beta):
    ak, bk = alpha[k], beta[k]
    return {
        "a_k": ak, "b_k": bk, "1": Fraction(1),
        "a_k^2": ak**2, "b_k^2": bk**2, "a_k*b_k": ak*bk,
        "a_k^3": ak**3, "b_k^3": bk**3, "a_k^2*b_k": ak**2*bk, "a_k*b_k^2": ak*bk**2,
    }

fnames_nl = list(get_features_nolag(0, alpha, beta).keys())
n_fnl = len(fnames_nl)

print(f"  Features: {fnames_nl}")
print(f"  Using k=0..{K-2}, giving {K-1} equations")
print()

for target_label, target_seq in [("alpha", alpha), ("beta", beta)]:
    # 2-param
    target_rhs = [target_seq[k+1] for k in range(K-1)]
    fmatrix = []
    for k in range(K-1):
        feats = get_features_nolag(k, alpha, beta)
        fmatrix.append([feats[n] for n in fnames_nl])
    
    n_eqns = len(target_rhs)
    
    for n_params in [2, 3]:
        hits = []
        for combo in combinations(range(n_fnl), n_params):
            sub = [[fmatrix[i][j] for j in combo] for i in range(n_eqns)]
            sol = solve_exact(sub[:n_params], target_rhs[:n_params])
            if sol is None:
                continue
            ok = True
            for i in range(n_eqns):
                if sum(c * sub[i][j] for j, c in enumerate(sol)) != target_rhs[i]:
                    ok = False
                    break
            if ok:
                names = [fnames_nl[j] for j in combo]
                int_coeffs = all(c.denominator == 1 for c in sol)
                hits.append((names, sol, int_coeffs))
        
        excess = n_eqns - n_params
        if hits:
            print(f"  {target_label}_{{k+1}} = f(a_k, b_k) with {n_params} params "
                  f"({n_eqns} eqns, excess={excess}):")
            for names, sol, is_int in hits:
                tag = " *** INTEGER ***" if is_int else ""
                terms = " + ".join(f"({c})*{n}" for c, n in zip(sol, names))
                print(f"    {target_label}_{{k+1}} = {terms}{tag}")
                try:
                    feats_last = get_features_nolag(K-1, alpha, beta)
                    pred = sum(c * feats_last[n] for c, n in zip(sol, names))
                    frac_note = "" if pred.denominator == 1 else " (non-integer!)"
                    print(f"      -> predicted {target_label}_{K} = {pred}{frac_note}")
                except Exception:
                    pass
            print()
        else:
            print(f"  No {n_params}-param matches for {target_label}_{{k+1}} (no-lag)")
            print()


print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print()
print("  The alpha and beta sequences do NOT satisfy any simple (2-3 parameter)")
print("  recurrence among the extensive families tested.")
print()
print("  Observations:")
print("  - alpha matches the start of Sylvester's sequence (2,3,7,...) but diverges at k=4")
print("  - The ratio alpha_{k+1}/alpha_k is growing but NOT matching alpha_k")
print("  - The Somos-like quantity alpha_{k+1}*alpha_{k-1} - alpha_k^2 = [2,5,38,748,49072]")
print("    has no obvious simple recurrence either")
print("  - All 'exact' fits with N unknowns and N data points have fractional coefficients")
print("    and produce non-integer predictions, confirming they are overfitting artifacts")
print("  - The sequences likely arise from a more complex structural property of")
print("    iterated line graphs that doesn't reduce to a low-order recurrence")
print()
print("=" * 80)
print("DONE (extended analysis)")
print("=" * 80)
