#!/usr/bin/env python3
"""
Compute the rank of the coefficient matrix restricted to acyclic (tree)
subgraph types from the dumbbell/chorded-C6 example.

Uses the k=8 coefficient data computed previously.
"""

from fractions import Fraction
from math import gcd
from functools import reduce


# ============================================================
#  Coefficient data from k=8 computation (acyclic types only)
# ============================================================

# Format: (label, n_edges, [c1, c2, c3, c4, c5, c6, c7, c8])

tree_types = [
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

N = len(tree_types)  # 10
K = 8                # grades

labels = [t[0] for t in tree_types]
cvecs = [t[2] for t in tree_types]

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
            print(f" {'·':>8s}", end="")
    print()

# Build exact rational matrix M (K x N)
M = []
for k in range(K):
    row = [Fraction(cvecs[j][k]) for j in range(N)]
    M.append(row)

# Row reduce
mat = [row[:] for row in M]
n_rows = K
n_cols = N

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

print(f"\n  Rank: {rank}")
print(f"  Null space dimension: {N - rank}")

print(f"\n  Pivot types ({rank}):")
for pc in pivot_cols:
    print(f"    {labels[pc]}")

if free_cols:
    print(f"\n  Free types ({len(free_cols)}):")
    for fc in free_cols:
        print(f"    {labels[fc]}")

    # Extract null vectors
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

# Also check rank growth grade by grade
print(f"\n  {'='*60}")
print(f"  Rank growth by grade")
print(f"  {'='*60}")

for max_grade in range(1, K + 1):
    M_sub = []
    for k in range(max_grade):
        row = [Fraction(cvecs[j][k]) for j in range(N)]
        M_sub.append(row)

    # Row reduce
    mat_sub = [row[:] for row in M_sub]
    pivots_sub = []
    cr = 0
    for col in range(N):
        piv = None
        for row in range(cr, max_grade):
            if mat_sub[row][col] != 0:
                piv = row
                break
        if piv is None:
            continue
        mat_sub[cr], mat_sub[piv] = mat_sub[piv], mat_sub[cr]
        pivots_sub.append(col)
        sc = mat_sub[cr][col]
        for j in range(N):
            mat_sub[cr][j] /= sc
        for row in range(max_grade):
            if row == cr:
                continue
            fac = mat_sub[row][col]
            if fac != 0:
                for j in range(N):
                    mat_sub[row][j] -= fac * mat_sub[cr][j]
        cr += 1

    r = len(pivots_sub)
    new_pivot = labels[pivots_sub[-1]] if pivots_sub and r > (len(pivots_sub) - 1) else "—"
    pivot_names = [labels[p] for p in pivots_sub]
    print(f"    K={max_grade}: rank={r}, pivots={pivot_names}")