#!/usr/bin/env python3
"""
Enumerate all unlabeled trees with 1..6 edges (2..7 vertices),
group by degree sequence, compute embedding polynomials via falling
factorials, form coefficient matrix, and compute its rank.

Embedding polynomial for a tree rooted at vertex 0:
    emb(tau, r) = r^{(fall d_root)} * prod_{v != root} (r-1)^{(fall c_v)}
where
    d_root  = degree of the root,
    c_v     = deg(v) - 1  for every non-root vertex v,
    x^{(fall n)} = x(x-1)...(x-n+1)   (falling factorial).
"""

import networkx as nx
from sympy import Symbol, Rational, Matrix, ff, expand, Poly
from collections import defaultdict

r = Symbol("r")


def falling_factorial_poly(base_expr, n):
    """Return the symbolic falling factorial base_expr^{(fall n)} as a sympy expression."""
    # ff(x, n) from sympy gives the falling factorial x*(x-1)*...*(x-n+1)
    return ff(base_expr, n)


def degree_sequence(G):
    """Sorted tuple of vertex degrees (non-increasing)."""
    return tuple(sorted((d for _, d in G.degree()), reverse=True))


def embedding_polynomial(T, root=0):
    """
    Compute the embedding polynomial for tree T rooted at `root`.
    Returns a sympy expression in r, fully expanded.
    """
    d_root = T.degree(root)
    poly_expr = falling_factorial_poly(r, d_root)

    for v in T.nodes():
        if v == root:
            continue
        c_v = T.degree(v) - 1  # children count (degree minus parent edge)
        if c_v > 0:
            poly_expr *= falling_factorial_poly(r - 1, c_v)
        # c_v == 0 contributes factor 1 (leaf)

    return expand(poly_expr)


def main():
    # ------------------------------------------------------------------
    # 1. Enumerate all unlabeled trees with 2..7 vertices (1..6 edges)
    # ------------------------------------------------------------------
    all_trees = []
    for n_verts in range(2, 8):
        for T in nx.nonisomorphic_trees(n_verts):
            all_trees.append(T)
    print(f"Total trees enumerated: {len(all_trees)}")
    expected_counts = [1, 1, 2, 3, 6, 11]
    for i, nv in enumerate(range(2, 8)):
        count = sum(1 for _ in nx.nonisomorphic_trees(nv))
        print(f"  {nv} vertices ({nv-1} edges): {count} trees")
    print()

    # ------------------------------------------------------------------
    # 2. Group by degree sequence
    # ------------------------------------------------------------------
    groups = defaultdict(list)
    for T in all_trees:
        ds = degree_sequence(T)
        groups[ds].append(T)

    sorted_keys = sorted(groups.keys(), key=lambda ds: (sum(ds), ds))
    print(f"Number of degree-sequence classes: {len(sorted_keys)}")
    print()

    for ds in sorted_keys:
        trees = groups[ds]
        print(f"  Degree seq {ds}: {len(trees)} tree(s)")
    print()

    # ------------------------------------------------------------------
    # 3. For each class, pick representative, compute embedding polynomial
    # ------------------------------------------------------------------
    class_data = []

    for ds in sorted_keys:
        T = groups[ds][0]  # pick first as representative
        edges = list(T.edges())
        root = 0
        emb = embedding_polynomial(T, root=root)

        class_data.append({
            "degree_seq": ds,
            "num_trees": len(groups[ds]),
            "edges": edges,
            "root": root,
            "emb_poly": emb,
        })

    print("=" * 80)
    print("Embedding polynomials per degree-sequence class")
    print("=" * 80)
    for i, cd in enumerate(class_data):
        ds = cd["degree_seq"]
        print(f"\nClass {i+1}: degree sequence = {ds}  ({cd['num_trees']} tree(s))")
        print(f"  Representative edges: {cd['edges']}")
        print(f"  Rooted at vertex {cd['root']}")
        print(f"  emb(tau, r) = {cd['emb_poly']}")

    print()

    # ------------------------------------------------------------------
    # 4. Form the coefficient matrix
    # ------------------------------------------------------------------
    max_deg = 0
    polys = []
    for cd in class_data:
        p = Poly(cd["emb_poly"], r)
        polys.append(p)
        if p.degree() > max_deg:
            max_deg = p.degree()

    n_rows = len(class_data)
    n_cols = max_deg + 1  # coefficients for r^0, r^1, ..., r^max_deg

    # Build matrix with exact rational entries
    mat_entries = []
    for p in polys:
        coeffs_dict = p.as_dict()
        row = []
        for j in range(n_cols):
            row.append(Rational(coeffs_dict.get((j,), 0)))
        mat_entries.append(row)

    M = Matrix(mat_entries)

    print("=" * 80)
    print(f"Coefficient matrix ({n_rows} classes x {n_cols} columns, r^0 .. r^{max_deg})")
    print("=" * 80)

    # Header
    header = "".join(f"{'r^'+str(j):>10}" for j in range(n_cols))
    print(f"{'Class':>8}{header}")
    for i, cd in enumerate(class_data):
        row_str = "".join(f"{str(mat_entries[i][j]):>10}" for j in range(n_cols))
        print(f"{i+1:>8}{row_str}")

    print()

    # ------------------------------------------------------------------
    # 5. Compute rank
    # ------------------------------------------------------------------
    rank = M.rank()
    print(f"Rank of the coefficient matrix: {rank}")
    print(f"  (matrix size: {n_rows} rows x {n_cols} columns)")
    print()

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total trees (1..6 edges): {len(all_trees)}")
    print(f"Degree-sequence classes:   {len(sorted_keys)}")
    print(f"Coefficient matrix size:   {n_rows} x {n_cols}")
    print(f"Matrix rank:               {rank}")


if __name__ == "__main__":
    main()
