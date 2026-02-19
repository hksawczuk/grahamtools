"""
Among the 24 trees with 1..6 edges, find all groups with linearly dependent
emb_1 polynomials. Identify proportional pairs, and find the null space to
characterize all linear dependencies.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Set, Tuple

import networkx as nx
import sympy
from sympy import Matrix, Poly, Rational, Symbol, expand, symbols

a, b = symbols('a b')


def falling_factorial(x, n: int):
    result = sympy.Integer(1)
    for i in range(n):
        result *= (x - i)
    return expand(result)


def bipartition(G, root=0):
    color = {root: 0}
    queue = [root]
    while queue:
        v = queue.pop(0)
        for u in G.neighbors(v):
            if u not in color:
                color[u] = 1 - color[v]
                queue.append(u)
    A = {v for v, c in color.items() if c == 0}
    B = {v for v, c in color.items() if c == 1}
    return A, B


def root_tree(G, root=0):
    children = defaultdict(list)
    visited = {root}
    queue = [root]
    while queue:
        v = queue.pop(0)
        for u in sorted(G.neighbors(v)):
            if u not in visited:
                visited.add(u)
                children[v].append(u)
                queue.append(u)
    return children


def compute_emb1(G, root, A, B):
    children_map = root_tree(G, root)
    d_root = G.degree(root)
    result = falling_factorial(a, d_root)
    for v in G.nodes():
        if v == root:
            continue
        c_v = len(children_map[v])
        if c_v == 0:
            continue
        if v in A:
            result *= falling_factorial(a - 1, c_v)
        else:
            result *= falling_factorial(b - 1, c_v)
    return expand(result)


def poly_to_coeff_dict(p):
    poly = Poly(p, a, b)
    return {monom: coeff for monom, coeff in zip(poly.monoms(), poly.coeffs())}


def main():
    # Enumerate all trees
    all_trees = []
    for n in range(2, 8):
        for G in nx.nonisomorphic_trees(n):
            root = 0
            A, B = bipartition(G, root)
            edges = sorted(G.edges())
            deg_seq = tuple(sorted((G.degree(v) for v in G.nodes()), reverse=True))
            emb1 = compute_emb1(G, root, A, B)
            emb2 = expand(emb1.subs({a: b, b: a}))
            all_trees.append({
                'idx': len(all_trees) + 1,
                'G': G,
                'edges': edges,
                'deg_seq': deg_seq,
                'n': G.number_of_nodes(),
                'A_size': len(A),
                'B_size': len(B),
                'emb1': emb1,
                'emb2': emb2,
            })

    print(f"Total trees: {len(all_trees)}")

    # Collect all monomials
    all_monoms = set()
    for t in all_trees:
        all_monoms |= set(poly_to_coeff_dict(t['emb1']).keys())
    monoms_sorted = sorted(all_monoms, key=lambda m: (sum(m), -m[0]))

    monom_labels = []
    for (i, j) in monoms_sorted:
        if i == 0 and j == 0:
            monom_labels.append("1")
        else:
            parts = []
            if i > 0:
                parts.append(f"a^{i}" if i > 1 else "a")
            if j > 0:
                parts.append(f"b^{j}" if j > 1 else "b")
            monom_labels.append("".join(parts))

    print(f"Monomial basis: {monom_labels} ({len(monoms_sorted)} monomials)")

    # Build coefficient matrix
    rows = []
    for t in all_trees:
        cd = poly_to_coeff_dict(t['emb1'])
        row = [cd.get(m, 0) for m in monoms_sorted]
        rows.append(row)

    M = Matrix(rows)
    rank = M.rank()
    print(f"Coefficient matrix: {M.shape[0]} x {M.shape[1]}, rank = {rank}")
    print(f"Null space dimension: {M.shape[0] - rank} (dependencies among rows)")

    # ============================================================
    # 1. Find proportional pairs
    # ============================================================
    print()
    print("=" * 80)
    print("PROPORTIONAL PAIRS (emb_1(tree_i) = c * emb_1(tree_j))")
    print("=" * 80)

    n_prop = 0
    for i, j in combinations(range(len(all_trees)), 2):
        p_i = all_trees[i]['emb1']
        p_j = all_trees[j]['emb1']
        # Check if p_i = c * p_j for some constant c
        # Find first nonzero coeff of p_j
        cd_i = poly_to_coeff_dict(p_i)
        cd_j = poly_to_coeff_dict(p_j)
        # Find a monomial where p_j is nonzero
        ratio = None
        is_proportional = True
        all_monoms_union = set(cd_i.keys()) | set(cd_j.keys())
        for m in all_monoms_union:
            ci = cd_i.get(m, 0)
            cj = cd_j.get(m, 0)
            if cj != 0:
                r = Rational(ci, cj)
                if ratio is None:
                    ratio = r
                elif r != ratio:
                    is_proportional = False
                    break
            elif ci != 0:
                is_proportional = False
                break

        if is_proportional and ratio is not None:
            ti = all_trees[i]
            tj = all_trees[j]
            n_prop += 1
            print(f"\n  Trees #{ti['idx']} and #{tj['idx']}: emb_1(#{ti['idx']}) = {ratio} * emb_1(#{tj['idx']})")
            print(f"    #{ti['idx']}: deg_seq={ti['deg_seq']}, edges={ti['edges']}, |A|={ti['A_size']}, |B|={ti['B_size']}")
            print(f"    #{tj['idx']}: deg_seq={tj['deg_seq']}, edges={tj['edges']}, |A|={tj['A_size']}, |B|={tj['B_size']}")
            print(f"    emb_1(#{ti['idx']}) = {ti['emb1']}")
            print(f"    emb_1(#{tj['idx']}) = {tj['emb1']}")

    if n_prop == 0:
        print("  No proportional pairs found.")

    # ============================================================
    # 2. Find all linear dependencies via null space of M^T
    # ============================================================
    print()
    print("=" * 80)
    print("ALL LINEAR DEPENDENCIES (null space of the row space)")
    print("=" * 80)

    # The null space of M^T gives vectors c such that c^T M = 0,
    # i.e., sum_i c_i * row_i = 0
    # Equivalently, null space of M (as rows) = left null space
    # We want: which rows are in the same dependency?

    # Compute the left null space: vectors c with c^T M = 0
    Mt = M.T
    null_vecs = Mt.nullspace()
    print(f"\nLeft null space dimension: {len(null_vecs)}")
    print(f"(There are {len(null_vecs)} independent linear dependencies among the 24 emb_1 polynomials)")

    for idx_n, nv in enumerate(null_vecs):
        print(f"\n--- Dependency {idx_n + 1} ---")
        # Find nonzero entries
        nonzero = [(i, nv[i]) for i in range(len(all_trees)) if nv[i] != 0]
        print(f"  Involves {len(nonzero)} trees:")
        terms = []
        for i, coeff in nonzero:
            t = all_trees[i]
            sign = "+" if coeff > 0 else ""
            print(f"    {sign}{coeff} * emb_1(#{t['idx']}): deg_seq={t['deg_seq']}, "
                  f"|A|={t['A_size']},|B|={t['B_size']}, edges={t['edges']}")

        # Verify
        check = sum(nv[i] * all_trees[i]['emb1'] for i in range(len(all_trees)))
        assert expand(check) == 0, f"Dependency {idx_n+1} failed verification!"
        print(f"  Verified: sum = 0 ✓")

    # ============================================================
    # 3. Group trees that participate in the same dependencies
    # ============================================================
    print()
    print("=" * 80)
    print("DEPENDENCY STRUCTURE: Which trees are entangled?")
    print("=" * 80)

    # For each tree, list which dependencies it participates in
    tree_deps = defaultdict(set)
    for idx_n, nv in enumerate(null_vecs):
        for i in range(len(all_trees)):
            if nv[i] != 0:
                tree_deps[i].add(idx_n)

    # Find connected components via shared dependencies
    # Two trees are connected if they share a dependency
    from collections import deque
    visited_trees = set()
    components = []
    for i in range(len(all_trees)):
        if i not in tree_deps or i in visited_trees:
            continue
        # BFS
        comp = set()
        queue = deque([i])
        while queue:
            t = queue.popleft()
            if t in comp:
                continue
            comp.add(t)
            visited_trees.add(t)
            # Find all trees sharing a dependency with t
            for dep_idx in tree_deps[t]:
                for j in range(len(all_trees)):
                    if null_vecs[dep_idx][j] != 0 and j not in comp:
                        queue.append(j)
        components.append(comp)

    print(f"\n{len(components)} connected component(s) of dependent trees:")
    for ci, comp in enumerate(components):
        trees_in_comp = sorted(comp)
        print(f"\n  Component {ci+1} ({len(comp)} trees):")
        for i in trees_in_comp:
            t = all_trees[i]
            print(f"    Tree #{t['idx']}: deg_seq={t['deg_seq']}, |A|={t['A_size']},|B|={t['B_size']}, edges={t['edges']}")

    # Trees NOT involved in any dependency (linearly independent from all others)
    independent = set(range(len(all_trees))) - visited_trees
    print(f"\n  {len(independent)} tree(s) not involved in any dependency (fully independent):")
    for i in sorted(independent):
        t = all_trees[i]
        print(f"    Tree #{t['idx']}: deg_seq={t['deg_seq']}, |A|={t['A_size']},|B|={t['B_size']}")

    # ============================================================
    # 4. Analyze what distinguishes the dependent groups
    # ============================================================
    print()
    print("=" * 80)
    print("ANALYSIS: What do dependent trees have in common?")
    print("=" * 80)

    for idx_n, nv in enumerate(null_vecs):
        nonzero = [(i, nv[i]) for i in range(len(all_trees)) if nv[i] != 0]
        trees_involved = [all_trees[i] for i, _ in nonzero]

        print(f"\n--- Dependency {idx_n + 1} ---")

        # Check common properties
        edge_counts = set(t['n'] - 1 for t in trees_involved)
        deg_seqs = [t['deg_seq'] for t in trees_involved]
        a_sizes = set(t['A_size'] for t in trees_involved)
        b_sizes = set(t['B_size'] for t in trees_involved)
        n_vertices = set(t['n'] for t in trees_involved)

        print(f"  # edges: {edge_counts}")
        print(f"  # vertices: {n_vertices}")
        print(f"  |A| values: {a_sizes}")
        print(f"  |B| values: {b_sizes}")
        print(f"  Degree sequences:")
        for ds in deg_seqs:
            print(f"    {ds}")

        # Check: same number of edges?
        if len(edge_counts) == 1:
            print(f"  → All have {edge_counts.pop()} edges")
        else:
            print(f"  → MIXED edge counts!")

        # Check: same degree sequence?
        if len(set(str(ds) for ds in deg_seqs)) == 1:
            print(f"  → All have the SAME degree sequence")
        else:
            print(f"  → Different degree sequences")

    # ============================================================
    # 5. Pairwise: for trees with same emb_2, check emb_1 relation
    # ============================================================
    print()
    print("=" * 80)
    print("TREES WITH IDENTICAL emb_2 (= emb_1(b,a))")
    print("=" * 80)

    emb2_groups = defaultdict(list)
    for t in all_trees:
        emb2_groups[str(expand(t['emb2']))].append(t)

    for key, group in sorted(emb2_groups.items(), key=lambda x: (-len(x[1]), x[1][0]['idx'])):
        if len(group) > 1:
            print(f"\n  emb_2 = {group[0]['emb2']}")
            print(f"  Shared by {len(group)} trees:")
            for t in group:
                print(f"    #{t['idx']}: deg_seq={t['deg_seq']}, |A|={t['A_size']},|B|={t['B_size']}, edges={t['edges']}")
                print(f"      emb_1 = {t['emb1']}")

    # ============================================================
    # 6. For each pair of trees with same deg_seq, compare emb_1
    # ============================================================
    print()
    print("=" * 80)
    print("SAME DEGREE SEQUENCE: comparing emb_1 polynomials")
    print("=" * 80)

    ds_groups = defaultdict(list)
    for t in all_trees:
        ds_groups[t['deg_seq']].append(t)

    for ds, group in sorted(ds_groups.items(), key=lambda x: (-len(x[1]), x[1][0]['idx'])):
        if len(group) > 1:
            print(f"\n  Degree sequence: {ds} — {len(group)} trees")
            for t in group:
                print(f"    #{t['idx']}: |A|={t['A_size']},|B|={t['B_size']}, edges={t['edges']}")
                print(f"      emb_1 = {t['emb1']}")

            # Check pairwise differences
            for i, j in combinations(range(len(group)), 2):
                ti, tj = group[i], group[j]
                diff = expand(ti['emb1'] - tj['emb1'])
                print(f"    emb_1(#{ti['idx']}) - emb_1(#{tj['idx']}) = {diff}")


if __name__ == "__main__":
    main()
