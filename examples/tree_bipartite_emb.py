"""
Enumerate all unlabeled trees with 1..6 edges (2..7 vertices, 24 trees total).
For each tree τ with unique bipartition (A, B) where 0 ∈ A, compute embedding
polynomials emb_1(a,b) and emb_2(a,b) = emb_1(b,a) for embedding τ into T_{a,b}.

Then: report collision groups and matrix ranks.
"""

import networkx as nx
from sympy import symbols, expand, Poly, Matrix, Rational
from sympy import Expr
from collections import defaultdict
from itertools import chain


def falling_factorial_poly(x, n):
    """Compute x^{(fall n)} = x(x-1)...(x-n+1) as a sympy expression."""
    result = 1
    for k in range(n):
        result = result * (x - k)
    return expand(result)


def get_bipartition(T, root=0):
    """Return (A, B) where A contains root, via BFS coloring."""
    color = {}
    color[root] = 0
    queue = [root]
    while queue:
        v = queue.pop(0)
        for u in T.neighbors(v):
            if u not in color:
                color[u] = 1 - color[v]
                queue.append(u)
    A = {v for v, c in color.items() if c == 0}
    B = {v for v, c in color.items() if c == 1}
    return A, B


def root_tree(T, root=0):
    """Return dict: parent[v] and children[v] for tree T rooted at root."""
    parent = {root: None}
    children = defaultdict(list)
    queue = [root]
    while queue:
        v = queue.pop(0)
        for u in T.neighbors(v):
            if u not in parent:
                parent[u] = v
                children[v].append(u)
                queue.append(u)
    # Ensure all vertices have an entry in children
    for v in T.nodes():
        if v not in children:
            children[v] = []
    return parent, children


def compute_emb1(T, root, A, B, a, b):
    """
    Compute emb_1(τ, a, b) — mapping A→a-type, B→b-type.
    
    Root is in A:
      root contributes: a^{(fall d_root)} where d_root = deg(root) = #children(root)
      non-root v in A: (a-1)^{(fall c_v)}
      non-root v in B: (b-1)^{(fall c_v)}
    """
    parent, children_dict = root_tree(T, root)
    
    d_root = len(children_dict[root])
    result = falling_factorial_poly(a, d_root)
    
    for v in T.nodes():
        if v == root:
            continue
        c_v = len(children_dict[v])
        if v in A:
            result = result * falling_factorial_poly(a - 1, c_v)
        else:  # v in B
            result = result * falling_factorial_poly(b - 1, c_v)
    
    return expand(result)


def poly_sort_key(monom):
    """Sort monomials by total degree, then by power of a (descending)."""
    i, j = monom
    return (i + j, -i)


def format_poly(expr, a, b):
    """Format polynomial sorted by total degree, then by power of a."""
    p = Poly(expr, a, b)
    terms = p.as_dict()
    # Sort monomials
    sorted_monoms = sorted(terms.keys(), key=poly_sort_key)
    
    parts = []
    for monom in sorted_monoms:
        coeff = terms[monom]
        i, j = monom
        if coeff == 0:
            continue
        # Build term string
        if i == 0 and j == 0:
            term = str(coeff)
        elif i == 0:
            if j == 1:
                term = f"{coeff}*b" if coeff != 1 else "b"
                if coeff == -1:
                    term = "-b"
            else:
                term = f"{coeff}*b**{j}" if coeff != 1 else f"b**{j}"
                if coeff == -1:
                    term = f"-b**{j}"
        elif j == 0:
            if i == 1:
                term = f"{coeff}*a" if coeff != 1 else "a"
                if coeff == -1:
                    term = "-a"
            else:
                term = f"{coeff}*a**{i}" if coeff != 1 else f"a**{i}"
                if coeff == -1:
                    term = f"-a**{i}"
        else:
            a_part = f"a**{i}" if i > 1 else "a"
            b_part = f"b**{j}" if j > 1 else "b"
            if coeff == 1:
                term = f"{a_part}*{b_part}"
            elif coeff == -1:
                term = f"-{a_part}*{b_part}"
            else:
                term = f"{coeff}*{a_part}*{b_part}"
        parts.append(term)
    
    if not parts:
        return "0"
    
    result = parts[0]
    for part in parts[1:]:
        if part.startswith('-'):
            result += f" - {part[1:]}"
        else:
            result += f" + {part}"
    return result


def main():
    a, b = symbols('a b')
    
    all_trees = []
    tree_index = 0
    
    print("=" * 80)
    print("ENUMERATION OF ALL UNLABELED TREES WITH 1..6 EDGES")
    print("=" * 80)
    
    for n in range(2, 8):  # vertices 2..7 => edges 1..6
        trees_n = list(nx.nonisomorphic_trees(n))
        print(f"\n--- {len(trees_n)} tree(s) with {n} vertices ({n-1} edges) ---")
        
        for T in trees_n:
            tree_index += 1
            edges = sorted(T.edges())
            deg_seq = sorted([T.degree(v) for v in T.nodes()], reverse=True)
            
            A, B = get_bipartition(T, root=0)
            
            emb1 = compute_emb1(T, 0, A, B, a, b)
            emb2 = expand(emb1.subs([(a, b), (b, a)]))
            
            all_trees.append({
                'index': tree_index,
                'n_vertices': n,
                'n_edges': n - 1,
                'edges': edges,
                'deg_seq': deg_seq,
                'A_size': len(A),
                'B_size': len(B),
                'emb1': emb1,
                'emb2': emb2,
            })
            
            print(f"\nTree #{tree_index}: edges={edges}")
            print(f"  Degree sequence: {deg_seq}")
            print(f"  Bipartition: |A|={len(A)}, |B|={len(B)}")
            print(f"  emb_1(a,b) = {format_poly(emb1, a, b)}")
            print(f"  emb_2(a,b) = {format_poly(emb2, a, b)}")
    
    print(f"\n\nTotal trees enumerated: {tree_index}")
    
    # ---- Collision groups ----
    print("\n" + "=" * 80)
    print("COLLISION GROUPS")
    print("(Trees sharing the same unordered pair {emb_1, emb_2})")
    print("=" * 80)
    
    # Group by frozenset of (emb1, emb2) as an unordered pair
    collision_map = defaultdict(list)
    for t in all_trees:
        # Use frozenset of the two polynomial expressions (expanded)
        key = frozenset([t['emb1'], t['emb2']])
        collision_map[key].append(t['index'])
    
    n_collisions = 0
    for key, indices in sorted(collision_map.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True):
        if len(indices) > 1:
            n_collisions += 1
            print(f"\nCollision group (size {len(indices)}): Trees {indices}")
            for idx in indices:
                t = all_trees[idx - 1]
                print(f"  Tree #{idx}: edges={t['edges']}, deg_seq={t['deg_seq']}")
                print(f"    emb_1 = {format_poly(t['emb1'], a, b)}")
                print(f"    emb_2 = {format_poly(t['emb2'], a, b)}")
    
    if n_collisions == 0:
        print("\nNo collisions found! All 24 trees have distinct {emb_1, emb_2} pairs.")
    else:
        print(f"\nTotal collision groups: {n_collisions}")
        n_distinct = len(collision_map)
        print(f"Distinct unordered pairs: {n_distinct}")
    
    # ---- Coefficient matrix ----
    print("\n" + "=" * 80)
    print("COEFFICIENT MATRIX AND RANK COMPUTATION")
    print("=" * 80)
    
    # Collect all monomials appearing in any polynomial
    all_monoms = set()
    for t in all_trees:
        p1 = Poly(t['emb1'], a, b)
        p2 = Poly(t['emb2'], a, b)
        all_monoms.update(p1.as_dict().keys())
        all_monoms.update(p2.as_dict().keys())
    
    # Sort monomials
    sorted_monoms = sorted(all_monoms, key=poly_sort_key)
    
    print(f"\nMonomial basis ({len(sorted_monoms)} monomials):")
    monom_strs = []
    for (i, j) in sorted_monoms:
        if i == 0 and j == 0:
            s = "1"
        elif i == 0:
            s = f"b^{j}" if j > 1 else "b"
        elif j == 0:
            s = f"a^{i}" if i > 1 else "a"
        else:
            a_s = f"a^{i}" if i > 1 else "a"
            b_s = f"b^{j}" if j > 1 else "b"
            s = f"{a_s}*{b_s}"
        monom_strs.append(s)
    print(f"  {monom_strs}")
    
    # Build 24-row matrix (emb_1 only)
    rows_emb1 = []
    for t in all_trees:
        p = Poly(t['emb1'], a, b)
        d = p.as_dict()
        row = [d.get(m, 0) for m in sorted_monoms]
        rows_emb1.append(row)
    
    M_emb1 = Matrix(rows_emb1)
    rank_emb1 = M_emb1.rank()
    
    print(f"\n24-row matrix (emb_1 only): {M_emb1.shape[0]} rows x {M_emb1.shape[1]} cols")
    print(f"Rank of 24-row matrix (emb_1 only): {rank_emb1}")
    
    # Build 48-row matrix (emb_1 and emb_2)
    rows_both = []
    for t in all_trees:
        p1 = Poly(t['emb1'], a, b)
        d1 = p1.as_dict()
        row1 = [d1.get(m, 0) for m in sorted_monoms]
        rows_both.append(row1)
    for t in all_trees:
        p2 = Poly(t['emb2'], a, b)
        d2 = p2.as_dict()
        row2 = [d2.get(m, 0) for m in sorted_monoms]
        rows_both.append(row2)
    
    M_both = Matrix(rows_both)
    rank_both = M_both.rank()
    
    print(f"\n48-row matrix (emb_1 and emb_2): {M_both.shape[0]} rows x {M_both.shape[1]} cols")
    print(f"Rank of 48-row matrix (emb_1 + emb_2): {rank_both}")
    
    # ---- Summary ----
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total trees: {tree_index}")
    print(f"Distinct {'{emb_1, emb_2}'} pairs: {len(collision_map)}")
    print(f"Collision groups (size > 1): {n_collisions}")
    print(f"Monomial basis size: {len(sorted_monoms)}")
    print(f"Rank (24-row, emb_1 only): {rank_emb1}")
    print(f"Rank (48-row, emb_1 + emb_2): {rank_both}")


if __name__ == '__main__':
    main()
