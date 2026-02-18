#!/usr/bin/env python3
"""
Compute coeff_k(tau) for all connected graph types via:

  gamma_k(K_n) = sum_{tau: |tau| <= k} coeff_k(tau) * sub(tau, K_n)

Key fix: at grade k, only include types with <= k edges as unknowns.
Types with > k edges have coeff_k = 0 by definition.
"""

import networkx as nx
from itertools import combinations
from collections import defaultdict
from fractions import Fraction
import time
import sys

def enumerate_connected_graphs(max_edges):
    """Enumerate all connected graphs with 1..max_edges edges."""
    all_types = []
    
    for e in range(1, max_edges + 1):
        types_this_e = []
        min_v = 2
        max_v = e + 1
        
        for v in range(min_v, max_v + 1):
            if e > v * (v - 1) // 2 or e < v - 1:
                continue
            
            all_possible_edges = list(combinations(range(v), 2))
            for edge_combo in combinations(all_possible_edges, e):
                G = nx.Graph()
                G.add_nodes_from(range(v))
                G.add_edges_from(edge_combo)
                
                if not nx.is_connected(G):
                    continue
                
                is_new = True
                for G2, v2, e2, aut2 in types_this_e:
                    if v2 == v and nx.is_isomorphic(G, G2):
                        is_new = False
                        break
                
                if is_new:
                    aut_size = count_automorphisms(G)
                    types_this_e.append((G, v, e, aut_size))
        
        all_types.extend(types_this_e)
        print(f"  e={e}: {len(types_this_e)} types (cumulative: {len(all_types)})", flush=True)
    
    return all_types


def count_automorphisms(G):
    from itertools import permutations
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    edge_set = set(frozenset(e) for e in G.edges())
    count = 0
    for perm in permutations(nodes):
        mapping = dict(zip(nodes, perm))
        mapped_edges = set(frozenset((mapping[u], mapping[v])) for u, v in G.edges())
        if mapped_edges == edge_set:
            count += 1
    return count


def classify_type(G):
    v = G.number_of_nodes()
    e = G.number_of_edges()
    deg_seq = tuple(sorted(dict(G.degree()).values()))
    
    if e == v - 1:  # tree
        if e == 1: return "K2"
        elif e == 2: return "P3"
        elif e == 3:
            if max(deg_seq) == 3: return "K1_3"
            else: return "P4"
        elif e == 4:
            if max(deg_seq) == 4: return "K1_4"
            elif max(deg_seq) == 3: return "fork"
            else: return "P5"
        elif e == 5:
            if max(deg_seq) == 5: return "K1_5"
            elif deg_seq == (1,1,2,2,2,2): return "P6"
            elif deg_seq == (1,1,1,1,3,3): return "dblstar"
            elif deg_seq == (1,1,1,1,2,4): return "spider"
            elif deg_seq == (1,1,1,2,2,3):
                v3 = [n for n in G.nodes() if G.degree(n) == 3][0]
                nbr_degs = sorted(G.degree(u) for u in G.neighbors(v3))
                return "catA" if nbr_degs == [1, 2, 2] else "catB"
        return f"tree_{v}v_{deg_seq}"
    else:
        return f"g{v}v{e}e_{deg_seq}"


def gamma_k_Kn(k, n):
    """Compute gamma_k(K_n) = |E(L^k(K_n))| exactly."""
    v = n * (n - 1) // 2
    d = 2 * (n - 2)
    for j in range(2, k + 1):
        v_new = v * d // 2
        d_new = 2 * d - 2
        v = v_new
        d = d_new
    return v * d // 2


def sub_tau_Kn(v_tau, aut_size, n):
    if n < v_tau:
        return 0
    ff = 1
    for i in range(v_tau):
        ff *= (n - i)
    return ff // aut_size


def solve_grade(k, active_indices, all_types, n_values):
    """Solve for coeff_k(tau) for active types (those with <= k edges)."""
    n_unknowns = len(active_indices)
    n_eqs = len(n_values)
    
    S = []
    gamma = []
    for n_val in n_values:
        row = []
        for j in active_indices:
            G, v, e, aut = all_types[j]
            row.append(Fraction(sub_tau_Kn(v, aut, n_val)))
        S.append(row)
        gamma.append(Fraction(gamma_k_Kn(k, n_val)))
    
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
    
    solution = {}
    for ci, j in enumerate(active_indices):
        val = aug[ci][n_unknowns] / aug[ci][ci]
        if val.denominator != 1:
            return None, f"Non-integer coeff for type {j}: {val}"
        solution[j] = int(val)
    
    # Verify with extra equations
    max_residual = Fraction(0)
    for i in range(n_unknowns, n_eqs):
        residual = abs(aug[i][n_unknowns])  # should be 0 after elimination
        max_residual = max(max_residual, residual)
    
    return solution, max_residual


def main():
    max_edges = 5
    max_grade = 16
    
    if len(sys.argv) > 1:
        max_edges = int(sys.argv[1])
    if len(sys.argv) > 2:
        max_grade = int(sys.argv[2])
    
    print(f"Parameters: max_edges={max_edges}, max_grade={max_grade}\n")
    
    print("Enumerating connected graph types...")
    t0 = time.time()
    all_types = enumerate_connected_graphs(max_edges)
    print(f"Total types: {len(all_types)} ({time.time()-t0:.1f}s)\n")
    
    type_names = []
    type_is_tree = []
    type_edges = []
    for G, v, e, aut in all_types:
        type_names.append(classify_type(G))
        type_is_tree.append(e == v - 1)
        type_edges.append(e)
    
    for e in range(1, max_edges + 1):
        types_e = [(i, type_names[i], all_types[i][1], all_types[i][3]) 
                   for i in range(len(all_types)) if type_edges[i] == e]
        trees_e = [x for x in types_e if type_is_tree[x[0]]]
        print(f"  e={e}: {len(types_e)} types ({len(trees_e)} trees)")
        for idx, name, v, aut in types_e:
            tree_mark = " [TREE]" if type_is_tree[idx] else ""
            print(f"    {name}: {v}v, |Aut|={aut}{tree_mark}")
    
    v_max = max(v for G, v, e, aut in all_types)
    
    all_coeffs = {}  # all_coeffs[k][type_idx] = value
    
    print(f"\nComputing coefficients grade by grade...\n")
    
    for k in range(1, max_grade + 1):
        t0 = time.time()
        
        active = [i for i in range(len(all_types)) if type_edges[i] <= k]
        n_active = len(active)
        n_extra = 5
        n_values = list(range(v_max, v_max + n_active + n_extra))
        
        solution, info = solve_grade(k, active, all_types, n_values)
        elapsed = time.time() - t0
        
        if solution is None:
            print(f"  Grade {k}: FAILED — {info} ({elapsed:.2f}s)")
            continue
        
        all_coeffs[k] = solution
        
        nonzero = sum(1 for v in solution.values() if v != 0)
        tree_coeffs = [(type_names[j], solution[j]) 
                       for j in active if type_is_tree[j] and solution[j] != 0]
        
        print(f"  Grade {k}: {n_active} unknowns, {nonzero} nonzero, residual={info} ({elapsed:.2f}s)")
        if tree_coeffs:
            tc_str = ", ".join(f"{name}={val}" for name, val in tree_coeffs)
            print(f"    Trees: {tc_str}")
    
    # ============================================================
    # Tree independence analysis
    # ============================================================
    
    print(f"\n{'='*70}")
    print(f"TREE COEFFICIENT INDEPENDENCE ANALYSIS")
    print(f"{'='*70}")
    
    for target_e in range(1, max_edges + 1):
        tree_indices = [i for i in range(len(all_types)) 
                        if type_is_tree[i] and type_edges[i] == target_e]
        n_trees = len(tree_indices)
        if n_trees <= 1:
            continue
        
        tree_labels = [type_names[i] for i in tree_indices]
        min_grade = target_e
        grades = list(range(min_grade, max_grade + 1))
        n_rows = len(grades)
        
        M = []
        for k in grades:
            row = [all_coeffs.get(k, {}).get(j, 0) for j in tree_indices]
            M.append(row)
        
        print(f"\nTrees with {target_e} edges ({n_trees} types): {tree_labels}")
        header = "  grade " + "".join(f"{name:>14}" for name in tree_labels)
        print(header)
        for ki, k in enumerate(grades):
            row_str = f"  k={k:2d}  " + "".join(f"{M[ki][ti]:>14}" for ti in range(n_trees))
            print(row_str)
        
        # Exact rank via Fraction Gaussian elimination
        rank = exact_rank(M, n_rows, n_trees)
        
        print(f"\n  Rank: {rank} / {n_trees}", end="")
        print(" — FULL RANK ✓" if rank == n_trees else " — RANK DEFICIENT")
        
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
    
    print(f"\n{'='*70}")
    print(f"ALL TREES ({n_all_trees} types): {all_tree_labels}")
    print(f"{'='*70}")
    header = "  grade " + "".join(f"{name:>14}" for name in all_tree_labels)
    print(header)
    for ki, k in enumerate(grades):
        row_str = f"  k={k:2d}  " + "".join(f"{M[ki][ti]:>14}" for ti in range(n_all_trees))
        print(row_str)
    
    rank = exact_rank(M, n_rows, n_all_trees)
    print(f"\n  Rank: {rank} / {n_all_trees}", end="")
    print(" — FULL RANK ✓" if rank == n_all_trees else " — RANK DEFICIENT")
    
    print("  Cumulative rank:")
    for nk in range(1, min(n_rows + 1, n_all_trees + 3)):
        r = exact_rank(M[:nk], nk, n_all_trees)
        print(f"    Grades {min_e}..{min_e + nk - 1}: rank {r}")


def exact_rank(M, n_rows, n_cols):
    """Compute exact rank of integer matrix using Fraction arithmetic."""
    Mf = [[Fraction(M[i][j]) for j in range(n_cols)] for i in range(n_rows)]
    rank = 0
    rp = 0
    for col in range(n_cols):
        piv = None
        for r in range(rp, n_rows):
            if Mf[r][col] != 0:
                piv = r
                break
        if piv is None:
            continue
        Mf[rp], Mf[piv] = Mf[piv], Mf[rp]
        for r in range(n_rows):
            if r != rp and Mf[r][col] != 0:
                f = Mf[r][col] / Mf[rp][col]
                for c in range(n_cols):
                    Mf[r][c] -= f * Mf[rp][c]
        rp += 1
        rank += 1
    return rank


if __name__ == "__main__":
    main()