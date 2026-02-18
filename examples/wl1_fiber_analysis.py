#!/usr/bin/env python3
"""
Fiber decomposition analysis for WL-1 equivalent graphs:
  - Dumbbell: two triangles connected by a bridge (6 vertices, 7 edges)
  - Chorded C6: 6-cycle with antipodal chord (6 vertices, 7 edges)

These graphs are WL-1 equivalent, which implies identical Graham sequences.
We decompose γ_k(G) = Σ_τ coeff_k(τ) · count(τ, G) and analyze how
differing subgraph counts produce identical weighted sums.

Usage:
    python3 wl1_fiber_analysis.py [--max-k K] [--max-edges M]
"""

import sys
import time
from collections import defaultdict
from itertools import combinations, permutations


# ============================================================
#  Graph canonical forms (general graphs, not just trees)
# ============================================================

def canonical_graph(edges, vertices=None):
    """Canonical form for a graph: smallest adjacency representation
    under all vertex permutations.
    
    For small graphs (≤ 8 vertices), brute-force over permutations.
    Returns a frozenset of edges as canonical form.
    """
    if not edges:
        n = len(vertices) if vertices else 0
        return ("empty", n)
    
    if vertices is None:
        vertices = set()
        for u, v in edges:
            vertices.add(u)
            vertices.add(v)
    
    vlist = sorted(vertices)
    n = len(vlist)
    
    if n > 10:
        # For larger graphs, use a simple invariant-based approach
        # (degree sequence + sorted adjacency)
        # This may not be perfect but works for connected subgraphs of small graphs
        return _canon_large(edges, vlist)
    
    # Brute force: try all permutations of vertex labels
    best = None
    for perm in permutations(range(n)):
        v_map = {vlist[i]: perm[i] for i in range(n)}
        relabeled = tuple(sorted(
            (min(v_map[u], v_map[v]), max(v_map[u], v_map[v]))
            for u, v in edges
        ))
        if best is None or relabeled < best:
            best = relabeled
    return best


def _canon_large(edges, vlist):
    """Fallback canonical form for larger graphs."""
    n = len(vlist)
    v_map = {v: i for i, v in enumerate(vlist)}
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[v_map[u]].add(v_map[v])
        adj[v_map[v]].add(v_map[u])
    
    # Sort by (degree, sorted neighbor degrees, ...)
    deg_seq = tuple(sorted([len(adj[i]) for i in range(n)], reverse=True))
    edge_tup = tuple(sorted(
        (min(v_map[u], v_map[v]), max(v_map[u], v_map[v]))
        for u, v in edges
    ))
    return (deg_seq, edge_tup)


# ============================================================
#  Connected components and subgraph enumeration
# ============================================================

def is_connected(edges, vertices=None):
    """Check if edges form a connected graph."""
    if not edges:
        return vertices is not None and len(vertices) <= 1
    
    adj = defaultdict(set)
    verts = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        verts.add(u)
        verts.add(v)
    
    if vertices is not None:
        verts = set(vertices)
    
    start = next(iter(verts))
    visited = set()
    stack = [start]
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        for u in adj[v]:
            if u in verts and u not in visited:
                stack.append(u)
    return len(visited) == len(verts)


def connected_components_edges(edges):
    """Return list of (vertex_set, edge_list) per connected component."""
    if not edges:
        return []
    
    adj = defaultdict(set)
    verts = set()
    edge_map = defaultdict(list)  # vertex -> list of edge indices
    for i, (u, v) in enumerate(edges):
        adj[u].add(v)
        adj[v].add(u)
        verts.add(u)
        verts.add(v)
    
    visited = set()
    components = []
    
    for start in sorted(verts):
        if start in visited:
            continue
        comp_verts = set()
        stack = [start]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            comp_verts.add(v)
            for u in adj[v]:
                if u not in visited:
                    stack.append(u)
        
        comp_edges = [(u, v) for u, v in edges if u in comp_verts]
        components.append((comp_verts, comp_edges))
    
    return components


def enumerate_connected_subgraphs(edges, max_size=None):
    """Enumerate all connected subgraphs by edge subsets.
    
    Returns dict: canonical_form -> (count, representative_edges, n_vertices, n_edges)
    """
    n_edges = len(edges)
    if max_size is None:
        max_size = n_edges
    
    counts = defaultdict(lambda: [0, None, 0, 0])
    
    for size in range(1, min(max_size, n_edges) + 1):
        for subset in combinations(range(n_edges), size):
            sub_edges = [edges[i] for i in subset]
            if not is_connected(sub_edges):
                continue
            
            canon = canonical_graph(sub_edges)
            entry = counts[canon]
            entry[0] += 1
            if entry[1] is None:
                verts = set()
                for u, v in sub_edges:
                    verts.add(u)
                    verts.add(v)
                entry[1] = sub_edges
                entry[2] = len(verts)
                entry[3] = size
    
    return {k: tuple(v) for k, v in counts.items()}


# ============================================================
#  Graph description utilities
# ============================================================

def describe_graph(edges, n_vertices):
    """Return a human-readable description of a small graph."""
    m = len(edges)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    verts = set()
    for u, v in edges:
        verts.add(u)
        verts.add(v)
    n = len(verts)
    
    deg_seq = sorted([len(adj[v]) for v in verts], reverse=True)
    
    # Detect special structures
    if m == 1:
        return f"K2"
    if m == n - 1:
        # Tree
        if all(d <= 2 for d in deg_seq):
            return f"P{n}"
        max_deg = max(deg_seq)
        if deg_seq.count(1) == n - 1:
            return f"K1,{n-1}"
        return f"Tree({n}v,ds={deg_seq})"
    if m == n:
        if all(d == 2 for d in deg_seq):
            return f"C{n}"
        # Cycle with a tail or unicyclic
        return f"Unicyclic({n}v,{m}e,ds={deg_seq})"
    if n == m and n == 3:
        return "K3"
    
    # General
    return f"Graph({n}v,{m}e,ds={deg_seq})"


# ============================================================
#  Line graph iteration
# ============================================================

def line_graph(edges, n_vertices):
    """Compute line graph. Returns (new_edges, new_n_vertices)."""
    m = len(edges)
    if m == 0:
        return [], 0
    
    incident = defaultdict(list)
    for idx, (u, v) in enumerate(edges):
        incident[u].append(idx)
        incident[v].append(idx)
    
    new_edges = set()
    for v in range(n_vertices):
        inc = incident.get(v, [])
        for i in range(len(inc)):
            for j in range(i + 1, len(inc)):
                a, b = inc[i], inc[j]
                if a > b: a, b = b, a
                new_edges.add((a, b))
    
    return sorted(new_edges), m


def line_graph_general(edges):
    """Line graph for graphs with arbitrary vertex labels."""
    if not edges:
        return [], 0
    
    # Relabel vertices to 0..n-1
    verts = set()
    for u, v in edges:
        verts.add(u)
        verts.add(v)
    v_map = {v: i for i, v in enumerate(sorted(verts))}
    n = len(verts)
    
    relabeled = [(v_map[u], v_map[v]) for u, v in edges]
    return line_graph(relabeled, n)


def gamma_sequence(edges, max_k, max_edges=2_000_000, verbose=False):
    """Compute γ_0, γ_1, ..., γ_max_k for a connected graph."""
    if not edges:
        return [0] * (max_k + 1)
    
    verts = set()
    for u, v in edges:
        verts.add(u)
        verts.add(v)
    n = len(verts)
    
    v_map = {v: i for i, v in enumerate(sorted(verts))}
    current_edges = [(v_map[u], v_map[v]) for u, v in edges]
    current_n = n
    
    seq = [current_n]
    
    for k in range(1, max_k + 1):
        t0 = time.time()
        new_edges, new_n = line_graph(current_edges, current_n)
        if verbose:
            print(f"      L^{k}: |V|={new_n}, |E|={len(new_edges)} ({time.time()-t0:.2f}s)",
                  flush=True)
        seq.append(new_n)
        if new_n == 0 or len(new_edges) > max_edges:
            while len(seq) <= max_k:
                seq.append(None)
            break
        current_edges = new_edges
        current_n = new_n
    
    return seq


# ============================================================
#  Coefficient extraction via Möbius inversion
# ============================================================

def compute_all_coefficients(all_types, max_k, max_edges=5_000_000):
    """Compute coeff_k(τ) for all types by bootstrapping.
    
    all_types: dict of canon -> (count, edges, n_verts, n_edges)
    
    Process types in order of increasing edge count.
    For each τ:
      1. Compute γ_k(τ) by line graph iteration
      2. Enumerate connected subgraphs of τ
      3. coeff_k(τ) = γ_k(τ) - Σ_{σ ⊊ τ} coeff_k(σ) · count(σ, τ)
    
    Returns:
      gammas: dict canon -> [γ_0, ..., γ_max_k]
      coeffs: dict canon -> [c_0, ..., c_max_k]
      subtypes: dict canon -> dict of sub_canon -> count
    """
    # Sort by edge count
    sorted_types = sorted(all_types.items(), key=lambda x: x[1][3])
    
    gammas = {}
    coeffs = {}
    subtypes = {}
    
    total = len(sorted_types)
    
    for idx, (canon, (count, edges, nv, ne)) in enumerate(sorted_types):
        print(f"  [{idx+1}/{total}] {describe_graph(edges, nv)} ({ne} edges)...",
              end="", flush=True)
        
        t0 = time.time()
        
        # Step 1: compute γ_k(τ)
        gamma = gamma_sequence(edges, max_k, max_edges)
        gammas[canon] = gamma
        
        # Step 2: enumerate proper connected subgraphs of τ
        if ne > 1:
            sub_counts = enumerate_connected_subgraphs(edges, max_size=ne - 1)
        else:
            sub_counts = {}
        subtypes[canon] = {sc: sv[0] for sc, sv in sub_counts.items()}
        
        # Step 3: Möbius extraction
        coeff = list(gamma)  # start with γ_k
        for sub_canon, sub_count in subtypes[canon].items():
            if sub_canon in coeffs:
                for k in range(min(len(coeff), len(coeffs[sub_canon]))):
                    if coeff[k] is not None and coeffs[sub_canon][k] is not None:
                        coeff[k] -= coeffs[sub_canon][k] * sub_count
        
        coeffs[canon] = coeff
        
        elapsed = time.time() - t0
        
        # Show nonzero coefficients
        nonzero = [(k, c) for k, c in enumerate(coeff) if c is not None and c != 0 and k > 0]
        if nonzero:
            nz_str = ", ".join(f"c_{k}={c}" for k, c in nonzero[:5])
            print(f" {nz_str} ({elapsed:.2f}s)")
        else:
            print(f" all zero ({elapsed:.2f}s)")
    
    return gammas, coeffs, subtypes


# ============================================================
#  Main analysis
# ============================================================

def main():
    max_k = 5
    max_edges = 2_000_000
    
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--max-k" and i + 1 < len(args):
            max_k = int(args[i + 1]); i += 2
        elif args[i] == "--max-edges" and i + 1 < len(args):
            max_edges = int(args[i + 1]); i += 2
        else:
            i += 1
    
    # ---- Define the graphs ----
    
    # Dumbbell: triangle 0-1-2, triangle 3-4-5, bridge 2-3
    dumbbell = [(0,1), (1,2), (0,2), (2,3), (3,4), (4,5), (3,5)]
    
    # Chorded C6: cycle 0-1-2-3-4-5-0 plus chord 0-3
    chorded_c6 = [(0,1), (1,2), (2,3), (3,4), (4,5), (0,5), (0,3)]
    
    print("=" * 70)
    print("  WL-1 Equivalent Pair: Dumbbell vs Chorded C6")
    print("=" * 70)
    
    for name, edges in [("Dumbbell", dumbbell), ("Chorded C6", chorded_c6)]:
        degs = [0] * 6
        for u, v in edges:
            degs[u] += 1; degs[v] += 1
        print(f"\n  {name}:")
        print(f"    Edges: {edges}")
        print(f"    Degree seq: {sorted(degs, reverse=True)}")
    
    # ---- Compute Graham sequences directly ----
    
    print(f"\n{'=' * 70}")
    print(f"  Graham Sequences (direct line graph iteration)")
    print(f"{'=' * 70}")
    
    seq_d = gamma_sequence(dumbbell, max_k, max_edges, verbose=True)
    seq_c = gamma_sequence(chorded_c6, max_k, max_edges, verbose=True)
    
    print(f"\n  {'k':>3s} {'Dumbbell':>14s} {'Chorded C6':>14s} {'diff':>10s}")
    print(f"  {'-' * 45}")
    for k in range(min(len(seq_d), len(seq_c))):
        if seq_d[k] is None or seq_c[k] is None:
            print(f"  {k:>3d} {'—':>14s} {'—':>14s}")
        else:
            d = seq_d[k] - seq_c[k]
            eq = "✓" if d == 0 else f"✗ (diff={d})"
            print(f"  {k:>3d} {seq_d[k]:>14,} {seq_c[k]:>14,} {eq:>10s}")
    
    # ---- Enumerate subgraph types ----
    
    print(f"\n{'=' * 70}")
    print(f"  Enumerating connected subgraph types")
    print(f"{'=' * 70}")
    
    print("\n  Dumbbell subgraphs:")
    t0 = time.time()
    types_d = enumerate_connected_subgraphs(dumbbell)
    print(f"    {len(types_d)} types found ({time.time()-t0:.3f}s)")
    
    print("\n  Chorded C6 subgraphs:")
    t0 = time.time()
    types_c = enumerate_connected_subgraphs(chorded_c6)
    print(f"    {len(types_c)} types found ({time.time()-t0:.3f}s)")
    
    # Merge all types
    all_types = {}
    for canon, (count, edges, nv, ne) in types_d.items():
        all_types[canon] = (count, edges, nv, ne)
    for canon, (count, edges, nv, ne) in types_c.items():
        if canon not in all_types:
            all_types[canon] = (count, edges, nv, ne)
    
    print(f"\n  Total distinct types across both graphs: {len(all_types)}")
    
    # Show counts comparison
    print(f"\n  {'Type':>40s} {'#edges':>6s} {'Dumb':>6s} {'Chord':>6s} {'diff':>6s}")
    print(f"  {'-' * 68}")
    
    all_canons = sorted(all_types.keys(), key=lambda c: (all_types[c][3], c))
    for canon in all_canons:
        _, edges, nv, ne = all_types[canon]
        cd = types_d.get(canon, (0, None, 0, 0))[0]
        cc = types_c.get(canon, (0, None, 0, 0))[0]
        diff = cd - cc
        desc = describe_graph(edges, nv)
        mark = " ←" if diff != 0 else ""
        print(f"  {desc:>40s} {ne:>6d} {cd:>6d} {cc:>6d} {diff:>+6d}{mark}")
    
    # ---- Compute coefficients ----
    
    print(f"\n{'=' * 70}")
    print(f"  Computing fiber coefficients (bootstrap)")
    print(f"{'=' * 70}\n")
    
    gammas, coeffs, subtypes_map = compute_all_coefficients(
        all_types, max_k, max_edges
    )
    
    # ---- Verify decomposition ----
    
    print(f"\n{'=' * 70}")
    print(f"  Verification: γ_k = Σ coeff_k(τ) · count(τ, G)")
    print(f"{'=' * 70}")
    
    for name, graph_edges, graph_types, graph_seq in [
        ("Dumbbell", dumbbell, types_d, seq_d),
        ("Chorded C6", chorded_c6, types_c, seq_c),
    ]:
        print(f"\n  {name}:")
        for k in range(1, max_k + 1):
            if graph_seq[k] is None:
                break
            total = 0
            for canon in all_canons:
                count = graph_types.get(canon, (0,))[0]
                if count == 0:
                    continue
                ck = coeffs.get(canon, [])
                if k < len(ck) and ck[k] is not None:
                    total += ck[k] * count
            
            match = "✓" if total == graph_seq[k] else f"✗ (got {total})"
            print(f"    k={k}: Σ = {total:,}  vs  γ_{k} = {graph_seq[k]:,}  {match}")
    
    # ---- Analyze cancellation for WL-1 equivalence ----
    
    print(f"\n{'=' * 70}")
    print(f"  Cancellation Analysis: how differing counts produce equal γ_k")
    print(f"{'=' * 70}")
    
    for k in range(1, max_k + 1):
        if seq_d[k] is None:
            break
        
        # Compute per-type contributions
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
                desc = describe_graph(edges, nv)
                contribs.append((ne, desc, coeff_val, cd, cc, contrib_d, contrib_c, diff_contrib))
        
        if not contribs:
            continue
        
        # Only show grades where there's something interesting
        has_diff = any(c[7] != 0 for c in contribs)
        
        print(f"\n  Grade k = {k}:")
        print(f"    {'Type':>35s} {'coeff':>8s} {'cnt_D':>6s} {'cnt_C':>6s} "
              f"{'ctrb_D':>10s} {'ctrb_C':>10s} {'diff':>10s}")
        print(f"    {'-' * 90}")
        
        total_d = 0
        total_c = 0
        
        for ne, desc, cv, cd, cc, ctd, ctc, dc in sorted(contribs):
            total_d += ctd
            total_c += ctc
            mark = " ←" if dc != 0 else ""
            print(f"    {desc:>35s} {cv:>8,} {cd:>6d} {cc:>6d} "
                  f"{ctd:>10,} {ctc:>10,} {dc:>+10,}{mark}")
        
        print(f"    {'':>35s} {'':>8s} {'':>6s} {'':>6s} "
              f"{total_d:>10,} {total_c:>10,} {total_d - total_c:>+10,}")
        
        if has_diff:
            # Show just the nonzero-diff types
            print(f"\n    Types contributing to cancellation at k={k}:")
            net = 0
            for ne, desc, cv, cd, cc, ctd, ctc, dc in sorted(contribs):
                if dc != 0:
                    net += dc
                    print(f"      {desc}: coeff={cv}, count_diff={cd-cc}, "
                          f"contribution_diff={dc:+d} (running: {net:+d})")
            print(f"      Net difference: {net}")


if __name__ == "__main__":
    main()