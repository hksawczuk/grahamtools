#!/usr/bin/env python3
"""
Compute the coefficient matrix M[k, tau] for trees on 6 vertices in K_6.

Trees on 6 vertices (6 isomorphism classes):
  1. P_6: path, deg seq (1,1,2,2,2,2)
  2. Caterpillar 1: (1,1,1,2,2,3) - path with one leaf at degree-3 end
     0-1-2-3-4, 2-5  => degs: 1,2,3,2,1,1
  3. Caterpillar 2: (1,1,1,2,2,3) - path with one leaf at interior
     0-1-2-3-4, 1-5  => degs: 1,3,2,2,1,1  (isomorphic to #2? No...)
     
Actually let me just list them carefully. Trees on 6 vertices:
  1. P_6: 0-1-2-3-4-5
  2. 0-1-2-3-4, 2-5  (caterpillar, "T-shape")
  3. 0-1-2-3, 2-4, 2-5  (K_{1,3} with path extension = "Y + tail")
     Actually: vertex 2 has deg 4? No: 2 connects to 1,3,4,5 => deg 4. 
     Hmm wait: 0-1, 1-2, 2-3, 2-4, 2-5 => degs 1,2,4,1,1,1 => this is 
     a star K_{1,4} with one edge subdivided. Deg seq (1,1,1,1,2,4).
     
Let me be systematic. There are exactly 6 unlabeled trees on 6 vertices.

By degree sequence:
  1. (1,1,2,2,2,2) - P_6
  2. (1,1,1,2,2,3) - exactly one vertex of degree 3
     Two non-iso trees with this deg seq? Let me check.
     a) 0-1-2-3-4, 2-5: center of T at vertex 2 (deg 3), path continues.
        Vertex degs: 0:1, 1:2, 2:3, 3:2, 4:1, 5:1
     b) 0-1-2-3-4, 3-5: vertex 3 has deg 3.
        Vertex degs: 0:1, 1:2, 2:2, 3:3, 4:1, 5:1
        Same deg seq (1,1,1,2,2,3). Is this isomorphic to (a)?
        In (a): deg-3 vertex has neighbors of degrees 1,2,2
        In (b): deg-3 vertex has neighbors of degrees 1,1,2
        Different! So these are non-isomorphic.
  3. (1,1,1,1,2,4) - one vertex of degree 4
     0-1, 1-2, 1-3, 1-4, 4-5: degs 1,4,1,1,2,1 => (1,1,1,1,2,4) ✓
  4. (1,1,1,1,1,5) - K_{1,5}
  5. (1,1,1,1,3,3) - double star / dumbbell
     0-1-2, 1-3, 1-4, 2-5? No: degs 1,3,2,1,1,1 => (1,1,1,1,2,3)
     Try: 0-1, 1-2, 1-3, 2-4, 2-5: degs 1,3,3,1,1,1 => (1,1,1,1,3,3) ✓

That's 6 total: P_6, caterpillar-a, caterpillar-b, spider(4), K_{1,5}, double-star.
Wait, that's only 6 if caterpillar-a and caterpillar-b are distinct.

Let me verify the count: Cayley's formula gives labeled trees, but the OEIS 
sequence for unlabeled trees on n vertices (A000055) gives:
n=1:1, n=2:1, n=3:1, n=4:2, n=5:3, n=6:6.
Yes, 6 unlabeled trees on 6 vertices. ✓

So the 6 trees on 6 vertices are:

1. P_6: path
   edges: {0-1, 1-2, 2-3, 3-4, 4-5}
   deg seq: (2,2,2,2,1,1)

2. Caterpillar A: "T with long arm" (deg-3 has nbrs of deg 1,2,2)
   edges: {0-1, 1-2, 2-3, 3-4, 2-5}
   deg seq: (1,2,3,2,1,1)

3. Caterpillar B: "T with short arms" (deg-3 has nbrs of deg 1,1,2)  
   edges: {0-1, 1-2, 2-3, 3-4, 3-5}
   deg seq: (1,2,2,3,1,1)

4. Spider/fork-star: one vertex deg 4, one deg 2
   edges: {0-1, 1-2, 1-3, 1-4, 4-5}
   deg seq: (1,4,1,1,2,1)

5. Double star S(3,3): two vertices of deg 3
   edges: {0-1, 1-2, 1-3, 2-4, 2-5}
   deg seq: (1,3,3,1,1,1)

6. K_{1,5}: star
   edges: {0-1, 0-2, 0-3, 0-4, 0-5}
   deg seq: (5,1,1,1,1,1)

Strategy: Build L^k(K_6) for k=5..10, count fiber elements for each tree type.
coeff_k(tau) = (# elements in L^k(K_6) with base type tau) / (# labeled copies of tau in K_6)

But actually by universality we just need the total count in K_6.
Wait - the coefficient is defined so that g(G,k) = sum_tau coeff_k(tau) * count(tau, G).
For G = K_n, count(tau, K_n) = (# labeled copies of tau in K_n) / |Aut(tau)|... 
No: count(tau, G) = # of subgraphs of G isomorphic to tau (as unlabeled subgraphs).

Actually in our fiber decomposition:
gamma_k(K_n) = sum_tau coeff_k(tau) * sub(tau, K_n)
where sub(tau, K_n) = number of edge-subsets of K_n isomorphic to tau.

And coeff_k(tau) is the number of grade-k elements whose base edges form a 
SPECIFIC labeled copy of tau. By universality this is the same for every copy.

So: total elements of type tau at grade k = coeff_k(tau) * sub(tau, K_n)
=> coeff_k(tau) = (total elements of type tau) / sub(tau, K_n)

sub(tau, K_n) = n! / |Aut(tau)| for trees on n vertices in K_n.

For efficiency:
- We only need to count how many grade-k elements have each base type
- We DON'T need adjacency at each grade
- But we do need base edge sets, which requires tracking through line graph iterations

Optimization: represent each vertex by its base edge frozenset.
At each grade, a new vertex = pair of adjacent vertices from previous grade.
Its base = union of parents' bases.
We only need to count by base type, but to know adjacency we need the structure.

Actually we DO need adjacency to build the next level. The line graph operation
requires knowing which vertices are adjacent.

Let me think about memory optimization. L^10(K_6) could be enormous.
L^1(K_6) = 15
L^2 = 60
L^3 = 420  
L^4 = 5460
L^5 = 136500
L^6 = ?

L^6 will be huge. Let's estimate: avg degree of L^5 is 2*136500*d/2... 
Actually |E(L^k)| = sum_{v in L^{k-1}} C(deg(v), 2).
For L^5(K_6) with 136500 vertices, if avg degree ~ 20, then 
|E| ~ 136500 * 10 = 1.3M, so L^6 has ~1.3M vertices.

That might be feasible but tight. Let's try and see.

Key optimization: we don't need to store full adjacency lists if we process 
level by level and only keep two levels in memory at once.

But we need base edge sets for ALL vertices at the current level to classify 
them. For large levels, storing 136500 frozensets is fine, but 1.3M might 
need care.

Alternative: use numpy/scipy for the adjacency and batch process.
Or: use compressed representations (base edge sets as bitmasks).

Let me use bitmask representation: K_6 has 15 edges, so base sets fit in 
a 16-bit integer. This is very compact.
"""

import time
import sys
from collections import defaultdict
import numpy as np

def edge_index(i, j, n):
    """Map edge (i,j) with i<j to index in K_n."""
    # Standard ordering: (0,1)=0, (0,2)=1, ..., (0,n-1)=n-2, (1,2)=n-1, ...
    if i > j: i, j = j, i
    return i * n - i * (i + 1) // 2 + (j - i - 1)

def edges_of_kn(n):
    """List all edges of K_n as (i,j) pairs."""
    return [(i, j) for i in range(n) for j in range(i+1, n)]

def base_to_edge_list(base_mask, edge_list):
    """Convert bitmask to list of edges."""
    edges = []
    for i in range(len(edge_list)):
        if base_mask & (1 << i):
            edges.append(edge_list[i])
    return edges

def classify_tree_from_mask(base_mask, edge_list, n_vertices):
    """Classify a tree type from its base edge bitmask. Returns type index 0-5 or -1."""
    edges = base_to_edge_list(base_mask, edge_list)
    ne = len(edges)
    if ne != 5:
        return -1
    
    verts = set()
    for u, v in edges:
        verts.add(u); verts.add(v)
    nv = len(verts)
    if nv != 6:
        return -1  # not a tree on 6 vertices
    
    # Check connectivity and compute degree sequence
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)
    
    # Quick connectivity check
    start = next(iter(verts))
    visited = {start}
    stack = [start]
    while stack:
        v = stack.pop()
        for u in adj[v]:
            if u not in visited:
                visited.add(u)
                stack.append(u)
    if len(visited) != nv:
        return -1
    
    deg_seq = tuple(sorted(adj[v].__len__() for v in verts))
    
    # Classify by degree sequence
    if deg_seq == (1, 1, 2, 2, 2, 2):
        return 0  # P_6
    elif deg_seq == (1, 1, 1, 1, 1, 5):
        return 5  # K_{1,5}
    elif deg_seq == (1, 1, 1, 1, 3, 3):
        return 4  # Double star
    elif deg_seq == (1, 1, 1, 1, 2, 4):
        return 3  # Spider
    elif deg_seq == (1, 1, 1, 2, 2, 3):
        # Two non-isomorphic trees: caterpillar A vs B
        # Distinguished by: degree-3 vertex's neighbors' degrees
        v3 = [v for v in verts if len(adj[v]) == 3][0]
        nbr_degs = sorted(len(adj[u]) for u in adj[v3])
        if nbr_degs == [1, 2, 2]:
            return 1  # Caterpillar A
        elif nbr_degs == [1, 1, 2]:
            return 2  # Caterpillar B
        else:
            return -1
    else:
        return -1

def count_labeled_copies(n):
    """Count sub(tau, K_n) for each of the 6 tree types on 6 vertices in K_6.
    sub(tau, K_n) = n! / |Aut(tau)|"""
    # For n=6, K_6:
    # |Aut| values:
    # P_6: 2 (reflection) => 6!/2 = 360
    # Cat A: 1 (no symmetry... let me think)
    #   0-1-2-3-4, 2-5: swapping 0,1? No, 0 is leaf, 1 is deg-2 interior.
    #   Swapping 3,4? 3 is interior deg 2, 4 is leaf. No.
    #   Swapping the two branches at vertex 2: {0-1} and {5}. Not the same length.
    #   Actually: 0-1-2(-5)-3-4. At vertex 2: branches are 0-1 (length 2), 5 (length 1), 3-4 (length 2).
    #   Can swap the two length-2 branches: {0-1} <-> {3-4}. That's one non-trivial automorphism.
    #   |Aut| = 2 => 6!/2 = 360
    # Cat B: 0-1-2-3(-5)-4
    #   At vertex 3: branches are ...-2-1-0 (length 3), 5 (length 1), 4 (length 1).
    #   Can swap 4 and 5. |Aut| = 2 => 6!/2 = 360
    # Spider: 0-1(-2)(-3)-4-5
    #   At vertex 1: branches are 0 (length 1), 2 (length 1), 3 (length 1), 4-5 (length 2).
    #   Can permute the three length-1 branches: 3! = 6.
    #   |Aut| = 6 => 6!/6 = 120
    # Double star: 0-1(-2)(-3), 1 also connects... wait.
    #   0-1(-3), 1-2(-4)(-5): vertex 1 deg 3 (to 0,2,3), vertex 2 deg 3 (to 1,4,5).
    #   Swap the two leaves at vertex 1: {0,3}. That's 2! = 2.
    #   Swap the two leaves at vertex 2: {4,5}. That's 2! = 2.
    #   Swap the two stars (vertices 1 and 2) with their leaves: 2.
    #   Total |Aut| = 2 * 2 * 2 = 8? Let me verify...
    #   Actually swapping the two centers also swaps their leaf sets.
    #   Aut = (swap centers) x (permute leaves of center 1) x (permute leaves of center 2)
    #   But if we swap centers, we also need compatible leaf permutations.
    #   |Aut| = 2 * 2 * 2 = 8? Or is it (S_2 x S_2) ⋊ S_2 = ... 
    #   |Aut(S(2,2))| where S(a,b) is double star with a and b leaves.
    #   S(2,2): each center has 2 leaves. |Aut| = (2! * 2!) * 2 = 8 for S(2,2).
    #   But our double star is S(2,2) on 6 vertices (centers share an edge, each has 2 more leaves).
    #   => |Aut| = 8 => 6!/8 = 90
    # K_{1,5}: |Aut| = 5! = 120 => 6!/120 = 6
    
    # Let me just compute this directly by enumeration to be safe
    edge_list = edges_of_kn(n)
    ne = len(edge_list)
    
    from itertools import combinations
    counts = [0] * 6
    for combo in combinations(range(ne), 5):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        t = classify_tree_from_mask(mask, edge_list, n)
        if t >= 0:
            counts[t] += 1
    
    return counts


def build_line_graph_level(n_prev, adj_prev, base_prev, edge_list_kn):
    """Build next line graph level. 
    
    Args:
        n_prev: number of vertices at previous level
        adj_prev: adjacency as dict of sets (or list of sets)
        base_prev: list of base bitmasks for each vertex
    
    Returns:
        n_new, adj_new, base_new
    """
    # Enumerate edges of previous level
    # For each vertex, store sorted list of neighbors for efficient iteration
    
    # Build edge list
    new_edges = []
    for v in range(n_prev):
        for u in adj_prev[v]:
            if u > v:
                new_edges.append((v, u))
    
    n_new = len(new_edges)
    base_new = [base_prev[v] | base_prev[u] for v, u in new_edges]
    
    # Build adjacency: two new vertices are adjacent if they share an endpoint
    # Use incidence lists
    incident = [[] for _ in range(n_prev)]
    for idx, (v, u) in enumerate(new_edges):
        incident[v].append(idx)
        incident[u].append(idx)
    
    # Build adjacency as list of sets
    adj_new = [set() for _ in range(n_new)]
    for v_prev in range(n_prev):
        inc = incident[v_prev]
        for i in range(len(inc)):
            for j in range(i + 1, len(inc)):
                a, b = inc[i], inc[j]
                adj_new[a].add(b)
                adj_new[b].add(a)
    
    return n_new, adj_new, base_new


def main():
    n = 6  # vertices, so trees have 5 edges
    edge_list = edges_of_kn(n)
    ne = len(edge_list)  # 15
    
    tree_names = ["P_6", "Cat_A", "Cat_B", "Spider", "DblStar", "K_{1,5}"]
    n_types = 6
    
    print(f"=== Coefficient matrix for trees on {n} vertices in K_{n} ===\n")
    
    # Count labeled copies
    print("Counting labeled copies of each tree type in K_6...")
    t0 = time.time()
    sub_counts = count_labeled_copies(n)
    print(f"  sub(tau, K_6): {list(zip(tree_names, sub_counts))} ({time.time()-t0:.1f}s)")
    print(f"  Total tree edge-subsets: {sum(sub_counts)}")
    
    # Build L^1(K_6)
    print(f"\nBuilding iterated line graphs...")
    
    # Grade 1: vertices = edges of K_6
    adj1 = [set() for _ in range(ne)]
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
    
    # We need grades 5 through 10 (6 grades for 6 types)
    # Build up to grade 10
    max_grade = 10
    
    coeffs = np.zeros((n_types, max_grade - 4), dtype=np.int64)  # rows=types, cols=grades 5..10
    
    for grade in range(2, max_grade + 1):
        t0 = time.time()
        cur_n, cur_adj, cur_base = build_line_graph_level(cur_n, cur_adj, cur_base, edge_list)
        elapsed = time.time() - t0
        print(f"  L^{grade}(K_{n}): {cur_n} vertices ({elapsed:.1f}s)", flush=True)
        
        if grade >= 5:
            # Count fiber elements by type
            type_counts = [0] * n_types
            for v in range(cur_n):
                t = classify_tree_from_mask(cur_base[v], edge_list, n)
                if t >= 0:
                    type_counts[t] += 1
            
            col = grade - 5
            for t in range(n_types):
                if sub_counts[t] > 0:
                    assert type_counts[t] % sub_counts[t] == 0, \
                        f"Non-integer coefficient at grade {grade}, type {tree_names[t]}: {type_counts[t]}/{sub_counts[t]}"
                    coeffs[t][col] = type_counts[t] // sub_counts[t]
            
            print(f"    Fiber counts: {list(zip(tree_names, type_counts))}")
            print(f"    Coefficients: {list(zip(tree_names, coeffs[:, col].tolist()))}")
        
        # Memory check
        mem_mb = sys.getsizeof(cur_adj) / 1e6
        print(f"    (approx adj mem: {mem_mb:.0f} MB)", flush=True)
        
        # Check if we should bail
        if cur_n > 5_000_000:
            print(f"    Too large, stopping at grade {grade}")
            break
    
    # Print coefficient matrix
    print(f"\n=== Coefficient Matrix (rows=types, cols=grades 5..{4+coeffs.shape[1]}) ===")
    header = "".ljust(12) + "".join(f"k={k}".rjust(12) for k in range(5, 5 + coeffs.shape[1]))
    print(header)
    for t in range(n_types):
        row = tree_names[t].ljust(12) + "".join(f"{coeffs[t][c]}".rjust(12) for c in range(coeffs.shape[1]))
        print(row)
    
    # Compute rank
    # Use the submatrix with as many columns as we have
    n_cols = coeffs.shape[1]
    n_rows = n_types
    
    # Transpose: we want types as columns, grades as rows
    M = coeffs.T  # shape (n_grades, n_types)
    
    print(f"\n=== Rank analysis ===")
    print(f"Matrix shape: {M.shape} (grades x types)")
    
    # Use float64 for rank computation
    Mf = M.astype(np.float64)
    rank = np.linalg.matrix_rank(Mf)
    print(f"Rank: {rank} (out of {n_types} types)")
    
    if rank == n_types:
        print("FULL RANK — coefficient vectors are linearly independent! ✓")
    else:
        print(f"RANK DEFICIENT — rank {rank} < {n_types}")
    
    # Also compute singular values for numerical confidence
    sv = np.linalg.svd(Mf, compute_uv=False)
    print(f"Singular values: {sv}")
    print(f"Condition number: {sv[0]/sv[-1]:.2e}" if sv[-1] > 0 else "Infinite condition number")
    
    # Check rank at each grade
    print(f"\nCumulative rank by grade:")
    for k in range(1, n_cols + 1):
        r = np.linalg.matrix_rank(M[:k, :].astype(np.float64))
        print(f"  Grades 5..{4+k}: rank {r}")

if __name__ == "__main__":
    main()