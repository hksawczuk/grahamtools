#!/usr/bin/env python3
"""
Compute Graham sequences via pure dynamic programming — no line graph iteration.

Core recursion:

    γ_k(T) = (1 - |Aut(T)|/n!)^{-1} * [
        Σ_{τ ⊊ T} (|Aut(τ)|/n!) * Σ_{S ⊆ E(τ)} (-1)^{|E(τ)|-|S|} γ_k(F[S])
        + Σ_{S ⊊ E(T)} (-1)^{|E(T)|-|S|} γ_k(F[S])
    ]

where:
  - The outer sum is over connected subtree types τ strictly contained in T
  - F[S] decomposes into connected components, each strictly smaller than T
  - γ_k of disconnected F[S] = sum of γ_k over components
  - Base case: γ_k(K2) for all k (single edge → 1 vertex, no edges → dies)

Every γ_k on the RHS involves strictly fewer edges, so the recursion terminates.

Usage:
    python3 graham_dp.py [--verify] [--max-k K]
"""

import sys
import time
from math import factorial
from collections import defaultdict
from itertools import combinations, permutations
from fractions import Fraction

from grahamtools.utils.automorphisms import aut_size_edges
from grahamtools.utils.connectivity import is_connected_edges, connected_components_edges
from grahamtools.utils.linegraph_edgelist import line_graph_edgelist, gamma_sequence_edgelist


# ============================================================
#  Canonical forms for trees
# ============================================================

def edges_to_adj(edges):
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return dict(adj)


def canonical_tree(edges):
    """Canonical string for a connected tree given by edge list."""
    if not edges:
        return "empty"

    adj = defaultdict(list)
    verts = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        verts.add(u)
        verts.add(v)

    vertices = sorted(verts)
    n = len(vertices)

    if n == 1:
        return "()"
    if n == 2:
        return "(())"

    deg = {v: len(adj[v]) for v in vertices}
    leaves = [v for v in vertices if deg[v] <= 1]
    removed = set()
    remaining = n

    while remaining > 2:
        new_leaves = []
        for v in leaves:
            removed.add(v)
            remaining -= 1
            for u in adj[v]:
                if u not in removed:
                    deg[u] -= 1
                    if deg[u] == 1:
                        new_leaves.append(u)
        leaves = new_leaves

    centers = [v for v in vertices if v not in removed]

    def rc(root, parent):
        ch = sorted(rc(u, root) for u in adj[root] if u != parent)
        return "(" + "".join(ch) + ")"

    if len(centers) == 1:
        return rc(centers[0], None)
    else:
        c0, c1 = centers[0], centers[1]
        ch0 = sorted(rc(u, c0) for u in adj[c0] if u != c1)
        ch1 = sorted(rc(u, c1) for u in adj[c1] if u != c0)
        opt1 = (tuple(ch0), tuple(ch1))
        opt2 = (tuple(ch1), tuple(ch0))
        return "E" + str(min(opt1, opt2))


def compute_aut_size(edges):
    """Compute |Aut(T)| using grahamtools."""
    if not edges:
        return 1
    n = max(v for e in edges for v in e) + 1
    return aut_size_edges(edges, n)


def connected_components(edges):
    """Return list of edge-lists, one per connected component."""
    if not edges:
        return []
    return [comp_edges for _, comp_edges in connected_components_edges(edges)]


def is_connected(edges):
    return is_connected_edges(edges)


# ============================================================
#  Core DP: compute γ_k and coeff_k for all tree types
# ============================================================

class GrahamDP:
    """Dynamic programming computation of Graham sequences and fiber coefficients.

    Maintains a cache of:
      - gamma_k[canon][k] = γ_k(τ) for each tree type
      - coeff_k[canon][k] = coeff_k(τ)
      - aut[canon] = |Aut(τ)|
      - representative[canon] = edge list

    The recursion processes trees in order of increasing edge count.
    """

    def __init__(self, n=None):
        """n = number of vertices of ambient K_n. If None, uses universality
        (coefficients are independent of n for large enough n)."""
        self.n_ambient = n
        self.gamma = {}      # canon -> {k: value}
        self.coeff = {}      # canon -> {k: value}
        self.aut = {}        # canon -> |Aut|
        self.rep = {}        # canon -> representative edge list
        self.nedges = {}     # canon -> number of edges

        # Base case: single edge (K2)
        # L(K2) has 1 vertex (the single edge), 0 edges → L^2 has 0 vertices
        # So γ_0 = 2, γ_1 = 1, γ_k = 0 for k >= 2
        k2_canon = "(())"
        self.gamma[k2_canon] = {k: (1 if k == 1 else 0) for k in range(1, 100)}
        self.coeff[k2_canon] = {1: 1}  # coeff_1(K2) = 1, coeff_k = 0 for k >= 2
        self.aut[k2_canon] = 2
        self.rep[k2_canon] = [(0, 1)]
        self.nedges[k2_canon] = 1

    def _n_factorial(self):
        """Return n! for the ambient complete graph.
        For universality, we can use any n larger than the tree.
        We use Fraction to keep exact arithmetic."""
        if self.n_ambient is not None:
            return Fraction(factorial(self.n_ambient))
        else:
            # Use a symbolic large n — but actually for the formula
            # we need a concrete n. Let's use n = max tree size + 1.
            raise ValueError("Must specify n_ambient or use register_tree")

    def ensure_n(self, tree_verts):
        """Make sure n_ambient is large enough."""
        if self.n_ambient is None:
            self.n_ambient = tree_verts + 1
        elif self.n_ambient < tree_verts:
            # Universality: coefficients don't depend on n for n >= tree_verts
            # But n! / |Aut| does. We need n >= number of vertices of the tree.
            # For the formula to be valid, n must be >= vertices in τ.
            # Since we're computing γ_k(T) which is independent of n,
            # we can use any n >= |V(T)|.
            self.n_ambient = tree_verts

    def gamma_k_forest(self, edges, k):
        """Compute γ_k for a possibly disconnected forest.
        Decomposes into components and sums."""
        if not edges:
            # Empty forest: no edges, so L^1 has 0 vertices
            # γ_0 = number of isolated vertices, but we don't track those
            # For k >= 1, γ_k = 0
            if k == 0:
                return 0  # or should be vertex count? See below.
            return 0

        comps = connected_components(edges)
        total = 0
        for comp_edges in comps:
            canon = canonical_tree(comp_edges)
            if canon not in self.gamma or k not in self.gamma[canon]:
                raise KeyError(f"γ_{k} not computed for {canon}")
            total += self.gamma[canon][k]
        return total

    def compute_tree(self, edges, max_k):
        """Compute γ_k and coeff_k for a tree given by edges, for k = 1..max_k.

        Assumes all strictly smaller tree types have already been processed.
        """
        canon = canonical_tree(edges)
        ne = len(edges)

        if canon in self.gamma and all(k in self.gamma[canon] for k in range(1, max_k + 1)):
            return  # already computed

        # Count vertices
        verts = set()
        for u, v in edges:
            verts.add(u)
            verts.add(v)
        nv = len(verts)

        self.ensure_n(nv)
        nfact = Fraction(factorial(self.n_ambient))

        # Compute |Aut|
        if canon not in self.aut:
            self.aut[canon] = compute_aut_size(edges)
            self.rep[canon] = edges
            self.nedges[canon] = ne

        aut_T = Fraction(self.aut[canon])

        if canon not in self.gamma:
            self.gamma[canon] = {}
        if canon not in self.coeff:
            self.coeff[canon] = {}

        # Enumerate connected subtree types τ ⊊ T and their counts
        subtree_counts = defaultdict(int)  # canon -> count
        for size in range(1, ne):
            for subset in combinations(range(ne), size):
                sub_edges = [edges[i] for i in subset]
                if is_connected(sub_edges):
                    sub_canon = canonical_tree(sub_edges)
                    subtree_counts[sub_canon] += 1

        # For each k, compute γ_k(T)
        for k in range(1, max_k + 1):
            # Term 1: Σ_{τ ⊊ T} (|Aut(τ)|/n!) * Σ_{S ⊆ E(τ)} (-1)^{|E(τ)|-|S|} γ_k(F[S])
            # This equals Σ_{τ ⊊ T} coeff_k(τ) * count(τ, T)
            # which we can compute from stored coefficients
            term1 = Fraction(0)
            for sub_canon, count in subtree_counts.items():
                c = self.coeff.get(sub_canon, {}).get(k, None)
                if c is not None:
                    term1 += Fraction(c) * count

            # Term 2: Σ_{S ⊊ E(T)} (-1)^{|E(T)|-|S|} γ_k(F[S])
            term2 = Fraction(0)
            for mask in range(0, (1 << ne) - 1):  # all proper subsets (exclude full set)
                subset_size = bin(mask).count('1')
                sign = (-1) ** (ne - subset_size)

                sub_edges = [edges[i] for i in range(ne) if mask & (1 << i)]

                if not sub_edges:
                    # Empty subset: γ_k = 0 for k >= 1
                    continue

                # Sum γ_k over connected components
                comps = connected_components(sub_edges)
                gamma_fs = Fraction(0)
                valid = True
                for comp_edges in comps:
                    comp_canon = canonical_tree(comp_edges)
                    gval = self.gamma.get(comp_canon, {}).get(k, None)
                    if gval is None:
                        valid = False
                        break
                    gamma_fs += gval
                if not valid:
                    continue

                term2 += sign * gamma_fs

            # γ_k(T) = (1 - |Aut(T)|/n!)^{-1} * [term1 + (|Aut(T)|/n!) * term2]
            prefactor_inv = Fraction(1) - aut_T / nfact
            rhs = term1 + (aut_T / nfact) * term2

            gamma_val = rhs / prefactor_inv
            assert gamma_val.denominator == 1, \
                f"γ_{k}({canon}) = {gamma_val} is not an integer!"
            self.gamma[canon][k] = int(gamma_val)

            # Now extract coeff_k(T) via Möbius inversion
            # F_k(T) = γ_k(T) + term2  (term2 is the proper-subset Möbius sum)
            F_k = self.gamma[canon][k] + int(term2)
            coeff_val = Fraction(F_k) * aut_T / nfact
            assert coeff_val.denominator == 1, \
                f"coeff_{k}({canon}) = {coeff_val} is not an integer!"
            self.coeff[canon][k] = int(coeff_val)

    def compute_all_trees_up_to(self, max_edges, max_k):
        """Enumerate and compute all tree types with up to max_edges edges."""
        # Generate trees by size using additive construction
        trees_by_size = {1: [[(0, 1)]]}

        for ne in range(2, max_edges + 1):
            prev_trees = trees_by_size.get(ne - 1, [])
            seen = set()
            new_trees = []

            for tree in prev_trees:
                verts = set()
                for u, v in tree:
                    verts.add(u)
                    verts.add(v)
                new_v = max(verts) + 1

                for attach in sorted(verts):
                    new_edges = tree + [(attach, new_v)]
                    canon = canonical_tree(new_edges)
                    if canon not in seen:
                        seen.add(canon)
                        new_trees.append(new_edges)

            trees_by_size[ne] = new_trees

        # Process in order of increasing size
        for ne in range(1, max_edges + 1):
            trees = trees_by_size.get(ne, [])
            print(f"  Processing {len(trees)} tree type(s) with {ne} edge(s)...",
                  flush=True)
            for edges in trees:
                t0 = time.time()
                self.compute_tree(edges, max_k)
                elapsed = time.time() - t0
                canon = canonical_tree(edges)
                if elapsed > 0.1:
                    print(f"    {canon[:40]:40s} ({elapsed:.2f}s)")

    def compute_gamma_for_tree(self, edges, max_k):
        """Compute γ_k for an arbitrary tree, using stored coefficients.
        This is the simple forward formula:
            γ_k(T) = Σ_{τ ⊆ T, connected} coeff_k(τ) · count(τ, T)
        """
        ne = len(edges)

        # Count all connected subtree types (including T itself)
        subtree_counts = defaultdict(int)
        for size in range(1, ne + 1):
            for subset in combinations(range(ne), size):
                sub_edges = [edges[i] for i in subset]
                if is_connected(sub_edges):
                    sub_canon = canonical_tree(sub_edges)
                    subtree_counts[sub_canon] += 1

        result = {}
        for k in range(1, max_k + 1):
            total = 0
            missing = []
            for sub_canon, count in subtree_counts.items():
                c = self.coeff.get(sub_canon, {}).get(k, None)
                if c is not None:
                    total += c * count
                else:
                    missing.append(sub_canon)
            if not missing:
                result[k] = total
            else:
                result[k] = None
        return result


# ============================================================
#  Brute-force verification
# ============================================================

def gamma_bruteforce(edges, max_k, max_edge_limit=5_000_000):
    seq_list = gamma_sequence_edgelist(edges, max_k, max_edges=max_edge_limit)
    seq = {}
    for k, val in enumerate(seq_list):
        if val is not None:
            seq[k] = val
    return seq


# ============================================================
#  Main
# ============================================================

def main():
    do_verify = "--verify" in sys.argv
    max_k = 6
    max_bootstrap_edges = 6

    for i, arg in enumerate(sys.argv):
        if arg == "--max-k" and i + 1 < len(sys.argv):
            max_k = int(sys.argv[i + 1])
        if arg == "--max-edges" and i + 1 < len(sys.argv):
            max_bootstrap_edges = int(sys.argv[i + 1])

    # Use a large enough n for universality
    # n must be > max number of vertices in any tree we process
    dp = GrahamDP(n=max_bootstrap_edges + 2)

    print("=" * 60)
    print(f"  Bootstrapping coefficients for trees up to {max_bootstrap_edges} edges")
    print(f"  max_k = {max_k}, n_ambient = {dp.n_ambient}")
    print("=" * 60)

    t0 = time.time()
    dp.compute_all_trees_up_to(max_bootstrap_edges, max_k)
    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.2f}s)")

    # Print coefficient table
    print(f"\n{'=' * 60}")
    print(f"  Fiber coefficients")
    print(f"{'=' * 60}")
    for canon in sorted(dp.coeff.keys(), key=lambda c: (dp.nedges.get(c, 0), c)):
        ne = dp.nedges.get(canon, "?")
        aut = dp.aut.get(canon, "?")
        coeffs = dp.coeff[canon]
        cstr = ", ".join(f"k={k}:{v}" for k, v in sorted(coeffs.items()) if v != 0)
        if cstr:
            print(f"  [{ne}e, aut={aut}] {canon[:45]:45s} {cstr}")

    # ---- Verification ----
    if do_verify:
        print(f"\n{'=' * 60}")
        print(f"  Verification: DP γ_k vs brute-force line graph iteration")
        print(f"{'=' * 60}")

        for canon in sorted(dp.gamma.keys(), key=lambda c: (dp.nedges.get(c, 0), c)):
            edges = dp.rep[canon]
            ne = dp.nedges[canon]
            bf = gamma_bruteforce(edges, max_k)

            all_ok = True
            for k in range(1, max_k + 1):
                dp_val = dp.gamma[canon].get(k)
                bf_val = bf.get(k)
                if dp_val is not None and bf_val is not None and dp_val != bf_val:
                    all_ok = False

            status = "✓" if all_ok else "✗"
            dp_seq = [dp.gamma[canon].get(k, "—") for k in range(1, max_k + 1)]
            bf_seq = [bf.get(k, "—") for k in range(1, max_k + 1)]
            print(f"  {status} [{ne}e] {canon[:35]:35s}")
            print(f"       DP: {dp_seq}")
            print(f"       BF: {bf_seq}")




if __name__ == "__main__":
    main()