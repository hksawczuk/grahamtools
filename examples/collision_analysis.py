#!/usr/bin/env python3
"""
Enumerate connected bipartite graphs using nauty's geng,
analyze bipartition collisions.

Key insight: count(τ, K_{n,m}) = (n^↓a · m^↓b + n^↓b · m^↓a) / |Aut(τ)|
Since the numerator depends only on (a,b), columns for types with the
same bipartition are PROPORTIONAL across all K_{n,m}. So the true
collision condition is: same bipartition (a,b), regardless of |Aut|.

Requires: nauty (geng + dreadnaut), networkx

Usage: python3 collision_analysis.py [max_edges]
"""

import sys
import subprocess
from collections import defaultdict
import networkx as nx

from grahamtools.utils.automorphisms import aut_size_edges


# ============================================================
#  Enumeration via geng
# ============================================================

def enumerate_via_nauty(max_edges):
    """Use geng -cb to enumerate connected bipartite graphs."""
    all_graphs = []

    for ne in range(1, max_edges + 1):
        count = 0
        for nv in range(2, ne + 2):
            result = None
            for cmd in ["geng", "nauty-geng"]:
                try:
                    result = subprocess.run(
                        [cmd, "-cb", "-q", str(nv), f"{ne}:{ne}"],
                        capture_output=True, text=True, timeout=60
                    )
                    break
                except FileNotFoundError:
                    continue

            if result is None:
                print("ERROR: geng not found")
                sys.exit(1)

            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                G = nx.from_graph6_bytes(line.encode())

                if not nx.is_bipartite(G):
                    continue
                parts = nx.bipartite.sets(G)
                a, b = len(parts[0]), len(parts[1])
                if a > b:
                    a, b = b, a

                aut = aut_size_edges(
                    [(min(u, v), max(u, v)) for u, v in G.edges()], nv
                )
                is_tree = (ne == nv - 1)

                # Name
                deg_seq = sorted([G.degree(v) for v in G.nodes()], reverse=True)
                if is_tree:
                    if all(d <= 2 for d in deg_seq):
                        name = f"P{nv}"
                    elif deg_seq[0] == ne:
                        name = f"K1_{ne}"
                    else:
                        ds = "".join(str(d) for d in deg_seq)
                        name = f"T{nv}({ds})"
                elif ne == nv and all(d == 2 for d in deg_seq):
                    name = f"C{nv}"
                elif ne == a * b:
                    name = f"K{a},{b}"
                else:
                    ds = "".join(str(d) for d in deg_seq)
                    name = f"B{nv}({ds})"

                # Deduplicate names within same edge count
                existing = [g for g in all_graphs if g[6] == name and g[3] == ne]
                if existing:
                    name = f"{name}_{chr(ord('a') + len(existing))}"

                all_graphs.append((line, G, nv, ne, (a, b), aut, name, is_tree))
                count += 1

        n_trees = sum(1 for g in all_graphs if g[3] == ne and g[7])
        print(f"  {ne}-edge: {count} types ({n_trees} trees, {count - n_trees} non-trees)")

    return all_graphs


# ============================================================
#  Main
# ============================================================

def main():
    max_edges = int(sys.argv[1]) if len(sys.argv) > 1 else 7

    print(f"Enumerating connected bipartite graphs with ≤ {max_edges} edges")
    print(f"  (using nauty)\n")
    graphs = enumerate_via_nauty(max_edges)
    graphs.sort(key=lambda g: (g[3], g[2], g[6]))

    print(f"\nTotal: {len(graphs)} types")

    # ── Collision analysis by bipartition (a,b) ──
    print(f"\n{'='*70}")
    print(f"  Collision analysis by bipartition (a,b)")
    print(f"  count(τ, K_{{n,m}}) ∝ n^↓a · m^↓b + n^↓b · m^↓a")
    print(f"  → same (a,b) ⟹ proportional columns ⟹ indistinguishable")
    print(f"{'='*70}")

    bip_groups = defaultdict(list)
    for g in graphs:
        bip_groups[g[4]].append(g)

    n_bips = len(bip_groups)

    print(f"\n  {len(graphs)} types, {n_bips} distinct bipartitions")
    print(f"  Max rank with K_{{n,m}} alone: {n_bips}")
    print(f"  Rank deficit: {len(graphs) - n_bips}")

    for ne in range(1, max_edges + 1):
        types_ne = [g for g in graphs if g[3] == ne]
        bips_ne = defaultdict(list)
        for g in types_ne:
            bips_ne[g[4]].append(g)

        collisions_ne = {b: gs for b, gs in bips_ne.items() if len(gs) > 1}
        n_unique = sum(1 for gs in bips_ne.values() if len(gs) == 1)
        n_colliding = sum(len(gs) for gs in collisions_ne.values())

        print(f"\n  --- {ne}-edge: {len(types_ne)} total, "
              f"{n_unique} unique bipartitions, {n_colliding} in collisions ---")

        if collisions_ne:
            for bip, gs in sorted(collisions_ne.items()):
                labels = []
                for g in gs:
                    t = "T" if g[7] else " "
                    labels.append(f"{g[6]}[{t}|Aut|={g[5]}]")
                print(f"    bipart={bip}: {', '.join(labels)}")

    # ── Cumulative rank budget ──
    print(f"\n{'='*70}")
    print(f"  Cumulative rank budget per grade")
    print(f"{'='*70}")

    print(f"\n  {'grade':>5s} {'types':>7s} {'bips':>6s} {'deficit':>8s}"
          f" {'trees':>7s} {'t_bips':>7s} {'t_def':>6s}")
    for k in range(1, max_edges + 1):
        types_k = [g for g in graphs if g[3] <= k]
        bips_k = set(g[4] for g in types_k)
        trees_k = [g for g in types_k if g[7]]
        tree_bips_k = set(g[4] for g in trees_k)

        print(f"  k={k:>3d} {len(types_k):>7d} {len(bips_k):>6d}"
              f" {len(types_k)-len(bips_k):>8d}"
              f" {len(trees_k):>7d} {len(tree_bips_k):>7d}"
              f" {len(trees_k)-len(tree_bips_k):>6d}")

    # ── Tree-only collisions ──
    print(f"\n{'='*70}")
    print(f"  Tree-only collision summary")
    print(f"{'='*70}")

    tree_types = [g for g in graphs if g[7]]
    tree_bips = defaultdict(list)
    for g in tree_types:
        tree_bips[g[4]].append(g)

    tree_collisions = {b: gs for b, gs in tree_bips.items() if len(gs) > 1}

    print(f"\n  {len(tree_types)} tree types, {len(tree_bips)} distinct tree bipartitions")
    if tree_collisions:
        print(f"\n  Tree-tree collisions (same bipartition):")
        for bip, gs in sorted(tree_collisions.items()):
            names = [f"{g[6]}({g[3]}e, |Aut|={g[5]})" for g in gs]
            print(f"    bipart={bip}: {', '.join(names)}")
        total_tree_deficit = sum(len(gs) - 1 for gs in tree_collisions.values())
        print(f"\n  Total tree rank deficit from K_{{n,m}}: {total_tree_deficit}")
        print(f"  Tree types separable by K_{{n,m}}: {len(tree_types) - total_tree_deficit}")
    else:
        print(f"  No tree-tree collisions!")

    # ── Bipartition distribution of trees ──
    print(f"\n{'='*70}")
    print(f"  Bipartition distribution of trees by edge count")
    print(f"{'='*70}")

    for ne in range(1, max_edges + 1):
        trees_ne = [g for g in tree_types if g[3] == ne]
        if not trees_ne:
            continue
        bips = defaultdict(list)
        for g in trees_ne:
            bips[g[4]].append(g)
        print(f"\n  {ne}-edge trees ({len(trees_ne)} types):")
        for bip, gs in sorted(bips.items()):
            names = [f"{g[6]}(|Aut|={g[5]})" for g in gs]
            collision = " ← COLLISION" if len(gs) > 1 else ""
            print(f"    ({bip[0]},{bip[1]}): {', '.join(names)}{collision}")


if __name__ == "__main__":
    main()