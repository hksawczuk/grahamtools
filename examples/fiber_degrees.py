"""
Compute pre-quotient fiber graph Γ_k for each tree type in L^k(K_n).
Analyze degree distributions to assess regularity.

Usage: python3 fiber_degrees.py [--n 5] [--max-k 6]

Requires labels_kn.py in the same directory or parent.
"""
import argparse
import sys
import os
from collections import defaultdict, Counter

# Add parent dir to path for labels_kn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from labels_kn import (
    generate_levels_Kn_ids,
    expand_to_simple_base_edges_id,
    canon_key_bruteforce_bitset,
    iso_classes_with_stats,
)


def fiber_degree_analysis(n: int, max_k: int = 6):
    """
    For each grade k and each tree type (fiber) in L^k(K_n):
    - build the pre-quotient fiber graph (adjacency among labeled vertices)
    - compute degree distribution
    - report regularity
    """
    print(f"Generating L^k(K_{n}) for k=0..{max_k}...")
    V_by_level, ep = generate_levels_Kn_ids(n, max_k, prune_cycles=True)

    for k in range(2, max_k + 1):
        Vk = V_by_level.get(k, [])
        if not Vk:
            print(f"\n  Grade {k}: no tree vertices")
            continue

        print(f"\n{'='*70}")
        print(f"  Grade {k}: {len(Vk)} tree vertices in L^{k}(K_{n})")
        print(f"{'='*70}")

        # ── Group vertices by canonical tree type ──
        vertex_to_key = {}
        key_to_vertices = defaultdict(list)

        for v in Vk:
            edges = expand_to_simple_base_edges_id(v, k, ep)
            key = canon_key_bruteforce_bitset(edges, n)
            vertex_to_key[v] = key
            key_to_vertices[key].append(v)

        # Get iso class stats for naming
        iso_stats = iso_classes_with_stats(Vk, n, k, ep)
        key_to_info = {s['key']: s for s in iso_stats}

        # ── Build adjacency at grade k ──
        # Two vertices are adjacent iff they share a grade-(k-1) endpoint
        ep_k = ep[k]
        inc = defaultdict(list)  # grade-(k-1) vertex -> list of grade-k vertices
        for v in Vk:
            a, b = ep_k[v]
            inc[a].append(v)
            inc[b].append(v)

        # Adjacency
        adj = defaultdict(set)
        for parent, children in inc.items():
            for i in range(len(children)):
                for j in range(i + 1, len(children)):
                    adj[children[i]].add(children[j])
                    adj[children[j]].add(children[i])

        # ── For each fiber, compute intra-fiber degree distribution ──
        for key in sorted(key_to_vertices, key=lambda k: -len(key_to_vertices[k])):
            verts = key_to_vertices[key]
            info = key_to_info.get(key, {})
            orbit_sz = info.get('orbit', '?')
            fsize = len(verts)

            # Intra-fiber degrees
            fiber_set = set(verts)
            intra_degrees = []
            for v in verts:
                d = sum(1 for u in adj.get(v, set()) if u in fiber_set)
                intra_degrees.append(d)

            # Inter-fiber degrees (edges to other fibers)
            inter_degrees = []
            for v in verts:
                d = sum(1 for u in adj.get(v, set()) if u not in fiber_set)
                inter_degrees.append(d)

            # Full degrees
            full_degrees = [intra_degrees[i] + inter_degrees[i]
                            for i in range(len(verts))]

            # Degree distribution
            intra_counter = Counter(intra_degrees)
            inter_counter = Counter(inter_degrees)
            full_counter = Counter(full_degrees)

            is_regular = len(intra_counter) == 1
            n_distinct = len(intra_counter)

            print(f"\n  Fiber key={key}, orbit_sz={orbit_sz}, "
                  f"|fiber|={fsize}, #orbits={fsize // orbit_sz if orbit_sz != '?' else '?'}")

            if is_regular:
                d = intra_degrees[0]
                print(f"    Intra-fiber: REGULAR, degree = {d}")
            else:
                print(f"    Intra-fiber: NOT regular, {n_distinct} distinct degrees")
                for d in sorted(intra_counter):
                    print(f"      deg {d}: {intra_counter[d]} vertices")

            if len(full_counter) <= 5:
                print(f"    Full degree distribution: "
                      f"{dict(sorted(full_counter.items()))}")
            else:
                degs = sorted(full_degrees)
                print(f"    Full degree: min={degs[0]}, max={degs[-1]}, "
                      f"mean={sum(degs)/len(degs):.1f}, "
                      f"distinct={len(full_counter)}")

            # Also report inter-fiber summary
            if inter_degrees:
                print(f"    Inter-fiber: min={min(inter_degrees)}, "
                      f"max={max(inter_degrees)}, "
                      f"mean={sum(inter_degrees)/len(inter_degrees):.1f}")

        # ── Summary: total edges in each fiber graph ──
        print(f"\n  --- Summary ---")
        for key in sorted(key_to_vertices, key=lambda k: -len(key_to_vertices[k])):
            verts = key_to_vertices[key]
            fiber_set = set(verts)
            e_intra = sum(sum(1 for u in adj.get(v, set()) if u in fiber_set)
                          for v in verts) // 2
            e_inter = sum(sum(1 for u in adj.get(v, set()) if u not in fiber_set)
                          for v in verts) // 2
            info = key_to_info.get(key, {})
            print(f"    key={key:>6d}: |V|={len(verts):>6d}, "
                  f"E_intra={e_intra:>8d}, E_inter={e_inter:>8d}, "
                  f"ratio={e_intra/(e_intra+e_inter):.3f}" if e_intra+e_inter > 0
                  else f"    key={key:>6d}: |V|={len(verts):>6d}, isolated")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=5)
    parser.add_argument('--max-k', type=int, default=6)
    args = parser.parse_args()

    fiber_degree_analysis(args.n, args.max_k)


if __name__ == '__main__':
    main()