"""
Compute fiber coefficients for all tree types in L^k(K_n).

At each grade k, vertices of L^k(K_n) that are trees decompose into fibers
by their base-edge tree type τ. The quotient under S_n groups labeled
vertices into orbits.

For each tree type τ:
  fiber_size(τ, k) = total labeled vertices with base-edge type τ
  count(τ, K_n)    = n! / |Aut(τ)|  = number of labeled copies of τ in K_n
  coeff_k(τ)       = fiber_size / count(τ, K_n)   [universal coefficient]
  #orbits           = coeff_k(τ)  (since each labeled copy contributes one orbit)

Usage:
  python3 fiber_coefficients.py --n 5 --max-k 7
  python3 fiber_coefficients.py --n 6 --max-k 5

Requires labels_kn.py in the same directory or parent.
"""
import argparse
import sys
import os
import math
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from labels_kn import (
    generate_levels_Kn_ids,
    expand_to_simple_base_edges_id,
    canon_key_bruteforce_bitset,
    aut_size_via_color_classes,
)


def tree_name(edges, n):
    """Give a human-readable name to a tree from its edge list."""
    if not edges:
        return "K1"
    
    nedges = len(edges)
    nv = nedges + 1  # tree on nedges edges has nedges+1 vertices
    
    # Compute degree sequence
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    # Only count vertices actually in the tree
    active_degs = sorted([d for d in deg if d > 0], reverse=True)
    max_d = active_degs[0] if active_degs else 0
    
    # Identify common types
    if nedges == 1:
        return "K2"
    
    # Check if path
    if max_d <= 2:
        return f"P{nv}"
    
    # Check if star
    if active_degs.count(1) == nedges and max_d == nedges:
        return f"K1,{nedges}"
    
    # Count vertices of each degree
    deg_counts = defaultdict(int)
    for d in active_degs:
        deg_counts[d] += 1
    
    # Double star: exactly 2 vertices of degree > 1, both > 1, rest leaves
    high_deg = [(d, c) for d, c in deg_counts.items() if d > 1]
    if len(high_deg) == 1 and high_deg[0][0] > 2:
        d, c = high_deg[0]
        if c == 1:
            return f"K1,{d}"  # star (should have been caught above)
    
    # Caterpillar or general: use degree sequence as name
    ds_str = "".join(str(d) for d in active_degs)
    return f"T{nv}[{ds_str}]"


def fiber_analysis(n, max_k):
    """Compute fiber decomposition at each grade."""
    
    print(f"Generating L^k(K_{n}) for k=0..{max_k} (trees only)...")
    t0 = time.time()
    V_by_level, ep = generate_levels_Kn_ids(n, max_k, prune_cycles=True)
    t1 = time.time()
    print(f"  Done in {t1-t0:.1f}s")
    
    for k in range(max_k + 1):
        print(f"  Grade {k}: {len(V_by_level.get(k, []))} tree vertices")
    
    # Track all fiber data across grades
    # fiber_key -> {name, count_in_Kn, coeff_by_grade}
    all_fibers = {}
    
    for k in range(1, max_k + 1):
        Vk = V_by_level.get(k, [])
        if not Vk:
            continue
        
        t0 = time.time()
        
        # Group vertices by canonical base-edge type
        buckets = {}  # key -> {edges_rep, freq}
        for v in Vk:
            edges = expand_to_simple_base_edges_id(v, k, ep)
            key = canon_key_bruteforce_bitset(edges, n)
            if key not in buckets:
                buckets[key] = {"edges": edges, "freq": 0}
            buckets[key]["freq"] += 1
        
        t1 = time.time()
        
        # Process each fiber
        for key, bucket in buckets.items():
            edges = bucket["edges"]
            freq = bucket["freq"]
            aut = aut_size_via_color_classes(edges, n)
            orbit_sz = math.factorial(n) // aut
            count_in_Kn = orbit_sz  # number of labeled copies of τ in K_n
            coeff = freq // orbit_sz
            n_orbits = freq // orbit_sz
            
            if key not in all_fibers:
                name = tree_name(edges, n)
                all_fibers[key] = {
                    "name": name,
                    "edges": edges,
                    "aut": aut,
                    "orbit_sz": orbit_sz,
                    "count_in_Kn": count_in_Kn,
                    "nedges": len(edges),
                    "coeff": {},
                    "fiber_sz": {},
                    "n_orbits": {},
                }
            
            all_fibers[key]["coeff"][k] = coeff
            all_fibers[key]["fiber_sz"][k] = freq
            all_fibers[key]["n_orbits"][k] = n_orbits
        
        if t1 - t0 > 0.5:
            print(f"    Grade {k}: classified {len(Vk)} vertices into "
                  f"{len(buckets)} fibers in {t1-t0:.1f}s")
    
    return all_fibers


def print_results(all_fibers, n, max_k):
    """Print fiber coefficient table."""
    
    # Sort fibers by number of edges (grade of first appearance), then by size
    sorted_fibers = sorted(
        all_fibers.values(),
        key=lambda f: (f["nedges"], -max(f["fiber_sz"].values(), default=0))
    )
    
    # ── Per-fiber detail ──
    print(f"\n{'='*80}")
    print(f"  Fiber coefficients for L^k(K_{n})")
    print(f"{'='*80}")
    
    for fiber in sorted_fibers:
        name = fiber["name"]
        ne = fiber["nedges"]
        aut = fiber["aut"]
        orbit = fiber["orbit_sz"]
        
        print(f"\n  {name}  ({ne} edges, |Aut|={aut}, orbit_sz={orbit}, "
              f"count_in_K{n}={orbit})")
        
        print(f"    {'k':>4s} {'coeff_k':>12s} {'fiber_sz':>12s} "
              f"{'#orbits':>10s} {'ratio':>10s}")
        
        prev_coeff = None
        for k in sorted(fiber["coeff"]):
            c = fiber["coeff"][k]
            fs = fiber["fiber_sz"][k]
            no = fiber["n_orbits"][k]
            ratio = c / prev_coeff if prev_coeff and prev_coeff > 0 else None
            print(f"    {k:>4d} {c:>12d} {fs:>12d} {no:>10d} "
                  f"{f'{ratio:.2f}' if ratio else '—':>10s}")
            prev_coeff = c
    
    # ── Summary table: coefficients across grades ──
    print(f"\n{'='*80}")
    print(f"  Summary: coeff_k(τ) for all fibers")
    print(f"{'='*80}")
    
    # Collect all grades
    all_grades = set()
    for f in sorted_fibers:
        all_grades.update(f["coeff"].keys())
    grades = sorted(all_grades)
    
    # Header
    hdr = f"  {'Tree':>16s} {'|e|':>4s} {'|Aut|':>5s}"
    for k in grades:
        hdr += f" {'k='+str(k):>10s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    
    for fiber in sorted_fibers:
        row = f"  {fiber['name']:>16s} {fiber['nedges']:>4d} {fiber['aut']:>5d}"
        for k in grades:
            c = fiber["coeff"].get(k, None)
            if c is not None:
                row += f" {c:>10d}"
            else:
                row += f" {'':>10s}"
        print(row)
    
    # ── Growth rate table ──
    print(f"\n{'='*80}")
    print(f"  Growth rates: coeff_k / coeff_{{k-1}}")
    print(f"{'='*80}")
    
    hdr = f"  {'Tree':>16s}"
    for k in grades[1:]:
        hdr += f" {'k='+str(k):>10s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    
    for fiber in sorted_fibers:
        row = f"  {fiber['name']:>16s}"
        for k in grades[1:]:
            c = fiber["coeff"].get(k, None)
            cp = fiber["coeff"].get(k - 1, None)
            if c is not None and cp is not None and cp > 0:
                row += f" {c/cp:>10.2f}"
            else:
                row += f" {'':>10s}"
        print(row)
    
    # ── Fraction of total vertices ──
    print(f"\n{'='*80}")
    print(f"  Fiber fractions: fiber_sz / |V(L^k)|")
    print(f"{'='*80}")
    
    hdr = f"  {'Tree':>16s}"
    for k in grades:
        hdr += f" {'k='+str(k):>10s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    
    # Total tree vertices at each grade
    totals = {}
    for k in grades:
        totals[k] = sum(f["fiber_sz"].get(k, 0) for f in sorted_fibers)
    
    # Print totals row first
    row = f"  {'TOTAL':>16s}"
    for k in grades:
        row += f" {totals[k]:>10d}"
    print(row)
    print("  " + "-" * (len(hdr) - 2))
    
    for fiber in sorted_fibers:
        row = f"  {fiber['name']:>16s}"
        for k in grades:
            fs = fiber["fiber_sz"].get(k, 0)
            if totals[k] > 0:
                frac = fs / totals[k]
                row += f" {frac:>10.4f}"
            else:
                row += f" {'':>10s}"
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Compute fiber coefficients for tree types in L^k(K_n)")
    parser.add_argument('--n', type=int, default=5,
                        help='Complete graph K_n (default: 5)')
    parser.add_argument('--max-k', type=int, default=7,
                        help='Maximum grade (default: 7)')
    args = parser.parse_args()
    
    all_fibers = fiber_analysis(args.n, args.max_k)
    print_results(all_fibers, args.n, args.max_k)


if __name__ == '__main__':
    main()