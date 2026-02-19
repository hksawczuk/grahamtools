"""
Experiment 2: Cross-class growth rate comparison.

Runs fiber analysis for K_5, K_6, and K_7, then merges the coefficient
data to compare growth rates across edge-count classes. Since coefficients
are universal, we use K_5 for deep data on 4-edge trees, K_6 for 5-edge
trees, and K_7 for 6-edge trees — combined into one table.

This directly addresses the key question: are coefficient growth rates
distinct across ALL tree types, not just within the same edge-count class?

Usage:
  python3 exp2_cross_class.py
  python3 exp2_cross_class.py --deep    # push K_5 to grade 8

Output: unified coefficient table and growth rate ranking.
"""
import argparse
import math
import time
import json
from collections import defaultdict

from grahamtools.kn.levels import generate_levels_Kn_ids
from grahamtools.kn.expand import expand_to_simple_base_edges_id
from grahamtools.kn.classify import canon_key, _canon_key_cache as _canon_cache
from grahamtools.utils.automorphisms import aut_size_edges
from grahamtools.utils.naming import tree_name


def fiber_analysis(n, max_k):
    """Run fiber analysis for K_n up to grade max_k. Returns dict of fiber data."""
    print(f"\n  K_{n}, grades 1..{max_k}:")
    _canon_cache.clear()

    t0 = time.time()
    V, ep = generate_levels_Kn_ids(n, max_k, prune_cycles=True)
    print(f"    Generation: {time.time()-t0:.1f}s")
    for k in range(max_k + 1):
        nv = len(V.get(k, []))
        if nv > 0:
            print(f"    Grade {k}: {nv} vertices")

    all_fibers = {}
    for k in range(1, max_k + 1):
        Vk = V.get(k, [])
        if not Vk:
            continue

        t0 = time.time()
        buckets = {}
        for v in Vk:
            edges = expand_to_simple_base_edges_id(v, k, ep)
            key = canon_key(edges, n)
            if key not in buckets:
                buckets[key] = {"edges": edges, "freq": 0}
            buckets[key]["freq"] += 1
        elapsed = time.time() - t0

        for key, bucket in buckets.items():
            edges = bucket["edges"]
            freq = bucket["freq"]
            aut = aut_size_edges(edges, n)
            orbit_sz = math.factorial(n) // aut
            coeff = freq // orbit_sz

            if key not in all_fibers:
                all_fibers[key] = {
                    "name": tree_name(edges, n),
                    "nedges": len(edges),
                    "aut": aut,
                    "orbit_sz": orbit_sz,
                    "coeff": {},
                }
            all_fibers[key]["coeff"][k] = coeff

        if elapsed > 1:
            print(f"    Grade {k}: {len(buckets)} fibers in {elapsed:.1f}s")

    return all_fibers


def merge_fibers(*fiber_dicts):
    """
    Merge fiber data from different n values.

    Since coefficients are universal, the same tree type τ gives the same
    coeff_k(τ) regardless of which K_n we compute in (as long as n > |V(τ)|).
    We merge by name, taking the longest coefficient sequence available.
    """
    merged = {}  # name -> {nedges, coeff: {k: val}, sources: [n1, n2, ...]}

    for fibers in fiber_dicts:
        for key, fdata in fibers.items():
            name = fdata["name"]
            ne = fdata["nedges"]

            # Handle duplicate names (e.g., two T6[322111] types)
            # Use (name, nedges, aut) as a more unique identifier
            uid = (name, ne, fdata["aut"])

            if uid not in merged:
                merged[uid] = {
                    "name": name,
                    "nedges": ne,
                    "aut": fdata["aut"],
                    "coeff": {},
                }

            # Merge coefficients — take all available
            for k, c in fdata["coeff"].items():
                if k not in merged[uid]["coeff"]:
                    merged[uid]["coeff"][k] = c
                else:
                    # Universality check
                    if merged[uid]["coeff"][k] != c:
                        print(f"  WARNING: universality violation for {name} at k={k}: "
                              f"{merged[uid]['coeff'][k]} vs {c}")

    return merged


def run_experiment(deep=False):
    print(f"{'='*72}")
    print(f"  Experiment 2: Cross-class growth rate comparison")
    print(f"{'='*72}")

    # Determine depth for each n
    k5_max = 8 if deep else 7
    k6_max = 6
    k7_max = 5

    print(f"\n  Plan: K_5 to grade {k5_max}, K_6 to grade {k6_max}, K_7 to grade {k7_max}")

    fibers_5 = fiber_analysis(5, k5_max)
    fibers_6 = fiber_analysis(6, k6_max)
    fibers_7 = fiber_analysis(7, k7_max)

    merged = merge_fibers(fibers_5, fibers_6, fibers_7)

    # Sort by (nedges, -max_coeff)
    sorted_uids = sorted(merged.keys(),
                         key=lambda uid: (merged[uid]["nedges"],
                                          -max(merged[uid]["coeff"].values())))

    # ── Unified coefficient table ──
    all_grades = sorted(set(g for d in merged.values() for g in d["coeff"]))
    print(f"\n{'='*72}")
    print(f"  Unified coefficient table (universality-merged)")
    print(f"{'='*72}")
    hdr = f"  {'Tree':>16s} {'|e|':>4s}"
    for k in all_grades:
        hdr += f" {'k='+str(k):>12s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for uid in sorted_uids:
        d = merged[uid]
        row = f"  {d['name']:>16s} {d['nedges']:>4d}"
        for k in all_grades:
            c = d["coeff"].get(k)
            row += f" {c:>12d}" if c is not None else f" {'':>12s}"
        print(row)

    # ── Growth rate ranking (last available ratio) ──
    print(f"\n{'='*72}")
    print(f"  Growth rate ranking (latest ratio, all tree types)")
    print(f"{'='*72}")

    rate_data = []
    for uid in sorted_uids:
        d = merged[uid]
        ks = sorted(d["coeff"].keys())
        ratios = []
        for i in range(1, len(ks)):
            cp = d["coeff"][ks[i-1]]
            c = d["coeff"][ks[i]]
            if cp > 0:
                ratios.append((ks[i], c / cp))
        if ratios:
            last_k, last_ratio = ratios[-1]
            rate_data.append((last_ratio, d["name"], d["nedges"], last_k, len(ks)))

    rate_data.sort(reverse=True)
    print(f"  {'Rank':>4s} {'Tree':>16s} {'|e|':>4s} {'ratio':>12s} {'at grade':>10s} {'#points':>8s}")
    print("  " + "-" * 60)
    for rank, (ratio, name, ne, k, npts) in enumerate(rate_data, 1):
        print(f"  {rank:>4d} {name:>16s} {ne:>4d} {ratio:>12.2f} {k:>10d} {npts:>8d}")

    # ── Are any growth rates identical? ──
    print(f"\n{'='*72}")
    print(f"  Pairwise growth rate comparison")
    print(f"{'='*72}")

    # For fibers with 2+ data points, compare all pairs
    multi_point = [(uid, merged[uid]) for uid in sorted_uids
                   if len(merged[uid]["coeff"]) >= 3]

    if len(multi_point) >= 2:
        print(f"  Comparing {len(multi_point)} fibers with 3+ data points")
        print(f"  Looking for pairs with similar growth trajectories...\n")

        for i in range(len(multi_point)):
            uid_i, di = multi_point[i]
            for j in range(i+1, len(multi_point)):
                uid_j, dj = multi_point[j]

                # Find overlapping grades
                common_k = sorted(set(di["coeff"]) & set(dj["coeff"]))
                if len(common_k) < 2:
                    continue

                # Compute ratio of coefficients at each common grade
                ratios = []
                for k in common_k:
                    ci = di["coeff"][k]
                    cj = dj["coeff"][k]
                    if cj > 0:
                        ratios.append(ci / cj)

                if len(ratios) >= 2:
                    # Check if ratio is converging (would indicate same growth rate)
                    spread = max(ratios) / min(ratios) if min(ratios) > 0 else float('inf')
                    trend = ratios[-1] / ratios[0] if ratios[0] > 0 else float('inf')

                    if spread < 2.0:  # ratios close together → possible linear dependence
                        print(f"  ⚠ {di['name']} vs {dj['name']}: "
                              f"coeff ratios = {[f'{r:.3f}' for r in ratios]}, "
                              f"spread = {spread:.3f}")
    else:
        print(f"  Need more data points to compare growth trajectories")

    # ── Save ──
    out = {}
    for uid in sorted_uids:
        d = merged[uid]
        label = d["name"] + f"_aut{d['aut']}"
        out[label] = {
            "name": d["name"],
            "nedges": d["nedges"],
            "aut": d["aut"],
            "coeff": {str(k): v for k, v in d["coeff"].items()},
        }
    with open("cross_class_data.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRaw data saved to cross_class_data.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deep', action='store_true',
                        help='Push K_5 to grade 8 (slower)')
    args = parser.parse_args()
    run_experiment(deep=args.deep)


if __name__ == '__main__':
    main()