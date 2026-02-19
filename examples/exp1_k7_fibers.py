"""
Experiment 1: K_7 fiber coefficients through grade 5-6.

Computes all fiber coefficients for 6-edge trees (and smaller) in K_7.
This gives the first growth-rate data for 6-edge tree types and allows
cross-class comparison of growth rates (4-edge vs 5-edge vs 6-edge fibers
all in the same ambient graph).

Usage:
  python3 exp1_k7_fibers.py              # grade 5 (fast, ~30s)
  python3 exp1_k7_fibers.py --max-k 6    # grade 6 (slow, ~5-10min)

Output: coefficient table, growth rates, cross-class comparison.
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


def classify_grade(Vk, n, k, ep):
    """Classify vertices at grade k into fiber types. Returns dict of fibers."""
    buckets = {}
    t0 = time.time()
    for v in Vk:
        edges = expand_to_simple_base_edges_id(v, k, ep)
        key = canon_key(edges, n)
        if key not in buckets:
            buckets[key] = {"edges": edges, "freq": 0}
        buckets[key]["freq"] += 1
    t1 = time.time()

    fibers = {}
    for key, bucket in buckets.items():
        edges = bucket["edges"]
        freq = bucket["freq"]
        aut = aut_size_edges(edges, n)
        orbit_sz = math.factorial(n) // aut
        coeff = freq // orbit_sz
        fibers[key] = {
            "name": tree_name(edges, n),
            "edges": edges,
            "nedges": len(edges),
            "aut": aut,
            "orbit_sz": orbit_sz,
            "freq": freq,
            "coeff": coeff,
        }

    elapsed = t1 - t0
    return fibers, elapsed


def run_experiment(n, max_k):
    print(f"{'='*72}")
    print(f"  Experiment 1: K_{n} fiber coefficients, grades 1..{max_k}")
    print(f"{'='*72}")

    # Generate
    print(f"\nGenerating L^k(K_{n}) for k=0..{max_k} (trees only)...")
    t0 = time.time()
    V, ep = generate_levels_Kn_ids(n, max_k, prune_cycles=True)
    t_gen = time.time() - t0
    print(f"  Done in {t_gen:.1f}s")
    for k in range(max_k + 1):
        print(f"  Grade {k}: {len(V.get(k, []))} tree vertices")

    # Classify each grade
    all_data = {}  # key -> {name, nedges, coeff: {k: val}, ...}
    for k in range(1, max_k + 1):
        Vk = V.get(k, [])
        if not Vk:
            continue
        fibers, elapsed = classify_grade(Vk, n, k, ep)
        print(f"  Grade {k}: {len(fibers)} fibers, classified in {elapsed:.1f}s "
              f"(cache size: {len(_canon_cache)})")

        for key, fdata in fibers.items():
            if key not in all_data:
                all_data[key] = {
                    "name": fdata["name"],
                    "nedges": fdata["nedges"],
                    "aut": fdata["aut"],
                    "orbit_sz": fdata["orbit_sz"],
                    "coeff": {},
                    "fiber_sz": {},
                }
            all_data[key]["coeff"][k] = fdata["coeff"]
            all_data[key]["fiber_sz"][k] = fdata["freq"]

    # Sort by (nedges, -max_coeff)
    sorted_keys = sorted(all_data.keys(),
                         key=lambda k: (all_data[k]["nedges"],
                                        -max(all_data[k]["coeff"].values())))

    # ── Coefficient table ──
    all_grades = sorted(set(g for d in all_data.values() for g in d["coeff"]))
    print(f"\n{'='*72}")
    print(f"  Coefficients: coeff_k(τ)")
    print(f"{'='*72}")
    hdr = f"  {'Tree':>16s} {'|e|':>4s} {'|Aut|':>5s}"
    for k in all_grades:
        hdr += f" {'k='+str(k):>12s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for key in sorted_keys:
        d = all_data[key]
        row = f"  {d['name']:>16s} {d['nedges']:>4d} {d['aut']:>5d}"
        for k in all_grades:
            c = d["coeff"].get(k)
            row += f" {c:>12d}" if c is not None else f" {'':>12s}"
        print(row)

    # ── Growth rates ──
    print(f"\n{'='*72}")
    print(f"  Growth rates: coeff_k / coeff_{{k-1}}")
    print(f"{'='*72}")
    hdr = f"  {'Tree':>16s} {'|e|':>4s}"
    for k in all_grades[1:]:
        hdr += f" {'k='+str(k):>12s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for key in sorted_keys:
        d = all_data[key]
        row = f"  {d['name']:>16s} {d['nedges']:>4d}"
        for k in all_grades[1:]:
            c = d["coeff"].get(k)
            cp = d["coeff"].get(k - 1)
            if c is not None and cp is not None and cp > 0:
                row += f" {c/cp:>12.2f}"
            else:
                row += f" {'':>12s}"
        print(row)

    # ── Cross-class comparison ──
    print(f"\n{'='*72}")
    print(f"  Cross-class growth comparison (latest available ratio)")
    print(f"{'='*72}")
    print(f"  {'Tree':>16s} {'|e|':>4s} {'last_ratio':>12s} {'at grade':>10s}")
    print("  " + "-" * 50)

    for key in sorted_keys:
        d = all_data[key]
        ks = sorted(d["coeff"].keys())
        last_ratio = None
        last_k = None
        for i in range(1, len(ks)):
            cp = d["coeff"][ks[i-1]]
            c = d["coeff"][ks[i]]
            if cp > 0:
                last_ratio = c / cp
                last_k = ks[i]
        if last_ratio is not None:
            print(f"  {d['name']:>16s} {d['nedges']:>4d} {last_ratio:>12.2f} {last_k:>10d}")
        else:
            print(f"  {d['name']:>16s} {d['nedges']:>4d} {'—':>12s}")

    # ── Log analysis ──
    print(f"\n{'='*72}")
    print(f"  Log₂ analysis (first and second differences)")
    print(f"{'='*72}")

    for key in sorted_keys:
        d = all_data[key]
        ks = sorted(d["coeff"].keys())
        if len(ks) < 2:
            continue
        vals = [d["coeff"][k] for k in ks]
        if all(v == vals[0] for v in vals):
            continue  # skip constant

        logs = [math.log2(v) for v in vals]
        deltas = [logs[i] - logs[i-1] for i in range(1, len(logs))]
        ddeltas = [deltas[i] - deltas[i-1] for i in range(1, len(deltas))]

        print(f"\n  {d['name']} ({d['nedges']} edges)")
        print(f"    {'k':>4s} {'coeff':>12s} {'log2':>8s} {'Δ':>8s} {'ΔΔ':>8s}")
        for i, k in enumerate(ks):
            dl = f"{deltas[i-1]:.3f}" if i > 0 else "—"
            ddl = f"{ddeltas[i-2]:.3f}" if i >= 2 else "—"
            print(f"    {k:>4d} {vals[i]:>12d} {logs[i]:>8.3f} {dl:>8s} {ddl:>8s}")

    # ── Save raw data as JSON ──
    out = {}
    for key in sorted_keys:
        d = all_data[key]
        out[d["name"] + f"_key{key}"] = {
            "name": d["name"],
            "nedges": d["nedges"],
            "aut": d["aut"],
            "orbit_sz": d["orbit_sz"],
            "coeff": {str(k): v for k, v in d["coeff"].items()},
            "fiber_sz": {str(k): v for k, v in d["fiber_sz"].items()},
        }

    json_path = f"k{n}_fibers_grade{max_k}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRaw data saved to {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=7)
    parser.add_argument('--max-k', type=int, default=5,
                        help='Maximum grade (default: 5; try 6 if you have time)')
    args = parser.parse_args()
    run_experiment(args.n, args.max_k)


if __name__ == '__main__':
    main()