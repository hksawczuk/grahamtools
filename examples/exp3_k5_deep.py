"""
Experiment 3: K_5 deep iteration — hunt for recurrences.

Pushes K_5 classification as deep as possible (grade 8-10) to get
long coefficient sequences for the 4-edge trees. With enough terms
we can test for linear recurrences with grade-dependent coefficients.

The key quantities to look for:
- K_{1,4}: coefficients 24, 168, 1608, 27528, ...  — does f_k satisfy
  a recurrence like f_k = a(k)*f_{k-1} + b(k)*f_{k-2}?
- Caterpillar T5[32111]: coefficients 5, 15, 61, 393, ...
- Ratio K_{1,4}/caterpillar: does it diverge? At what rate?

Usage:
  python3 exp3_k5_deep.py                 # default: grade 8
  python3 exp3_k5_deep.py --max-k 9      # grade 9 (may take ~30min+)

Output: coefficient sequences, ratio analysis, recurrence search.
"""
import argparse
import sys
import os
import math
import time
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from labels_kn import (
    generate_levels_Kn_ids,
    expand_to_simple_base_edges_id,
    canon_key_bruteforce_bitset,
    aut_size_via_color_classes,
    _canon_cache,
)


def tree_name(edges, n):
    if not edges:
        return "K1"
    nedges = len(edges)
    nv = nedges + 1
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    active_degs = sorted([d for d in deg if d > 0], reverse=True)
    max_d = active_degs[0] if active_degs else 0
    if nedges == 1:
        return "K2"
    if max_d <= 2:
        return f"P{nv}"
    if active_degs.count(1) == nedges and max_d == nedges:
        return f"K1,{nedges}"
    ds_str = "".join(str(d) for d in active_degs)
    return f"T{nv}[{ds_str}]"


def run_experiment(max_k):
    n = 5
    print(f"{'='*72}")
    print(f"  Experiment 3: K_{n} deep iteration, grades 1..{max_k}")
    print(f"{'='*72}")

    _canon_cache.clear()

    # Generate incrementally — classify after each grade so we can
    # report progress and bail early if it gets too slow.
    print(f"\nGenerating all grades at once...")
    t0 = time.time()
    V, ep = generate_levels_Kn_ids(n, max_k, prune_cycles=True)
    t_gen = time.time() - t0
    print(f"  Generation: {t_gen:.1f}s")
    for k in range(max_k + 1):
        nv = len(V.get(k, []))
        print(f"  Grade {k}: {nv} vertices")

    all_fibers = {}  # key -> {name, nedges, coeff: {k: val}}

    for k in range(1, max_k + 1):
        Vk = V.get(k, [])
        if not Vk:
            print(f"\n  Grade {k}: no vertices, stopping.")
            break

        t0 = time.time()
        buckets = {}
        for vi, v in enumerate(Vk):
            edges = expand_to_simple_base_edges_id(v, k, ep)
            key = canon_key_bruteforce_bitset(edges, n)
            if key not in buckets:
                buckets[key] = {"edges": edges, "freq": 0}
            buckets[key]["freq"] += 1

            # Progress for large grades
            if (vi + 1) % 100000 == 0:
                elapsed = time.time() - t0
                rate = (vi + 1) / elapsed
                eta = (len(Vk) - vi - 1) / rate
                print(f"    ... {vi+1}/{len(Vk)} ({rate:.0f}/s, ETA {eta:.0f}s)")

        elapsed = time.time() - t0

        for key, bucket in buckets.items():
            edges = bucket["edges"]
            freq = bucket["freq"]
            aut = aut_size_via_color_classes(edges, n)
            orbit_sz = math.factorial(n) // aut
            coeff = freq // orbit_sz

            if key not in all_fibers:
                all_fibers[key] = {
                    "name": tree_name(edges, n),
                    "nedges": len(edges),
                    "coeff": {},
                }
            all_fibers[key]["coeff"][k] = coeff

        print(f"\n  Grade {k}: {len(buckets)} fibers, {elapsed:.1f}s")
        for key in sorted(all_fibers, key=lambda k: -all_fibers[k]["coeff"].get(k, 0)):
            d = all_fibers[key]
            c = d["coeff"].get(k)
            if c is not None:
                print(f"    {d['name']:>16s}: coeff = {c}")

    sorted_keys = sorted(all_fibers.keys(),
                         key=lambda k: (all_fibers[k]["nedges"],
                                        -max(all_fibers[k]["coeff"].values())))

    # ── Full coefficient sequences ──
    print(f"\n{'='*72}")
    print(f"  Full coefficient sequences")
    print(f"{'='*72}")

    for key in sorted_keys:
        d = all_fibers[key]
        ks = sorted(d["coeff"].keys())
        vals = [d["coeff"][k] for k in ks]
        print(f"\n  {d['name']} ({d['nedges']} edges):")
        print(f"    coeffs = {vals}")
        print(f"    grades = {ks}")

        if len(vals) >= 2:
            ratios = [vals[i]/vals[i-1] for i in range(1, len(vals)) if vals[i-1] > 0]
            print(f"    ratios = [{', '.join(f'{r:.4f}' for r in ratios)}]")

        if len(vals) >= 3:
            logs = [math.log2(v) for v in vals if v > 0]
            deltas = [logs[i]-logs[i-1] for i in range(1, len(logs))]
            ddeltas = [deltas[i]-deltas[i-1] for i in range(1, len(deltas))]
            print(f"    Δ log₂ = [{', '.join(f'{d:.4f}' for d in deltas)}]")
            print(f"    ΔΔ log₂ = [{', '.join(f'{d:.4f}' for d in ddeltas)}]")

    # ── Recurrence search for K_{1,4} ──
    print(f"\n{'='*72}")
    print(f"  Recurrence search")
    print(f"{'='*72}")

    for key in sorted_keys:
        d = all_fibers[key]
        ks = sorted(d["coeff"].keys())
        vals = [d["coeff"][k] for k in ks]
        if len(vals) < 4 or all(v == vals[0] for v in vals):
            continue

        print(f"\n  {d['name']}:")
        print(f"    Testing f_k = a(k)*f_{{k-1}} + b(k)*f_{{k-2}}...")

        # For each triple (k-2, k-1, k), compute what a(k), b(k) would be
        # if f_k = a(k)*f_{k-1} + b(k)*f_{k-2}
        # This is underdetermined with 2 unknowns per equation.
        # Instead, test simpler forms:
        # (1) f_k = c*f_{k-1}  (geometric)
        # (2) f_k = (2^k + c)*f_{k-1}  (grade-dependent multiplier)
        # (3) f_k = a*f_{k-1} + b*f_{k-2}  (constant-coefficient order 2)

        # Test (1): constant ratio
        ratios = [vals[i]/vals[i-1] for i in range(1, len(vals)) if vals[i-1] > 0]
        if len(set(round(r, 4) for r in ratios)) == 1:
            print(f"    → Geometric with ratio {ratios[0]:.4f}")
            continue

        # Test (3): f_k = a*f_{k-1} + b*f_{k-2}
        if len(vals) >= 4:
            # Use first 3 non-trivial values to solve for a, b
            # f[2] = a*f[1] + b*f[0]
            # f[3] = a*f[2] + b*f[1]
            # Solve: [f[1] f[0]] [a]   [f[2]]
            #        [f[2] f[1]] [b] = [f[3]]
            f = vals
            for start in range(len(f) - 3):
                f0, f1, f2, f3 = f[start], f[start+1], f[start+2], f[start+3]
                det = f1*f1 - f2*f0
                if det != 0:
                    a = (f2*f1 - f3*f0) / det
                    b = (f3*f1 - f2*f2) / det
                    # Verify on remaining terms
                    ok = True
                    predicted = list(f[:start+3])
                    for i in range(start+3, len(f)):
                        pred = a * f[i-1] + b * f[i-2]
                        predicted.append(pred)
                        if abs(pred - f[i]) > 0.5:
                            ok = False
                    if ok:
                        print(f"    → Constant-coeff order-2 recurrence: "
                              f"f_k = {a:.4f}*f_{{k-1}} + {b:.4f}*f_{{k-2}} "
                              f"(starting from index {start})")
                        break
            else:
                # Test grade-dependent: f_k = (alpha*2^k + beta)*f_{k-1}
                # ratio[i] = alpha*2^{k_i} + beta
                # Two unknowns, use first two ratios to solve
                if len(ratios) >= 3:
                    k0, k1 = ks[1], ks[2]
                    r0, r1 = ratios[0], ratios[1]
                    # r0 = alpha*2^k0 + beta
                    # r1 = alpha*2^k1 + beta
                    denom = 2**k0 - 2**k1
                    if denom != 0:
                        alpha = (r0 - r1) / denom
                        beta = r0 - alpha * 2**k0
                        # Verify
                        ok = True
                        for i, r in enumerate(ratios):
                            pred = alpha * 2**ks[i+1] + beta
                            if abs(pred - r) / max(r, 1) > 0.01:
                                ok = False
                        if ok:
                            print(f"    → Grade-dependent: ratio_k = "
                                  f"{alpha:.6f}*2^k + {beta:.4f}")
                        else:
                            print(f"    → No simple recurrence found")
                            print(f"      ratios: {[f'{r:.4f}' for r in ratios]}")
                    else:
                        print(f"    → No simple recurrence found")
                else:
                    print(f"    → Not enough data for recurrence search")

    # ── Pairwise coefficient ratios ──
    print(f"\n{'='*72}")
    print(f"  Pairwise coefficient ratios (divergence check)")
    print(f"{'='*72}")

    growing = [(key, all_fibers[key]) for key in sorted_keys
               if len(all_fibers[key]["coeff"]) >= 3
               and not all(v == list(all_fibers[key]["coeff"].values())[0]
                          for v in all_fibers[key]["coeff"].values())]

    for i in range(len(growing)):
        ki, di = growing[i]
        for j in range(i+1, len(growing)):
            kj, dj = growing[j]
            common = sorted(set(di["coeff"]) & set(dj["coeff"]))
            if len(common) < 2:
                continue
            ratios = [di["coeff"][k] / dj["coeff"][k]
                      for k in common if dj["coeff"][k] > 0]
            if len(ratios) >= 2:
                trend = "diverging ↑" if ratios[-1] > ratios[0] * 1.1 else \
                        "diverging ↓" if ratios[-1] < ratios[0] * 0.9 else \
                        "STABLE ⚠"
                print(f"  {di['name']:>16s} / {dj['name']:<16s}: "
                      f"{[f'{r:.3f}' for r in ratios]}  {trend}")

    # ── Save ──
    out = {}
    for key in sorted_keys:
        d = all_fibers[key]
        out[d["name"]] = {
            "nedges": d["nedges"],
            "coeff": {str(k): v for k, v in d["coeff"].items()},
        }
    with open(f"k5_deep_grade{max_k}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRaw data saved to k5_deep_grade{max_k}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-k', type=int, default=8,
                        help='Maximum grade (default: 8)')
    args = parser.parse_args()
    run_experiment(args.max_k)


if __name__ == '__main__':
    main()