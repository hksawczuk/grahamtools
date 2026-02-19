"""
Experiments for iterated line graph coefficient analysis.

Usage:
    python experiments.py [--max-k MAX_K] [--max-n MAX_N] [--exp {1,2,3,4,all}]

Experiments:
  1. Coefficient table: coeff_k(G) for all connected G appearing in L^k(K_n)
  2. K_{1,4} recurrence verification
  3. Fiber closure boundary detection
  4. Fiber graph structure (SS/SC/CC breakdown) for non-closed fibers

Requires the grahamtools package.
"""

from __future__ import annotations
import argparse
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from functools import lru_cache
from typing import Dict, List, Tuple, Set

from grahamtools.kn.levels import generate_levels_Kn_ids
from grahamtools.kn.expand import expand_to_simple_base_edges_id
from grahamtools.kn.classify import canon_key
from grahamtools.utils.automorphisms import aut_size_edges
from grahamtools.kn.labels import format_label
import math


def _normalize_target_1based(target_edges_1based):
    """Normalize 1-based edge list to 0-based sorted unique, i<j."""
    target01 = []
    for u, v in target_edges_1based:
        u -= 1; v -= 1
        if u > v: u, v = v, u
        target01.append((u, v))
    return sorted(set(target01))


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def edges_to_name(edges_0based: List[Tuple[int, int]], n: int) -> str:
    """Heuristic naming for small common graphs.
    Priority: path > cycle > K_n > K_{1,n} > named > generic."""
    m = len(edges_0based)
    if m == 0:
        return f"empty({n})" if n > 1 else "K1"

    deg = [0] * n
    for u, v in edges_0based:
        deg[u] += 1
        deg[v] += 1
    active = sorted(set(u for e in edges_0based for u in e))
    n_a = len(active)
    deg_seq = tuple(sorted([deg[i] for i in active], reverse=True))

    # Paths (including K2 = P2)
    if m == n_a - 1:
        leaves = sum(1 for i in active if deg[i] == 1)
        if leaves == 2 and max(deg[i] for i in active) <= 2:
            return f"P{n_a}"

    # Cycles (including C3 = K3)
    if m == n_a and all(deg[i] == 2 for i in active):
        return f"C{n_a}"

    # Complete graphs (only those not already caught as cycles)
    if m == n_a * (n_a - 1) // 2 and n_a >= 2:
        return f"K{n_a}"

    # Stars
    if n_a >= 3 and m == n_a - 1 and deg_seq[0] == m:
        return f"K1,{m}"

    # Named 4-vertex graphs
    if n_a == 4 and m == 4 and deg_seq == (3, 2, 2, 1):
        return "paw"
    if n_a == 4 and m == 5:
        return "diamond"

    # Named 5-vertex graphs
    if n_a == 5 and m == 4 and deg_seq == (3, 2, 1, 1, 1):
        return "fork"
    if n_a == 5 and m == 5 and deg_seq == (3, 2, 2, 2, 1):
        return "bull"
    if n_a == 5 and m == 5 and deg_seq == (3, 3, 2, 1, 1):
        return "kite"
    if n_a == 5 and m == 5 and deg_seq == (4, 2, 2, 1, 1):
        return "K14+e"

    return f"({n_a}v{m}e{deg_seq})"


def get_all_coefficients(n: int, k: int, endpoints_by_level) -> Dict[int, Dict]:
    """
    Compute coeff_k(G) for every isomorphism class G appearing at grade k.
    """
    Vk_size = len(endpoints_by_level.get(k, []))
    if Vk_size == 0:
        return {}

    buckets = {}
    for v in range(Vk_size):
        edges = expand_to_simple_base_edges_id(v, k, endpoints_by_level)
        key = canon_key(edges, n)
        if key not in buckets:
            buckets[key] = {"edges": edges, "freq": 0}
        buckets[key]["freq"] += 1

    result = {}
    for key, b in buckets.items():
        edges = b["edges"]
        aut = aut_size_edges(edges, n)
        orbit = math.factorial(n) // aut
        freq = b["freq"]
        coeff = freq // orbit if orbit else 0
        name = edges_to_name(edges, n)

        result[key] = {
            "edges": edges,
            "name": name,
            "freq": freq,
            "orbit": orbit,
            "coeff": coeff,
            "n_edges": len(edges),
        }

    return result


# ─────────────────────────────────────────────
# Experiment 1: Full coefficient table
# ─────────────────────────────────────────────

def experiment_coefficient_table(max_n: int = 6, max_k: int = 7):
    """
    For each n, compute coeff_k(G) for every graph G that appears at each grade.
    """
    print("=" * 80)
    print("EXPERIMENT 1: Coefficient Table")
    print("=" * 80)

    all_series: Dict[str, Dict[int, int]] = defaultdict(dict)
    edges_for_key: Dict[str, str] = {}

    for n in range(3, max_n + 1):
        print(f"\n--- n = {n} ---")
        t0 = time.time()

        V_by_level, endpoints_by_level = generate_levels_Kn_ids(n, max_k)

        for k in range(1, max_k + 1):
            if k not in endpoints_by_level:
                break
            vk_size = len(endpoints_by_level[k])
            if vk_size > 500_000:
                print(f"  k={k}: |V|={vk_size} -- too large, stopping")
                break

            t1 = time.time()
            coeffs = get_all_coefficients(n, k, endpoints_by_level)
            dt = time.time() - t1

            print(f"  k={k}: |V|={vk_size}, {len(coeffs)} iso classes ({dt:.1f}s)")

            for key, info in coeffs.items():
                label = f"n={n}:{info['name']}"
                all_series[label][k] = info["coeff"]
                edges_for_key[label] = str([(u+1, v+1) for u, v in info["edges"]])

        print(f"  total: {time.time()-t0:.1f}s")

    # Summary
    print("\n" + "=" * 80)
    print("COEFFICIENT SERIES SUMMARY")
    print("=" * 80)

    for label in sorted(all_series.keys()):
        series = all_series[label]
        ks = sorted(series.keys())
        vals = [f"k={k}:{series[k]}" for k in ks]
        print(f"\n  {label:<35} edges={edges_for_key[label]}")
        print(f"    {', '.join(vals)}")

        if len(ks) >= 3:
            ratios = []
            for i in range(1, len(ks)):
                prev = series[ks[i-1]]
                ratios.append(series[ks[i]] / prev if prev > 0 else float('inf'))
            print(f"    ratios: {', '.join(f'{r:.2f}' for r in ratios)}")

            if all(1 < r < float('inf') for r in ratios) and len(ratios) >= 2:
                log_ratios = [math.log2(r) for r in ratios]
                print(f"    log2(ratios): {', '.join(f'{lr:.2f}' for lr in log_ratios)}")


# ─────────────────────────────────────────────
# Experiment 2: K_{1,4} recurrence verification
# ─────────────────────────────────────────────

def experiment_k14_recurrence(max_n: int = 7, max_k: int = 8):
    """
    Extract coeff_k(K_{1,4}) for a specific labeled copy and compare with:
        f_k = (2^{k-1} + 1) * f_{k-1} + 12 * 2^{k-1}
    """
    print("=" * 80)
    print("EXPERIMENT 2: K_{1,4} Recurrence Verification")
    print("=" * 80)

    n = max_n

    print(f"Computing L^k(K_{n}) for k up to {max_k}...")
    t0 = time.time()

    V_by_level, endpoints_by_level = generate_levels_Kn_ids(n, max_k)

    # Target: K_{1,4} with center 0, leaves 1,2,3,4 (0-based)
    target_key = tuple([(0, 1), (0, 2), (0, 3), (0, 4)])

    @lru_cache(maxsize=None)
    def support_key(v: int, t: int):
        return tuple(expand_to_simple_base_edges_id(v, t, endpoints_by_level))

    observed = {}
    for k in range(1, max_k + 1):
        if k not in endpoints_by_level:
            break
        vk_size = len(endpoints_by_level[k])
        if vk_size > 5_000_000:
            print(f"  k={k}: |V|={vk_size} -- too large, stopping")
            break

        t1 = time.time()
        count = sum(1 for v in range(vk_size) if support_key(v, k) == target_key)
        dt = time.time() - t1

        observed[k] = count
        print(f"  k={k}: labeled_fiber(K_{{1,4}}) = {count}  ({dt:.1f}s, |V|={vk_size})")

    print(f"\nTotal: {time.time() - t0:.1f}s")

    # Recurrence check
    print("\n--- Recurrence: f_k = (2^{k-3} + 1) * f_{k-1} + 12 * 2^{k-3} ---")
    ks = sorted(observed.keys())
    if len(ks) >= 2:
        print(f"\n{'k':>4} {'observed':>14} {'predicted':>14} {'match':>7}")
        print("-" * 43)
        for i, k in enumerate(ks):
            if i == 0:
                print(f"{k:4d} {observed[k]:14d} {'(base)':>14}")
            else:
                if k >= 4:
                    pred = (2**(k-3) + 1) * observed[ks[i-1]] + 12 * 2**(k-3)
                    match = "YES" if pred == observed[k] else "NO"
                    print(f"{k:4d} {observed[k]:14d} {pred:14d} {match:>7}")
                else:
                    print(f"{k:4d} {observed[k]:14d} {'(pre-base)':>14}")

    # Ratio analysis
    print("\n--- Growth rate analysis ---")
    for i in range(1, len(ks)):
        k = ks[i]
        if observed[ks[i-1]] > 0:
            ratio = observed[k] / observed[ks[i-1]]
            print(f"  f_{k}/f_{ks[i-1]} = {ratio:.4f}   (2^(k-1)+1 = {2**(k-1)+1})")

    return observed


# ─────────────────────────────────────────────
# Experiment 3: Fiber closure boundary
# ─────────────────────────────────────────────

def experiment_fiber_closure(max_n: int = 6, max_k: int = 7):
    """
    For each graph G appearing in L^k(K_n), determine at which grades
    the fiber is closed vs. open.
    """
    print("=" * 80)
    print("EXPERIMENT 3: Fiber Closure Boundary")
    print("=" * 80)

    for n in range(4, max_n + 1):
        print(f"\n--- n = {n} ---")
        V_by_level, endpoints_by_level = generate_levels_Kn_ids(n, max_k)

        @lru_cache(maxsize=None)
        def support_key(v: int, t: int):
            return tuple(expand_to_simple_base_edges_id(v, t, endpoints_by_level))

        # Collect supports at each grade
        supports_at_grade: Dict[int, Set] = {}
        for k in range(1, max_k + 1):
            if k not in endpoints_by_level:
                break
            vk_size = len(endpoints_by_level[k])
            if vk_size > 300_000:
                print(f"  k={k}: too large ({vk_size}), stopping")
                break
            supports_at_grade[k] = set(support_key(v, k) for v in range(vk_size))

        all_supports = set()
        for s in supports_at_grade.values():
            all_supports.update(s)

        print(f"\n  {'Graph':<40} {'Realized':<15} {'Closed':<15} {'Open':<15}")
        print("  " + "-" * 85)

        for supp in sorted(all_supports, key=lambda s: (len(s), s)):
            if len(supp) > 6:
                continue
            name = edges_to_name(list(supp), n)
            edges_1 = [(u+1, v+1) for u, v in supp]

            realized, closed, opened = [], [], []

            for k in sorted(supports_at_grade.keys()):
                if supp not in supports_at_grade[k]:
                    continue
                realized.append(k)
                if k < 2:
                    continue

                # Build F_{k-1}(G)
                Fkm1 = set()
                if k-1 in supports_at_grade and supp in supports_at_grade.get(k-1, set()):
                    for v in range(len(endpoints_by_level[k-1])):
                        if support_key(v, k-1) == supp:
                            Fkm1.add(v)

                is_closed = True
                for v in range(len(endpoints_by_level[k])):
                    if support_key(v, k) != supp:
                        continue
                    a, b = endpoints_by_level[k][v]
                    if a not in Fkm1 or b not in Fkm1:
                        is_closed = False
                        break

                (closed if is_closed else opened).append(k)

            print(f"  {name:<40} {str(realized):<15} {str(closed):<15} {str(opened):<15}")

        support_key.cache_clear()


# ─────────────────────────────────────────────
# Experiment 4: Fiber structure over k
# ─────────────────────────────────────────────

def experiment_fiber_structure(targets, max_n: int = 7, max_k: int = 8):
    """
    For target graphs, track: fiber size, SS/SC/CC counts, degree distribution.
    """
    print("=" * 80)
    print("EXPERIMENT 4: Fiber Graph Structure Over k")
    print("=" * 80)

    for target_name, target_edges_1based in targets:
        min_n = max(v for e in target_edges_1based for v in e)
        n = max(min_n + 1, max_n)

        print(f"\n--- {target_name} (n={n}) ---")

        target_key = tuple(_normalize_target_1based(target_edges_1based))

        V_by_level, endpoints_by_level = generate_levels_Kn_ids(n, max_k)

        @lru_cache(maxsize=None)
        def support_key(v: int, t: int):
            return tuple(expand_to_simple_base_edges_id(v, t, endpoints_by_level))

        print(f"  {'k':>3} {'|F_k|':>10} {'SS':>8} {'SC':>8} {'CC':>8} {'|E(Γ)|':>10} {'E/F':>8} {'avg_deg':>8} {'f_k/f_{k-1}':>12} {'deg_dist'}")
        print("  " + "-" * 110)

        prev_fk = None
        for k in range(1, max_k + 1):
            if k not in endpoints_by_level:
                break
            vk_size = len(endpoints_by_level[k])
            if vk_size > 2_000_000:
                print(f"  {k:3d}  -- too large (|V|={vk_size})")
                break

            hits = [v for v in range(vk_size) if support_key(v, k) == target_key]
            m = len(hits)
            if m == 0:
                print(f"  {k:3d} {'0':>10}")
                prev_fk = 0
                continue

            # F_{k-1}
            Fkm1 = set()
            if k >= 1 and (k-1) in endpoints_by_level:
                for v in range(len(endpoints_by_level[k-1])):
                    if support_key(v, k-1) == target_key:
                        Fkm1.add(v)

            # Build fiber graph adjacency
            idx_of = {v: i for i, v in enumerate(hits)}
            inc = defaultdict(list)
            for v in hits:
                a, b = endpoints_by_level[k][v]
                inc[a].append(idx_of[v])
                inc[b].append(idx_of[v])

            nbrs = [set() for _ in range(m)]
            for lst in inc.values():
                for i, j in combinations(lst, 2):
                    nbrs[i].add(j)
                    nbrs[j].add(i)

            n_ss = n_sc = n_cc = 0
            degs = []
            for v in hits:
                a, b = endpoints_by_level[k][v]
                s = int(a in Fkm1) + int(b in Fkm1)
                if s == 2: n_ss += 1
                elif s == 1: n_sc += 1
                else: n_cc += 1
                degs.append(len(nbrs[idx_of[v]]))

            E = sum(degs) // 2
            e_over_f = f"{E/m:.2f}" if m > 0 else "—"
            avg_deg = f"{2*E/m:.2f}" if m > 0 else "—"
            ratio = f"{m/prev_fk:.4f}" if prev_fk and prev_fk > 0 else "—"
            deg_counter = dict(sorted(Counter(degs).items()))

            print(f"  {k:3d} {m:10d} {n_ss:8d} {n_sc:8d} {n_cc:8d} {E:10d} {e_over_f:>8} {avg_deg:>8} {ratio:>12} {deg_counter}")
            prev_fk = m

        # ── Recurrence detection ──
        # Collect the fiber sizes we computed
        fk_data = {}
        for k in range(1, max_k + 1):
            if k not in endpoints_by_level:
                break
            vk_size = len(endpoints_by_level[k])
            if vk_size > 2_000_000:
                break
            count = sum(1 for v in range(vk_size) if support_key(v, k) == target_key)
            if count > 0:
                fk_data[k] = count

        ks_with_data = sorted(fk_data.keys())
        if len(ks_with_data) >= 3:
            print(f"\n  --- Recurrence analysis for {target_name} ---")
            print(f"  Data: {', '.join(f'f_{k}={fk_data[k]}' for k in ks_with_data)}")

            # For consecutive triples, solve f_k = a*f_{k-1} + b for (a, b)
            # Given f_k = a*f_{k-1}+b and f_{k-1} = a*f_{k-2}+b:
            #   f_k - f_{k-1} = a*(f_{k-1} - f_{k-2})
            for i in range(2, len(ks_with_data)):
                k2, k1, k0 = ks_with_data[i], ks_with_data[i-1], ks_with_data[i-2]
                if k2 != k1+1 or k1 != k0+1:
                    continue  # need consecutive grades
                f2, f1, f0 = fk_data[k2], fk_data[k1], fk_data[k0]
                if f1 == f0:
                    print(f"  k={k2}: f constant at {f1} (closed fiber)")
                    continue
                a = (f2 - f1) / (f1 - f0)
                b = f1 - a * f0
                print(f"  k={k0},{k1},{k2}: f={f0},{f1},{f2}  =>  a={a:.4f}  b={b:.4f}  (2^(k-?)={2**(k1-1)},{2**k1})")
        print()

        support_key.cache_clear()


# ─────────────────────────────────────────────
# Experiment 5: Deep single-target analysis
# (paw at n=4 with high k)
# ─────────────────────────────────────────────

def experiment_deep_paw(max_k: int = 9):
    """
    Run the paw at n=4 to high k. At n=4, k=6 has only 27540 vertices,
    so k=7 (~470k) and maybe k=8 (~8M) should be feasible.
    Reports fiber size, SS/SC/CC, |E(Γ)|, and tests SC doubling.
    """
    print("=" * 80)
    print("EXPERIMENT 5: Deep Paw Analysis (n=4)")
    print("=" * 80)

    n = 4
    target_edges = [(1, 2), (1, 3), (1, 4), (2, 3)]
    target_key = tuple(_normalize_target_1based(target_edges))

    print(f"Computing L^k(K_{n}) for k up to {max_k}...")
    t0 = time.time()
    V_by_level, endpoints_by_level = generate_levels_Kn_ids(n, max_k)
    print(f"Generation done in {time.time()-t0:.1f}s")

    for k in sorted(V_by_level.keys()):
        print(f"  |V(L^{k}(K_{n}))| = {len(V_by_level[k])}")

    @lru_cache(maxsize=None)
    def support_key(v: int, t: int):
        return tuple(expand_to_simple_base_edges_id(v, t, endpoints_by_level))

    print(f"\n  {'k':>3} {'|F_k|':>10} {'SS':>8} {'SC':>8} {'CC':>8} {'|E(Γ)|':>10} {'E/F':>8} {'SC/prev_SC':>11} {'f_k/f_{k-1}':>12}")
    print("  " + "-" * 100)

    prev_fk = None
    prev_sc = None
    fiber_sizes = {}

    for k in range(1, max_k + 1):
        if k not in endpoints_by_level:
            break
        vk_size = len(endpoints_by_level[k])
        if vk_size > 10_000_000:
            print(f"  {k:3d}  -- too large (|V|={vk_size})")
            break

        t1 = time.time()

        # Fiber at grade k
        hits = [v for v in range(vk_size) if support_key(v, k) == target_key]
        m = len(hits)
        fiber_sizes[k] = m

        if m == 0:
            print(f"  {k:3d} {'0':>10}")
            prev_fk = 0
            continue

        # F_{k-1}
        Fkm1 = set()
        if k >= 1 and (k-1) in endpoints_by_level:
            for v in range(len(endpoints_by_level[k-1])):
                if support_key(v, k-1) == target_key:
                    Fkm1.add(v)

        # Build fiber graph adjacency
        idx_of = {v: i for i, v in enumerate(hits)}
        inc = defaultdict(list)
        for v in hits:
            a, b = endpoints_by_level[k][v]
            inc[a].append(idx_of[v])
            inc[b].append(idx_of[v])

        nbrs = [set() for _ in range(m)]
        for lst in inc.values():
            for i, j in combinations(lst, 2):
                nbrs[i].add(j)
                nbrs[j].add(i)

        n_ss = n_sc = n_cc = 0
        degs = []
        for v in hits:
            a, b = endpoints_by_level[k][v]
            s = int(a in Fkm1) + int(b in Fkm1)
            if s == 2: n_ss += 1
            elif s == 1: n_sc += 1
            else: n_cc += 1
            degs.append(len(nbrs[idx_of[v]]))

        E = sum(degs) // 2
        e_over_f = f"{E/m:.2f}" if m > 0 else "—"
        ratio = f"{m/prev_fk:.4f}" if prev_fk and prev_fk > 0 else "—"
        sc_ratio = f"{n_sc/prev_sc:.4f}" if prev_sc and prev_sc > 0 else "—"

        dt = time.time() - t1
        print(f"  {k:3d} {m:10d} {n_ss:8d} {n_sc:8d} {n_cc:8d} {E:10d} {e_over_f:>8} {sc_ratio:>11} {ratio:>12}  ({dt:.1f}s)")

        prev_fk = m
        prev_sc = n_sc if n_sc > 0 else prev_sc

    # Summary analysis
    ks = sorted(k for k, v in fiber_sizes.items() if v > 0)
    if len(ks) >= 3:
        print(f"\n  --- Summary ---")
        print(f"  Fiber sizes: {', '.join(f'f_{k}={fiber_sizes[k]}' for k in ks)}")
        print(f"  Ratios f_k/f_{{k-1}}: ", end="")
        for i in range(1, len(ks)):
            r = fiber_sizes[ks[i]] / fiber_sizes[ks[i-1]]
            print(f"{r:.4f}  ", end="")
        print()

    print(f"\n  Total time: {time.time()-t0:.1f}s")


# ─────────────────────────────────────────────
# Experiment 6: Lightweight K_{1,4} fiber counts
# (skip fiber graph construction, just count SS/SC/CC)
# ─────────────────────────────────────────────

def experiment_k14_lightweight(max_n: int = 5, max_k: int = 7):
    """
    Count f_k, SS, SC, CC for K_{1,4} without building the fiber graph.
    This avoids the expensive O(f_k^2) adjacency construction.
    Also counts SC by subgraph type (which (|E|-1)-edge subgraph is the other parent).
    """
    print("=" * 80)
    print("EXPERIMENT 6: Lightweight K_{1,4} Fiber Counts")
    print("=" * 80)

    n = max_n
    target_edges = [(1, 2), (1, 3), (1, 4), (1, 5)]
    target_key = tuple(_normalize_target_1based(target_edges))

    # Also define the 3-edge connected subgraphs of K_{1,4}
    # K_{1,3} subgraphs: pick 3 of 4 leaves. Center is always vertex 0.
    # P4 subgraphs: remove the center-leaf edge that's the "middle" — 
    #   actually P4 = path on 4 vertices, which is not a subgraph of K_{1,4}
    #   since K_{1,4} has no path of length 3 (all paths go through center).
    #   The 3-edge connected subgraphs of K_{1,4} are exactly the 4 copies of K_{1,3}.

    print(f"Computing L^k(K_{n}) for k up to {max_k}...")
    t0 = time.time()
    V_by_level, endpoints_by_level = generate_levels_Kn_ids(n, max_k)
    print(f"Generation done in {time.time()-t0:.1f}s")

    for k in sorted(V_by_level.keys()):
        print(f"  |V(L^{k}(K_{n}))| = {len(V_by_level[k])}")

    @lru_cache(maxsize=None)
    def support_key(v: int, t: int):
        return tuple(expand_to_simple_base_edges_id(v, t, endpoints_by_level))

    # Precompute: which support keys are K_{1,3} subgraphs of our target?
    # Our target is edges {(0,1),(0,2),(0,3),(0,4)} (0-based)
    # K_{1,3} subs: remove one leaf edge
    k13_keys = set()
    target_edges_0 = [(0,1),(0,2),(0,3),(0,4)]
    for i in range(4):
        sub = tuple(sorted(target_edges_0[:i] + target_edges_0[i+1:]))
        k13_keys.add(sub)
    print(f"  K_{{1,3}} sub-support keys: {len(k13_keys)}")

    print(f"\n  {'k':>3} {'|F_k|':>10} {'SS':>8} {'SC':>8} {'CC':>8}"
          f" {'SC(K13)':>8} {'SC(other)':>9}"
          f" {'SC/prev':>9} {'f/prev':>9}")
    print("  " + "-" * 90)

    prev_fk = None
    prev_sc = None

    for k in range(1, max_k + 1):
        if k not in endpoints_by_level:
            break
        vk_size = len(endpoints_by_level[k])

        t1 = time.time()

        # Build F_{k-1}(target) — needed for SS/SC classification
        Fkm1 = set()
        if k >= 2 and (k-1) in endpoints_by_level:
            for v in range(len(endpoints_by_level[k-1])):
                if support_key(v, k-1) == target_key:
                    Fkm1.add(v)

        # Scan grade k: only check support, classify parents
        n_ss = n_sc = n_cc = 0
        sc_k13 = 0
        sc_other = 0
        fk = 0

        for v in range(vk_size):
            if support_key(v, k) != target_key:
                continue
            fk += 1
            a, b = endpoints_by_level[k][v]
            in_a = a in Fkm1
            in_b = b in Fkm1
            s = int(in_a) + int(in_b)

            if s == 2:
                n_ss += 1
            elif s == 1:
                n_sc += 1
                # Which parent is outside? Check its support type
                other = b if in_a else a
                other_supp = support_key(other, k - 1)
                if other_supp in k13_keys:
                    sc_k13 += 1
                else:
                    sc_other += 1
            else:
                n_cc += 1

        dt = time.time() - t1

        sc_ratio = f"{n_sc/prev_sc:.4f}" if prev_sc and prev_sc > 0 else "—"
        f_ratio = f"{fk/prev_fk:.4f}" if prev_fk and prev_fk > 0 else "—"

        print(f"  {k:3d} {fk:10d} {n_ss:8d} {n_sc:8d} {n_cc:8d}"
              f" {sc_k13:8d} {sc_other:9d}"
              f" {sc_ratio:>9} {f_ratio:>9}  ({dt:.1f}s)")

        prev_fk = fk if fk > 0 else prev_fk
        prev_sc = n_sc if n_sc > 0 else prev_sc

    # Recurrence check
    print(f"\n  Total time: {time.time()-t0:.1f}s")


# ─────────────────────────────────────────────
# Experiment 7: Quotient algebra L^k(K_n)/S_n
# ─────────────────────────────────────────────

def experiment_quotient(max_n: int = 5, max_k: int = 7, trees_only: bool = False,
                        target: str = None):
    """
    Build L^k(K_n), quotient by S_n at each grade.

    Each S_n-orbit gets a persistent name like paw_0, paw_1, ...
    assigned in order of first appearance.

    If trees_only=True, only orbits whose support is a tree are tracked.
    If target is specified, only orbits whose support is a subgraph of
    the target graph are tracked (subsumes trees_only for tree targets).
    """
    from itertools import permutations

    # ── Known graph names -> 0-indexed edge lists ──
    KNOWN_GRAPHS = {
        "P2":      [(0, 1)],
        "P3":      [(0, 1), (1, 2)],
        "P4":      [(0, 1), (1, 2), (2, 3)],
        "P5":      [(0, 1), (1, 2), (2, 3), (3, 4)],
        "C3":      [(0, 1), (1, 2), (0, 2)],
        "C4":      [(0, 1), (1, 2), (2, 3), (0, 3)],
        "C5":      [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)],
        "K3":      [(0, 1), (1, 2), (0, 2)],
        "K4":      [(i, j) for i in range(4) for j in range(i+1, 4)],
        "K1,3":    [(0, 1), (0, 2), (0, 3)],
        "K1,4":    [(0, 1), (0, 2), (0, 3), (0, 4)],
        "paw":     [(0, 1), (0, 2), (1, 2), (2, 3)],
        "diamond": [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)],
        "fork":    [(0, 1), (0, 2), (0, 3), (3, 4)],
        "bull":    [(0, 1), (1, 2), (2, 3), (1, 4), (2, 4)],
        "kite":    [(0, 1), (1, 2), (2, 3), (1, 3), (3, 4)],
    }

    # ── Parse target graph ──
    target_edges_0 = None  # 0-indexed edges of target
    target_adj = None      # adjacency set for fast lookup
    target_verts = None
    if target is not None:
        tkey = target.replace("_", "").replace("{", "").replace("}", "")
        # Try exact match and common variants
        matched = None
        for k_name, k_edges in KNOWN_GRAPHS.items():
            if tkey.lower() == k_name.lower().replace("_", "").replace(",", ""):
                matched = (k_name, k_edges)
                break
        # Also try with comma preserved
        if matched is None:
            for k_name, k_edges in KNOWN_GRAPHS.items():
                if tkey.lower() == k_name.lower().replace("_", ""):
                    matched = (k_name, k_edges)
                    break
        if matched is None:
            print(f"ERROR: Unknown target graph '{target}'.")
            print(f"  Known: {', '.join(sorted(KNOWN_GRAPHS.keys()))}")
            return
        target_name, target_edges_0 = matched
        target_verts = sorted(set(v for e in target_edges_0 for v in e))
        target_adj = set()
        for u, v in target_edges_0:
            target_adj.add((min(u, v), max(u, v)))
        print(f"  Target: {target_name} ({len(target_verts)}v, {len(target_edges_0)}e)")

    def is_subgraph_of_target(supp):
        """Check if support (frozenset of 1-indexed edge tuples) embeds into target."""
        if target_edges_0 is None:
            return True
        if not supp:
            return True  # empty graph is subgraph of anything
        s_verts = sorted(set(v for e in supp for v in e))
        if len(s_verts) > len(target_verts):
            return False
        if len(supp) > len(target_edges_0):
            return False
        s_edges = [(min(u, v), max(u, v)) for u, v in supp]
        # Try all injective maps from s_verts -> target_verts
        for perm in permutations(target_verts, len(s_verts)):
            mapping = dict(zip(s_verts, perm))
            ok = True
            for u, v in s_edges:
                mu, mv = mapping[u], mapping[v]
                if (min(mu, mv), max(mu, mv)) not in target_adj:
                    ok = False
                    break
            if ok:
                return True
        return False

    # Cache subgraph check per support
    _subgraph_cache = {}
    def passes_filter(supp):
        """Check if support passes all active filters."""
        if supp in _subgraph_cache:
            return _subgraph_cache[supp]
        ok = True
        if trees_only and not is_tree_support(supp):
            ok = False
        if ok and target_edges_0 is not None and not is_subgraph_of_target(supp):
            ok = False
        _subgraph_cache[supp] = ok
        return ok

    def is_tree_support(supp):
        """Check if a frozenset of edges forms a tree."""
        if not supp:
            return True
        verts = set()
        for u, v in supp:
            verts.add(u)
            verts.add(v)
        n_v = len(verts)
        n_e = len(supp)
        if n_e != n_v - 1:
            return False
        adj = defaultdict(set)
        for u, v in supp:
            adj[u].add(v)
            adj[v].add(u)
        start = next(iter(verts))
        visited = {start}
        queue = [start]
        while queue:
            u = queue.pop()
            for w in adj[u]:
                if w not in visited:
                    visited.add(w)
                    queue.append(w)
        return len(visited) == n_v

    has_filter = trees_only or (target is not None)
    mode_parts = []
    if trees_only:
        mode_parts.append("trees only")
    if target is not None:
        mode_parts.append(f"target={target}")
    mode_str = f" ({', '.join(mode_parts)})" if mode_parts else ""
    print("=" * 80)
    print(f"EXPERIMENT 7: Quotient Algebra  L^k(K_n) / S_n{mode_str}")
    print("=" * 80)

    for n in range(3, max_n + 1):
        _subgraph_cache.clear()
        print(f"\n{'='*70}")
        print(f"  n = {n}")
        print(f"{'='*70}")

        t0 = time.time()
        V_by_level, endpoints_by_level = generate_levels_Kn_ids(n, max_k)
        print(f"  Generation: {time.time()-t0:.1f}s")
        for kk in sorted(V_by_level.keys()):
            print(f"    |V(L^{kk})| = {len(V_by_level[kk])}")

        all_perms = list(permutations(range(n)))

        # ── Tree labels and canonicalization ──
        @lru_cache(maxsize=None)
        def tree_label(v, k):
            if k == 0:
                return v
            a, b = endpoints_by_level[k][v]
            la = tree_label(a, k - 1)
            lb = tree_label(b, k - 1)
            return (min(la, lb), max(la, lb))

        def apply_perm(label, perm):
            if isinstance(label, int):
                return perm[label]
            a, b = label
            pa = apply_perm(a, perm)
            pb = apply_perm(b, perm)
            return (min(pa, pb), max(pa, pb))

        @lru_cache(maxsize=None)
        def canonicalize(label):
            best = label
            for p in all_perms:
                c = apply_perm(label, p)
                if c < best:
                    best = c
            return best

        def extract_support(label):
            if isinstance(label, int):
                return frozenset()
            a, b = label
            if isinstance(a, int) and isinstance(b, int):
                return frozenset([(a, b)])
            return extract_support(a) | extract_support(b)

        # ── Persistent orbit registry ──
        # canon_label -> orbit_name
        orbit_name_of = {}
        # support_type -> next index
        support_counter = defaultdict(int)
        # orbit_name -> canon_label
        label_of_orbit = {}
        # orbit_name -> support frozenset
        support_of_orbit = {}
        # orbit_name -> grade of first appearance
        grade_of_orbit = {}

        def get_or_assign_name(canon_label, supp, k):
            if canon_label in orbit_name_of:
                return orbit_name_of[canon_label]
            base = edges_to_name(sorted(supp), n)
            idx = support_counter[base]
            support_counter[base] += 1
            name = f"{base}_{idx}"
            orbit_name_of[canon_label] = name
            label_of_orbit[name] = canon_label
            support_of_orbit[name] = supp
            grade_of_orbit[name] = k
            return name

        # ── Process each grade ──
        # Track which orbit names exist at each grade
        orbits_at_grade = {}  # grade -> {orbit_name: [list of raw vertices]}
        orbit_parent_pair = {}  # persistent across grades

        for k in range(1, max_k + 1):
            if k not in endpoints_by_level:
                break
            vk_size = len(endpoints_by_level[k])
            vk_limit = 100_000_000 if has_filter else 2_000_000
            if vk_size > vk_limit:
                print(f"\n  Grade {k}: |V|={vk_size} -- too large, stopping")
                break

            t1 = time.time()

            # Map vertices to canonical labels, then to orbit names
            vertex_orbit = {}  # raw vertex -> orbit_name
            orbit_vertices = defaultdict(list)  # orbit_name -> [raw vertices]
            skipped = 0

            for v in range(vk_size):
                tl = tree_label(v, k)

                # Fast filter: check support before expensive canonicalization
                if has_filter:
                    supp = extract_support(tl)
                    if not passes_filter(supp):
                        skipped += 1
                        continue

                cl = canonicalize(tl)
                supp = extract_support(cl)
                name = get_or_assign_name(cl, supp, k)
                vertex_orbit[v] = name
                orbit_vertices[name].append(v)

            if has_filter and skipped > 0:
                print(f"    (skipped {skipped} non-target vertices)")

            orbits_at_grade[k] = dict(orbit_vertices)

            # ── Parent pair for each orbit ──
            orbit_parent_pair_k = {}  # this grade's parent pairs
            for oname in orbit_vertices:
                v = orbit_vertices[oname][0]
                a, b = endpoints_by_level[k][v]
                # Parents may have been skipped in trees_only mode
                pa_tl = tree_label(a, k - 1)
                pb_tl = tree_label(b, k - 1)
                pa_cl = canonicalize(pa_tl)
                pb_cl = canonicalize(pb_tl)
                pa_supp = extract_support(pa_cl)
                pb_supp = extract_support(pb_cl)
                pa_name = get_or_assign_name(pa_cl, pa_supp, k - 1)
                pb_name = get_or_assign_name(pb_cl, pb_supp, k - 1)
                orbit_parent_pair_k[oname] = tuple(sorted([pa_name, pb_name]))
            orbit_parent_pair.update(orbit_parent_pair_k)

            # ── Group orbits by support ──
            support_groups = defaultdict(list)
            for oname in orbit_vertices:
                support_groups[support_of_orbit[oname]].append(oname)

            # ── AA/AB/BB classification ──
            # For orbit c with support G:
            #   AA: both parents have support G
            #   AB: exactly one parent has support G
            #   BB: neither parent has support G
            def classify_aabb(oname):
                supp = support_of_orbit[oname]
                pa, pb = orbit_parent_pair[oname]
                a_in = (support_of_orbit.get(pa) == supp)
                b_in = (support_of_orbit.get(pb) == supp)
                if a_in and b_in:
                    return "AA"
                elif a_in or b_in:
                    return "AB"
                else:
                    return "BB"

            # ── Quotient fiber graph: adjacency among orbits with same support ──
            # Two orbits c1, c2 with same support are adjacent if some raw
            # vertex in c1 shares a parent with some raw vertex in c2.
            # Quotient degree = number of adjacent orbits (not raw neighbors).

            # Build: for each raw parent vertex w, which child orbits does it touch?
            parent_to_child_orbits = defaultdict(set)
            for v in range(vk_size):
                if v not in vertex_orbit:
                    continue
                a, b = endpoints_by_level[k][v]
                oname = vertex_orbit[v]
                parent_to_child_orbits[a].add(oname)
                parent_to_child_orbits[b].add(oname)

            # Quotient adjacency per support group (including self-loops)
            quotient_adj = defaultdict(set)  # oname -> set of neighbor onames (same support)
            for supp, onames in support_groups.items():
                fiber_set = set(onames)
                for w, children in parent_to_child_orbits.items():
                    fc = children & fiber_set
                    for c1 in fc:
                        for c2 in fc:
                            quotient_adj[c1].add(c2)

            # ── Quotient multiplicity: for each orbit, count how many times
            # each parent-pair type produces it, per canonical representative ──
            # i.e. for the canonical rep vertex, how many of its raw edges
            # at grade k come from parent pair (α, β)?
            # ── Compute per-orbit neighbor counts in raw fiber graph,
            # then quotient by dividing by orbit_sz ──
            # For canonical rep of orbit c: count raw neighbors by their orbit name

            # Precompute: parent -> list of (child_vertex, child_orbit)
            parent_children = defaultdict(list)
            for v in range(vk_size):
                if v not in vertex_orbit:
                    continue
                a, b = endpoints_by_level[k][v]
                oname = vertex_orbit[v]
                parent_children[a].append((v, oname))
                parent_children[b].append((v, oname))

            orbit_neighbor_counts = {}
            for supp, onames in support_groups.items():
                fiber_set = set(onames)
                for oname in onames:
                    v0 = orbit_vertices[oname][0]  # canonical rep
                    a0, b0 = endpoints_by_level[k][v0]
                    nbr_counts = defaultdict(int)
                    for w in [a0, b0]:
                        for (v2, co) in parent_children[w]:
                            if v2 != v0 and co in fiber_set:
                                nbr_counts[co] += 1
                    orbit_neighbor_counts[oname] = dict(nbr_counts)

            # ── Product table: for each parent pair (α, β) at grade k,
            # which child orbits are produced? ──
            product_table = defaultdict(list)  # (α, β) -> [child_orbit_names]
            for oname in orbit_vertices:
                pp = orbit_parent_pair[oname]
                product_table[pp].append(oname)

            dt = time.time() - t1

            # ════════════════════════════════════════════
            # ── Output ──
            # ════════════════════════════════════════════

            total_orbits = len(orbit_vertices)
            print(f"\n  Grade {k}: {vk_size} vertices -> {total_orbits} orbits  ({dt:.1f}s)")

            # ── Compact summary table ──
            print(f"    {'support':<16s} {'coeff':>6s} {'AA':>4s} {'AB':>4s} {'BB':>4s} {'|E(Γ̃)|':>7s}")
            print(f"    {'-'*45}")
            for supp in sorted(support_groups, key=lambda s: (len(s), sorted(s))):
                onames = sorted(support_groups[supp])
                base_name = edges_to_name(sorted(supp), n)
                n_aa = sum(1 for o in onames if classify_aabb(o) == "AA")
                n_ab = sum(1 for o in onames if classify_aabb(o) == "AB")
                n_bb = sum(1 for o in onames if classify_aabb(o) == "BB")
                n_self = sum(1 for o in onames if o in quotient_adj.get(o, set()))
                deg_sum = sum(len(quotient_adj.get(o, set())) for o in onames)
                # deg_sum = 2*E_inter + E_self, total = E_inter + E_self = (deg_sum + n_self) // 2
                q_edges = (deg_sum + n_self) // 2
                print(f"    {base_name:<16s} {len(onames):>6d} {n_aa:>4d} {n_ab:>4d} {n_bb:>4d} {q_edges:>7d}")

            # ── Detailed per-support output ──

            for supp in sorted(support_groups, key=lambda s: (len(s), sorted(s))):
                onames = sorted(support_groups[supp])
                base_name = edges_to_name(sorted(supp), n)

                print(f"\n    --- {base_name} (coeff={len(onames)}) ---")

                # Print each orbit (cap at 30 for readability)
                show_detail = len(onames) <= 200
                if not show_detail:
                    print(f"      ({len(onames)} orbits, details suppressed)")
                else:
                    for oname in onames:
                        tag = classify_aabb(oname)
                        sz = len(orbit_vertices[oname])
                        pa, pb = orbit_parent_pair[oname]
                        nbrs = orbit_neighbor_counts.get(oname, {})
                        # Compact neighbor string: just orbit_name:count
                        nbr_parts = [f"{nb}:{ct}" for nb, ct in sorted(nbrs.items())]
                        nbr_str = " ".join(nbr_parts) if nbr_parts else "—"
                        print(f"      {oname:<20s} [{tag}] sz={sz:>5d}  = {pa} * {pb}")
                        if nbr_parts:
                            print(f"        {'':20s} nbrs: {nbr_str}")

            # ── Product table ──
            print(f"\n    --- Product Table (grade {k}, {len(product_table)} products) ---")
            sorted_products = sorted(product_table.items())
            if len(sorted_products) > 50:
                print(f"      ({len(sorted_products)} product types, showing first 50)")
                sorted_products = sorted_products[:50]
            for pp, children in sorted_products:
                children_s = sorted(children)
                print(f"      {pp[0]} * {pp[1]}  ->  {', '.join(children_s)}")

            # ── D(v): out-degree of each grade-(k-1) orbit as parent into grade k ──
            # For each orbit at grade k, both parents contribute.
            # D(v) = number of grade-k orbits that have v as a parent.
            parent_out_degree = defaultdict(int)
            for oname in orbit_vertices:
                pa, pb = orbit_parent_pair[oname]
                parent_out_degree[pa] += 1
                parent_out_degree[pb] += 1
                # Self-products: edge {α,α} has α as both endpoints,
                # but the orbit only appears once, so α gets +1 not +2.
                # Correction: undo the double-count for self-products
                if pa == pb:
                    parent_out_degree[pa] -= 1

            # Report D(v) for previous grade's orbits, grouped by support
            if k >= 2 and orbits_at_grade.get(k - 1):
                prev_orbits = orbits_at_grade[k - 1]
                # Group by support
                prev_by_support = defaultdict(list)
                for oname in prev_orbits:
                    prev_by_support[support_of_orbit[oname]].append(oname)

                print(f"\n    --- D(v) out-degrees from grade {k-1} into grade {k} ---")
                for supp in sorted(prev_by_support, key=lambda s: (len(s), sorted(s))):
                    onames = sorted(prev_by_support[supp])
                    base_name = edges_to_name(sorted(supp), n)
                    d_values = []
                    for oname in onames:
                        d = parent_out_degree.get(oname, 0)
                        sz = len(prev_orbits[oname])
                        tag = classify_aabb(oname) if oname in orbit_parent_pair else "?"
                        d_values.append((oname, d, sz, tag))

                    # Check uniformity
                    d_by_sz = defaultdict(set)
                    for _, d, sz, _ in d_values:
                        d_by_sz[sz].add(d)
                    uniform = all(len(vs) == 1 for vs in d_by_sz.values())

                    if uniform and len(d_values) > 4:
                        # Compact: just show D by size class
                        for sz_val in sorted(d_by_sz):
                            d_val = next(iter(d_by_sz[sz_val]))
                            count = sum(1 for _, _, sz, _ in d_values if sz == sz_val)
                            print(f"      {base_name} sz={sz_val}: D={d_val}  ({count} orbits, UNIFORM)")
                    else:
                        for oname, d, sz, tag in d_values:
                            print(f"      {oname:<20s} sz={sz:>5d} [{tag}] D={d}")

            print()

        tree_label.cache_clear()
        canonicalize.cache_clear()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Experiment 8: ALG Iteration for K_{1,4} fiber
# ─────────────────────────────────────────────

def experiment_alg_iteration(max_k: int = 12):
    """
    Iterate the Augmented Line Graph (ALG) for the K_{1,4} fiber in L^k(K_5)/S_5.

    Phase 1 (structural): Build G_k⁺ explicitly, compute all stats.
    Phase 2 (algebraic):  Use closed recurrences for (f, h, T, adj, E).

    Recurrence system (derived from ALG blowup rules):
      adj_{k+1} = 2 · adj_k                          (★-edges double each grade)
      T_{k+1}   = (E_k - f_k + adj_k) + 5·T_k        (μ=4 sibling cross-edges)
      f_{k+1}   = m_k + 2·(E_k - f_k) + 2·T_k + 2·adj_k
      h_{k+1}   = m_k                                 (self-loop children are sz=60)
      m_{k+1}   = 2·f_{k+1} - h_{k+1}
      S_{k+1}   = T_{k+1} + (f_k - h_k)
      E_{k+1}   = f_{k+1} + S_{k+1} + E_cross_{k+1}  (needs degree sequence)

    E_cross can be computed one grade beyond the structural phase using
    D_child formulas that depend only on G_k⁺ incidence data.
    """
    print("=" * 70)
    print("EXPERIMENT 8: ALG Iteration for K_{1,4} fiber (n=5)")
    print("=" * 70)

    # ── Seed: Grade 4 ──
    # 2 orbits (K1,4_0, K1,4_1), both sz=60, siblings, both ★-adjacent
    n = 2
    sz = [60, 60]
    star_adj = [True, True]
    adj = [{0, 1}, {0, 1}]
    sib = {frozenset({0, 1})}

    header = (f"    {'Grade':<8s} {'f':>12s} {'h':>10s} {'E':>16s} "
              f"{'T':>12s} {'adj':>8s} {'m':>12s} {'phase':>8s}")
    print(f"\n{header}")
    print(f"    {'-'*88}")
    print(f"    {4:<8d} {2:>12d} {2:>10d} {3:>16d} "
          f"{0:>12d} {2:>8d} {2:>12d} {'seed':>8s}")

    # Track T_k (sibling cross-edges where both sz=120)
    # At grade 4: T_4 = 0 (the one cross-edge is between two sz=60 orbits)
    T_prev = 0

    # Track stats for algebraic projection
    f_prev, h_prev, E_prev, adj_prev, m_prev = 2, 2, 3, 2, 2

    # Last G_k⁺ incidence data for D_child projection
    last_gplus = None  # will hold (edges, inc, D_factor) from last structural grade

    STRUCTURAL_LIMIT = 100_000  # max f for building full adjacency

    for k in range(4, max_k):
        if n is not None and n <= STRUCTURAL_LIMIT:
            # ═══════════ PHASE 1: STRUCTURAL ═══════════
            STAR = n

            # ── Build edges of G_k⁺ ──
            edges = []
            for i in range(n):
                mu = 1 if sz[i] == 60 else 2
                edges.append((i, i, mu))

            seen = set()
            for i in range(n):
                for j in adj[i]:
                    if j > i:
                        pair = frozenset({i, j})
                        if pair not in seen:
                            seen.add(pair)
                            is_sib = pair in sib
                            both_120 = (sz[i] == 120 and sz[j] == 120)
                            mu = 4 if (is_sib and both_120) else 2
                            edges.append((i, j, mu))

            for i in range(n):
                if star_adj[i]:
                    edges.append((STAR, i, 2))

            # ── Compute stats ──
            f_new = sum(mu for _, _, mu in edges)
            h_new = sum(mu for u, v, mu in edges if u == v)
            adj_new = sum(mu for u, v, mu in edges if u == STAR or v == STAR)
            m_new = 2 * f_new - h_new

            inc = defaultdict(list)
            for ei, (u, v, mu) in enumerate(edges):
                inc[u].append(ei)
                if u != v:
                    inc[v].append(ei)

            D_factor = {}
            D2_factor = {}
            for v, eis in inc.items():
                D_factor[v] = sum(edges[ei][2] for ei in eis)
                D2_factor[v] = sum(edges[ei][2] ** 2 for ei in eis)

            E_cross = sum((D_factor[v] ** 2 - D2_factor[v]) // 2 for v in D_factor)
            s_new = sum(mu * (mu - 1) // 2 for _, _, mu in edges)
            E_new = f_new + s_new + E_cross

            # Compute T (sibling cross-edges where both sz=120)
            T_new = 0
            for u, v, mu in edges:
                if u != v and u != STAR and v != STAR:
                    if frozenset({u, v}) in sib and sz[u] == 120 and sz[v] == 120:
                        T_new += 1
                # ★-edges: children are sz=120 and siblings → T counts them?
                # No: T counts edges of Γ̃_{k+1} that are sibling AND both sz=120.
                # ★-edge children are sz=120 siblings → they DO count.
                if (u == STAR or v == STAR) and mu >= 2:
                    # ★-edge with μ=2: 1 sibling pair, both sz=120 → +1
                    pass  # handled below

            # Recompute T properly from the μ rules
            # T_{k+1} = number of cross-edges in Γ̃_{k+1} with μ=4 in G_{k+1}⁺
            #         = number of sibling pairs at grade k+1 where both are sz=120
            # Children from self-loops: sz=60, so sibling pairs don't count
            # Children from cross-edges (μ=2): 1 pair, both sz=120 → count 1
            # Children from cross-edges (μ=4): C(4,2)=6 pairs, both sz=120 → count 6
            # Children from ★-edges (μ=2): 1 pair, both sz=120 → count 1
            T_new = 0
            for u, v, mu in edges:
                if u == v:
                    continue  # self-loop children are sz=60
                pairs = mu * (mu - 1) // 2
                T_new += pairs  # all cross/★ children are sz=120

            # Verify T recurrence: T_{k+1} = (E_k - f_k + adj_k) + 5·T_k
            T_predicted = (E_prev - f_prev + adj_prev) + 5 * T_prev
            T_check = "✓" if T_new == T_predicted else f"✗ (pred={T_predicted})"

            print(f"    {k+1:<8d} {f_new:>12d} {h_new:>10d} {E_new:>16d} "
                  f"{T_new:>12d} {adj_new:>8d} {m_new:>12d} {'struct':>8s}"
                  f"  T-check:{T_check}")

            # ── Save G_k⁺ data for D_child projection ──
            last_gplus = (edges, inc, D_factor, D2_factor, n)

            # ── Update algebraic state ──
            T_prev = T_new
            f_prev, h_prev, E_prev, adj_prev, m_prev = f_new, h_new, E_new, adj_new, m_new

            # ── Build adjacency for next iteration (if small enough) ──
            if f_new > STRUCTURAL_LIMIT:
                n = None  # switch to algebraic phase
                continue

            new_sz = []
            new_star_adj = []
            edge_to_oids = []
            oid = 0
            for u, v, mu in edges:
                ids = list(range(oid, oid + mu))
                edge_to_oids.append(ids)
                for _ in range(mu):
                    new_sz.append(60 if u == v else 120)
                    new_star_adj.append(u == STAR or v == STAR)
                oid += mu

            new_adj = [set() for _ in range(f_new)]
            for v_gk, eis in inc.items():
                all_oids = []
                for ei in eis:
                    all_oids.extend(edge_to_oids[ei])
                oid_set = set(all_oids)
                for a in all_oids:
                    new_adj[a].update(oid_set)

            new_sib = set()
            for ids in edge_to_oids:
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        new_sib.add(frozenset({ids[i], ids[j]}))

            n = f_new
            sz = new_sz
            star_adj = new_star_adj
            adj = new_adj
            sib = new_sib

        else:
            # ═══════════ PHASE 2: ALGEBRAIC + DEEP D-PROJECTION ═══════════

            if last_gplus is not None:
                # Compute as many grades as possible using iterated D-proj.
                # Level 1: D_child from G_k⁺ edges → D for grade k+1 factors → E_{k+1}
                # Level 2: iterate over G_k⁺ clique pairs → G_{k+1}⁺ edge types → E_{k+2}
                # etc.

                edges_gk, inc_gk, D_fac, D2_fac, n_gk = last_gplus
                STAR_gk = n_gk  # ★ index in G_k⁺

                # ── Helper: D_child and D2_child for an edge given endpoint D values ──
                def d_child_of_edge(Du, Dv, mu_e, is_self, is_star):
                    """Compute (D_child, D2_child) for children of edge (u,v,μ).
                    Du, Dv = D values of endpoints u, v.
                    """
                    if is_self:
                        # Self-loop: children sz=60
                        # D = 2·D(u) - 1, D₂ = 4·D(u) - 3
                        return (2*Du - 1, 4*Du - 3)
                    elif is_star:
                        # ★-edge: children sz=120, ★-adj
                        # D = 2·D(★) + 2·D(v), D₂ = 8μ + 4D(★) + 4D(v) - 8
                        D = 2*Du + 2*Dv
                        D2 = 8*mu_e + 4*Du + 4*Dv - 8
                        return (D, D2)
                    else:
                        # Regular cross: children sz=120
                        # D = 2D(u) + 2D(v) - 2, D₂ = 8μ + 4D(u) + 4D(v) - 12
                        D = 2*Du + 2*Dv - 2
                        D2 = 8*mu_e + 4*Du + 4*Dv - 12
                        return (D, D2)

                # ── Level 1: Compute D_child for each G_k⁺ edge ──
                Dc = {}   # edge_index → D_child
                D2c = {}  # edge_index → D2_child
                for ei, (u, v, mu_e) in enumerate(edges_gk):
                    is_self = (u == v)
                    is_star = (u == STAR_gk or v == STAR_gk)
                    Du = D_fac[u]
                    Dv = D_fac[v] if not is_self else Du
                    Dc[ei], D2c[ei] = d_child_of_edge(Du, Dv, mu_e, is_self, is_star)

                # ── Level 1: Compute E for the D-proj grade ──
                adj_new = 2 * adj_prev
                T_new = (E_prev - f_prev + adj_prev) + 5 * T_prev
                f_new = m_prev + 2 * (E_prev - f_prev) + 2 * T_prev + 2 * adj_prev
                h_new = m_prev
                m_new = 2 * f_new - h_new
                s_new = T_new + (f_prev - h_prev)

                E_cross_1 = sum(mu * (Dc[ei]**2 - D2c[ei]) // 2
                                for ei, (_, _, mu) in enumerate(edges_gk))
                D_star_1 = 2 * adj_new
                E_cross_1 += (D_star_1**2 - 4 * adj_new) // 2
                E_new = f_new + s_new + E_cross_1

                print(f"    {k+1:<8d} {f_new:>12d} {h_new:>10d} {E_new:>16d} "
                      f"{T_new:>12d} {adj_new:>8d} {m_new:>12d} {'D-proj':>8s}")

                T_prev = T_new
                f_prev, h_prev, E_prev, adj_prev, m_prev = f_new, h_new, E_new, adj_new, m_new
                k += 1

                if k >= max_k:
                    last_gplus = None
                    continue

                # ── Level 2: Compute E one more grade via clique-pair iteration ──
                # G_{k}⁺ edges decompose into 4 types derived from G_{k-1}⁺:
                #   (a) Self-loops on each child
                #   (b) Sibling cross-edges within each G_{k-1}⁺ edge's children
                #   (c) Non-sibling cross-edges: clique pairs from shared factors
                #   (d) ★-edges to ★-adj children
                # For each type, use D_child(level 1) as endpoint D values.

                adj_new2 = 2 * adj_prev
                T_new2 = (E_prev - f_prev + adj_prev) + 5 * T_prev
                f_new2 = m_prev + 2 * (E_prev - f_prev) + 2 * T_prev + 2 * adj_prev
                h_new2 = m_prev
                m_new2 = 2 * f_new2 - h_new2
                s_new2 = T_new2 + (f_prev - h_prev)

                D_star_2 = 2 * adj_new2  # D(★) in G_{k+1}⁺

                E_cross_2 = 0

                # (a) Self-loops: one per child orbit, grouped by G_{k-1}⁺ edge
                for ei, (u, v, mu_e) in enumerate(edges_gk):
                    child_sz = 60 if u == v else 120
                    mu_self = 1 if child_sz == 60 else 2
                    Da = Dc[ei]
                    Dc2, D2c2 = d_child_of_edge(Da, Da, mu_self, True, False)
                    # mu_e children from this edge, each with self-loop of weight mu_self
                    E_cross_2 += mu_e * mu_self * (Dc2**2 - D2c2) // 2

                # (b) Sibling cross-edges within each G_{k-1}⁺ edge's children
                for ei, (u, v, mu_e) in enumerate(edges_gk):
                    if mu_e < 2:
                        continue
                    child_sz = 60 if u == v else 120
                    # Sibling cross μ: 4 if both sz=120, else 2
                    mu_sib = 4 if child_sz == 120 else 2
                    n_sib_pairs = mu_e * (mu_e - 1) // 2
                    Da = Dc[ei]
                    Dc2, D2c2 = d_child_of_edge(Da, Da, mu_sib, False, False)
                    E_cross_2 += n_sib_pairs * mu_sib * (Dc2**2 - D2c2) // 2

                # (c) Non-sibling cross-edges from clique pairs
                print(f"      Computing E via clique-pair iteration over "
                      f"{len(inc_gk)} factors...")
                import sys
                n_factors_done = 0
                for v_gk, eis_v in inc_gk.items():
                    n_factors_done += 1
                    if n_factors_done % 500 == 0:
                        print(f"        factor {n_factors_done}/{len(inc_gk)}...",
                              file=sys.stderr)
                    nv = len(eis_v)
                    if nv < 2:
                        continue
                    for ii in range(nv):
                        ei = eis_v[ii]
                        mu_i = edges_gk[ei][2]
                        Di = Dc[ei]
                        for jj in range(ii + 1, nv):
                            ej = eis_v[jj]
                            mu_j = edges_gk[ej][2]
                            Dj = Dc[ej]
                            # mu_i × mu_j child pairs, each cross-edge μ=2
                            Dc2, D2c2 = d_child_of_edge(Di, Dj, 2, False, False)
                            E_cross_2 += mu_i * mu_j * 2 * (Dc2**2 - D2c2) // 2

                # (d) ★-edges: each ★-adj child gets edge to ★
                for ei, (u, v, mu_e) in enumerate(edges_gk):
                    is_star_parent = (u == STAR_gk or v == STAR_gk)
                    if not is_star_parent:
                        continue
                    Da = Dc[ei]
                    # ★-edge in G_k⁺ has μ=2, children are ★-adj
                    Dc2, D2c2 = d_child_of_edge(D_star_2, Da, 2, False, True)
                    E_cross_2 += mu_e * 2 * (Dc2**2 - D2c2) // 2

                # ★ factor contribution
                E_cross_2 += (D_star_2**2 - 4 * adj_new2) // 2

                E_new2 = f_new2 + s_new2 + E_cross_2

                print(f"    {k+1:<8d} {f_new2:>12d} {h_new2:>10d} {E_new2:>16d} "
                      f"{T_new2:>12d} {adj_new2:>8d} {m_new2:>12d} {'D-proj2':>8s}")

                T_prev = T_new2
                f_prev, h_prev, E_prev, adj_prev, m_prev = (
                    f_new2, h_new2, E_new2, adj_new2, m_new2)

                last_gplus = None
                continue

            # Beyond D-proj range: only f, h, T, adj (no E)
            adj_new = 2 * adj_prev
            T_new = (E_prev - f_prev + adj_prev) + 5 * T_prev
            f_new = m_prev + 2 * (E_prev - f_prev) + 2 * T_prev + 2 * adj_prev
            h_new = m_prev
            m_new = 2 * f_new - h_new

            print(f"    {k+1:<8d} {f_new:>12d} {h_new:>10d} {'—':>16s} "
                  f"{T_new:>12d} {adj_new:>8d} {m_new:>12d} {'alg':>8s}")

            # Cannot continue without E
            print(f"\n  *** Cannot compute further: E_{k+1} requires structural data ***")
            print(f"  *** f_{k+2} needs E_{k+1} which needs degree sequence of G_{k+1}⁺ ***")
            break

    # ── Summary ──
    print(f"\n  Recurrence system:")
    print(f"    adj_{{k+1}} = 2·adj_k")
    print(f"    T_{{k+1}}   = (E_k - f_k + adj_k) + 5·T_k")
    print(f"    f_{{k+1}}   = m_k + 2·(E_k - f_k) + 2·T_k + 2·adj_k")
    print(f"    h_{{k+1}}   = m_k")
    print(f"    m_{{k+1}}   = 2·f_{{k+1}} - m_k")
    print(f"    E_{{k+1}}   = f_{{k+1}} + S_{{k+1}} + Σ_v (D(v)²-D₂(v))/2  [structural]")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-n", type=int, default=6)
    ap.add_argument("--max-k", type=int, default=7)
    ap.add_argument("--exp", type=str, default="all",
                    help="Which experiment: 1, 2, 3, 4, 5, 6, or 'all'")
    ap.add_argument("--trees-only", action="store_true",
                    help="For exp 7: restrict to tree supports only (prune cycles)")
    ap.add_argument("--target", type=str, default=None,
                    help="For exp 7: only keep supports that are subgraphs of TARGET "
                         "(e.g. K1,4, K1,3, P4, P5, fork, paw, diamond, C4)")
    args = ap.parse_args()

    targets = [
        ("K_{1,3}", [(1, 2), (1, 3), (1, 4)]),
        ("K_{1,4}", [(1, 2), (1, 3), (1, 4), (1, 5)]),
        ("paw",     [(1, 2), (1, 3), (1, 4), (2, 3)]),
        ("C3",      [(1, 2), (2, 3), (1, 3)]),
        ("C4",      [(1, 2), (2, 3), (3, 4), (1, 4)]),
        ("P4",      [(1, 2), (2, 3), (3, 4)]),
        ("P5",      [(1, 2), (2, 3), (3, 4), (4, 5)]),
        ("diamond", [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]),
    ]

    if args.exp in ("1", "all"):
        experiment_coefficient_table(max_n=args.max_n, max_k=args.max_k)

    if args.exp in ("2", "all"):
        experiment_k14_recurrence(max_n=min(args.max_n, 7), max_k=args.max_k)

    if args.exp in ("3", "all"):
        experiment_fiber_closure(max_n=args.max_n, max_k=args.max_k)

    if args.exp in ("4", "all"):
        experiment_fiber_structure(targets, max_n=args.max_n, max_k=args.max_k)

    if args.exp in ("5", "all"):
        experiment_deep_paw(max_k=args.max_k)

    if args.exp in ("6", "all"):
        experiment_k14_lightweight(max_n=args.max_n, max_k=args.max_k)

    if args.exp in ("7",):
        experiment_quotient(max_n=args.max_n, max_k=args.max_k,
                            trees_only=args.trees_only,
                            target=args.target)

    if args.exp in ("8",):
        experiment_alg_iteration(max_k=args.max_k)