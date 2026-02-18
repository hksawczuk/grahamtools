#!/usr/bin/env python3
"""
Search for graph pairs that are provably Graham-equivalent at ALL grades
but NOT WL-1 equivalent.

Permanent null space (2D, verified through k=8):
  v1: Delta(C3) = +1, Delta(K13) = -1         [triangle/star trade]
  v2: Delta(C3) = -4, Delta(P4) = +12, Delta(C4) = +3  [triangle/path/square trade]

Any integer combination a*v1 + b*v2 gives a valid permanent null vector.
Two graphs G, G' are provably Graham-equivalent at all grades if:
  count(tau, G) = count(tau, G') for all tau NOT in {C3, K13, C4, P4}
  AND the vector (Delta(C3), Delta(K13), Delta(P4), Delta(C4)) is in span{v1, v2}.

Equivalently, define two Graham-invariant quantities:
  I1 = count(C3) + count(K13)                    [from v1: C3+K13 is invariant]
  I2 = 3*count(C3) + 3*count(K13) + 4*count(C4)  [asymptotic Graham contribution]
  I3 = count(P4) + 4*count(C4)                    [from v2 at k=3; see derivation]

Actually let me derive the invariants properly.

The Graham sequence sees {C3, K13, P4, C4} through the coefficient vectors:
  c(C3)  = [0, 0, 3, 3, 3, 3, ...]
  c(K13) = [0, 0, 3, 3, 3, 3, ...]
  c(P4)  = [0, 0, 1, 0, 0, 0, ...]
  c(C4)  = [0, 0, 0, 4, 4, 4, ...]

The contribution to gamma_k from these four types is:
  k=1,2: 0  (all coefficients zero)
  k=3:   3*(c3 + k13) + 1*p4
  k>=4:  3*(c3 + k13) + 4*c4

So the Graham sequence is determined by two combinations:
  A = 3*(c3 + k13) + p4      [determines k=3 contribution]
  B = 3*(c3 + k13) + 4*c4    [determines k>=4 contribution]

Or equivalently:
  S = c3 + k13   (sum, since coefficients identical)
  B = 3*S + 4*c4
  A = 3*S + p4

So the Graham-invariant signature is: (all other counts, S, p4, c4)
which means c3 and k13 individually don't matter, only their sum S.
And p4 and c4 ARE individually constrained!

Wait, that means the only freedom is the v1 direction (C3/K13 trade).
The v2 direction changes p4 and c4, which ARE individually visible
to the Graham sequence (p4 at k=3, c4 at k>=4).

Let me recheck v2. With Delta(C3)=-4, Delta(P4)=+12, Delta(C4)=+3:
  k=3: 3*(-4) + 1*(12) + 0*(3) = -12 + 12 = 0  ✓
  k=4: 3*(-4) + 0*(12) + 4*(3) = -12 + 12 = 0   ✓
  k>=5: same as k=4  ✓

So v2 works because at k=3, the P4 change (+12) cancels the C3 change
(3*(-4)=-12), and at k>=4, the C4 change (4*3=12) cancels the C3 change
(3*(-4)=-12). The P4 change is invisible at k>=4 (coeff=0) and the C4
change is invisible at k=3 (coeff=0).

So v2 IS a valid permanent null direction. The Graham sequence does NOT
individually constrain p4 and c4 — it constrains specific combinations.

Let me redo the invariants. The Graham sequence constrains:
  k=3: 3*(c3+k13) + p4                     = const
  k>=4: 3*(c3+k13) + 4*c4                  = const

These are TWO constraints on FOUR unknowns (c3, k13, p4, c4).
The null space is 4 - 2 = 2 dimensional, spanned by v1 and v2. ✓

So the Graham-invariant signature should encode:
  Q1 = 3*(c3 + k13) + p4        [k=3 invariant]
  Q2 = 3*(c3 + k13) + 4*c4      [k>=4 invariant]
  + all other subgraph counts unchanged

Usage: python3 search_full_null.py [--max-n 8] [--max-sub-edges 5]
"""

import sys
import time
from collections import defaultdict
from itertools import combinations, permutations


# ============================================================
#  Core utilities
# ============================================================

def canonical_subgraph(edges):
    if not edges:
        return ()
    verts = set()
    for u, v in edges:
        verts.add(u)
        verts.add(v)
    vlist = sorted(verts)
    n = len(vlist)
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


def canonical_graph(edges, n):
    if not edges:
        return ()
    best = None
    for perm in permutations(range(n)):
        relabeled = tuple(sorted(
            (min(perm[u], perm[v]), max(perm[u], perm[v]))
            for u, v in edges
        ))
        if best is None or relabeled < best:
            best = relabeled
    return best


def is_connected(edges):
    if not edges:
        return True
    adj = defaultdict(set)
    verts = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        verts.add(u)
        verts.add(v)
    start = next(iter(verts))
    visited = set()
    stack = [start]
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        for u in adj[v]:
            if u not in visited:
                stack.append(u)
    return len(visited) == len(verts)


def wl1_hash(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    colors = [len(adj[v]) for v in range(n)]
    for iteration in range(n):
        new_colors_raw = []
        for v in range(n):
            neighbor_colors = tuple(sorted(colors[u] for u in adj[v]))
            new_colors_raw.append((colors[v], neighbor_colors))
        mapping = {}
        counter = 0
        for raw in sorted(set(new_colors_raw)):
            mapping[raw] = counter
            counter += 1
        new_colors = [mapping[new_colors_raw[v]] for v in range(n)]
        if sorted(new_colors) == sorted(colors):
            break
        colors = new_colors
    return tuple(sorted(colors))


def line_graph(edges, n_vertices):
    m = len(edges)
    if m == 0:
        return [], 0
    incident = defaultdict(list)
    for idx, (u, v) in enumerate(edges):
        incident[u].append(idx)
        incident[v].append(idx)
    new_edges = set()
    for v, inc in incident.items():
        for i in range(len(inc)):
            for j in range(i + 1, len(inc)):
                a, b = inc[i], inc[j]
                if a > b: a, b = b, a
                new_edges.add((a, b))
    return sorted(new_edges), m


def gamma_sequence(edges, max_k, max_edges=1_000_000):
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
        new_edges, new_n = line_graph(current_edges, current_n)
        seq.append(new_n)
        if new_n == 0 or len(new_edges) > max_edges:
            while len(seq) <= max_k:
                seq.append(None)
            break
        current_edges = new_edges
        current_n = new_n
    return seq


# ============================================================
#  Reference canonical forms
# ============================================================

C3_CANON = canonical_subgraph([(0, 1), (1, 2), (0, 2)])
K13_CANON = canonical_subgraph([(0, 1), (0, 2), (0, 3)])
P4_CANON = canonical_subgraph([(0, 1), (1, 2), (2, 3)])
C4_CANON = canonical_subgraph([(0, 1), (1, 2), (2, 3), (0, 3)])

TRADE_TYPES = {C3_CANON, K13_CANON, P4_CANON, C4_CANON}


# ============================================================
#  Graph enumeration
# ============================================================

def enumerate_graphs(n, m):
    all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
    max_edges = len(all_possible)
    if m > max_edges or m < n - 1:
        return []
    seen = set()
    graphs = []
    for subset in combinations(range(max_edges), m):
        edges = [all_possible[i] for i in subset]
        if not is_connected(edges):
            continue
        canon = canonical_graph(edges, n)
        if canon in seen:
            continue
        seen.add(canon)
        graphs.append(edges)
    return graphs


# ============================================================
#  Main search
# ============================================================

def main():
    max_n = 8
    max_sub_edges = 5

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--max-n" and i + 1 < len(args):
            max_n = int(args[i + 1]); i += 2
        elif args[i] == "--max-sub-edges" and i + 1 < len(args):
            max_sub_edges = int(args[i + 1]); i += 2
        else:
            i += 1

    print(f"Search for Graham-equivalent, non-WL1-equivalent pairs")
    print(f"  Permanent null space (2D):")
    print(f"    v1: Delta(C3)=+1, Delta(K13)=-1")
    print(f"    v2: Delta(C3)=-4, Delta(P4)=+12, Delta(C4)=+3")
    print(f"  Graham invariants:")
    print(f"    Q1 = 3*(c3+k13) + p4    [k=3]")
    print(f"    Q2 = 3*(c3+k13) + 4*c4  [k>=4]")
    print(f"  max n: {max_n}, max subgraph edges: {max_sub_edges}")
    print(f"  C3={C3_CANON}, K13={K13_CANON}, P4={P4_CANON}, C4={C4_CANON}")
    print()

    total_found = 0
    total_graham_verified = 0

    for n in range(4, max_n + 1):
        min_m = n - 1
        max_m = n * (n - 1) // 2

        for m in range(min_m, max_m + 1):
            t0 = time.time()

            # Skip very large cases
            max_possible = 1
            for ii in range(min(m, 20)):
                max_possible = max_possible * (n * (n - 1) // 2 - ii) // (ii + 1)
                if max_possible > 10_000_000:
                    break
            if max_possible > 10_000_000:
                print(f"  n={n}, m={m}: skipped (too large)", flush=True)
                continue

            graphs = enumerate_graphs(n, m)
            ng = len(graphs)
            if ng < 2:
                continue

            # Compute subgraph counts and Graham-invariant signature
            signatures = defaultdict(list)

            for gi, edges in enumerate(graphs):
                counts = defaultdict(int)
                me = len(edges)
                for size in range(1, min(max_sub_edges, me) + 1):
                    for subset in combinations(range(me), size):
                        sub_edges = [edges[ii] for ii in subset]
                        if not is_connected(sub_edges):
                            continue
                        canon = canonical_subgraph(sub_edges)
                        counts[canon] += 1

                c3 = counts.get(C3_CANON, 0)
                k13 = counts.get(K13_CANON, 0)
                p4 = counts.get(P4_CANON, 0)
                c4 = counts.get(C4_CANON, 0)

                # Graham invariants
                q1 = 3 * (c3 + k13) + p4       # determines k=3
                q2 = 3 * (c3 + k13) + 4 * c4   # determines k>=4

                # Signature: all non-trade counts + Q1 + Q2
                sig_parts = []
                for canon_key in sorted(counts.keys()):
                    if canon_key in TRADE_TYPES:
                        continue
                    sig_parts.append((canon_key, counts[canon_key]))
                sig_parts.append(("Q1", q1))
                sig_parts.append(("Q2", q2))
                sig = tuple(sig_parts)

                signatures[sig].append((gi, edges, c3, k13, p4, c4, dict(counts)))

            # Check groups
            for sig, group in signatures.items():
                if len(group) < 2:
                    continue

                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        gi, ei, c3i, k13i, p4i, c4i, ci = group[i]
                        gj, ej, c3j, k13j, p4j, c4j, cj = group[j]

                        # Must differ in at least one of the trade types
                        if c3i == c3j and k13i == k13j and p4i == p4j and c4i == c4j:
                            continue

                        # Check WL-1
                        wl1_i = wl1_hash(ei, n)
                        wl1_j = wl1_hash(ej, n)

                        if wl1_i == wl1_j:
                            continue

                        total_found += 1

                        # Verify with actual Graham sequences
                        gseq_i = gamma_sequence(ei, 7)
                        gseq_j = gamma_sequence(ej, 7)

                        graham_match = all(
                            gseq_i[k] == gseq_j[k]
                            for k in range(min(len(gseq_i), len(gseq_j)))
                            if gseq_i[k] is not None and gseq_j[k] is not None
                        )

                        if graham_match:
                            total_graham_verified += 1

                        # Determine which null space directions are used
                        dc3 = c3i - c3j
                        dk13 = k13i - k13j
                        dp4 = p4i - p4j
                        dc4 = c4i - c4j

                        # Delta = a*v1 + b*v2 where
                        # v1 = (C3:+1, K13:-1, P4:0, C4:0)
                        # v2 = (C3:-4, K13:0, P4:+12, C4:+3)
                        # So: dp4 = 12*b => b = dp4/12
                        #     dc4 = 3*b  => b = dc4/3
                        #     dk13 = -a  => a = -dk13
                        #     dc3 = a - 4*b

                        b = dp4 / 12 if dp4 != 0 else dc4 / 3 if dc4 != 0 else 0
                        a = -dk13

                        print(f"\n  {'='*60}")
                        print(f"  PAIR #{total_found} (n={n}, m={m})")
                        print(f"  {'='*60}")
                        print(f"    Graph 1: {ei}")
                        print(f"    Graph 2: {ej}")
                        print(f"    Delta: C3={dc3:+d}, K13={dk13:+d}, P4={dp4:+d}, C4={dc4:+d}")
                        print(f"    Null space coords: a={a}, b={b} (Delta = a*v1 + b*v2)")
                        print(f"    WL-1 equivalent: False")
                        print(f"    Graham match (k<=7): {graham_match}")
                        if graham_match:
                            print(f"    *** PROVABLY GRAHAM-EQUIVALENT, NOT WL-1 EQUIVALENT ***")
                        else:
                            print(f"    Graham 1: {gseq_i}")
                            print(f"    Graham 2: {gseq_j}")
                            print(f"    WARNING: Graham sequences differ!")
                            print(f"    This means a NON-permanent dependence contributes.")
                            # Show which other counts differ
                            all_keys = sorted(set(ci.keys()) | set(cj.keys()))
                            for ck in all_keys:
                                if ck in TRADE_TYPES:
                                    continue
                                vi = ci.get(ck, 0)
                                vj = cj.get(ck, 0)
                                if vi != vj:
                                    print(f"      OTHER TYPE {ck}: {vi} vs {vj}")

            elapsed = time.time() - t0
            n_groups = sum(1 for g in signatures.values() if len(g) > 1)
            if ng >= 2:
                print(f"  n={n}, m={m}: {ng} graphs, {n_groups} sig-groups ({elapsed:.1f}s)",
                      flush=True)

    print(f"\n{'='*70}")
    print(f"  Total candidate pairs: {total_found}")
    print(f"  Graham-verified pairs: {total_graham_verified}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()