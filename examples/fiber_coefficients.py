#!/usr/bin/env python3
"""
Fiber coefficient computation and analysis for iterated line graphs.

Subcommands
-----------
kn (default)
    Compute fiber coefficients from L^k(K_n) using the grahamtools
    level-generation / canonical-classification machinery.

bootstrap
    Compute fiber coefficients for all tree types up to a given edge
    count via Mobius inversion bootstrapping — much faster than kn for
    deep grades since it iterates line graphs of each small tree
    individually rather than building the full L^k(K_n).

matrix
    Hardcoded coefficient-matrix analysis for the 6 tree types on 6
    vertices in K_6 (bitmask-based fiber counting through iterated
    line graphs).

Usage examples
--------------
  python3 fiber_coefficients.py kn --n 5 --max-k 7
  python3 fiber_coefficients.py bootstrap --max-edges 4 --max-k 10
  python3 fiber_coefficients.py matrix
"""

import argparse
import math
import sys
import time
from collections import defaultdict
from itertools import combinations, product

# ---------------------------------------------------------------------------
# grahamtools imports
# ---------------------------------------------------------------------------
from grahamtools.kn import generate_levels_Kn_ids, expand_to_simple_base_edges_id
from grahamtools.kn.classify import canon_key
from grahamtools.utils.automorphisms import aut_size_edges, orbit_size_under_Sn
from grahamtools.utils.naming import tree_name, describe_graph
from grahamtools.utils.linalg import exact_rank
from grahamtools.utils.linegraph_edgelist import gamma_sequence_edgelist
from grahamtools.utils.subgraphs import enumerate_connected_subgraphs
from grahamtools.utils.canonical import canonical_graph_nauty, canonical_graph_bruteforce
from grahamtools.external.nauty import nauty_available
from grahamtools.invariants.fiber import compute_all_coefficients


# ===================================================================
#  Subcommand: kn  (fiber coefficients from L^k(K_n))
# ===================================================================

def cmd_kn(args: argparse.Namespace) -> None:
    """Compute fiber coefficients by classifying vertices of L^k(K_n)."""
    n = args.n
    max_k = args.max_k

    # ---- generate levels ----
    print(f"Generating L^k(K_{n}) for k=0..{max_k} (trees only)...")
    t0 = time.time()
    V_by_level, ep = generate_levels_Kn_ids(n, max_k, prune_cycles=True)
    print(f"  Done in {time.time() - t0:.1f}s")

    for k in range(max_k + 1):
        print(f"  Grade {k}: {len(V_by_level.get(k, []))} tree vertices")

    # ---- classify fibers per grade ----
    all_fibers: dict[object, dict] = {}

    for k in range(1, max_k + 1):
        Vk = V_by_level.get(k, [])
        if not Vk:
            continue

        t0 = time.time()

        buckets: dict[object, dict] = {}
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
            orbit = orbit_size_under_Sn(edges, n)
            coeff = freq // orbit

            if key not in all_fibers:
                name = tree_name(edges, n)
                all_fibers[key] = {
                    "name": name,
                    "edges": edges,
                    "aut": aut,
                    "orbit_sz": orbit,
                    "count_in_Kn": orbit,
                    "nedges": len(edges),
                    "coeff": {},
                    "fiber_sz": {},
                    "n_orbits": {},
                }

            all_fibers[key]["coeff"][k] = coeff
            all_fibers[key]["fiber_sz"][k] = freq
            all_fibers[key]["n_orbits"][k] = coeff

        if elapsed > 0.5:
            print(
                f"    Grade {k}: classified {len(Vk)} vertices into "
                f"{len(buckets)} fibers in {elapsed:.1f}s"
            )

    _print_kn_results(all_fibers, n, max_k)


def _print_kn_results(all_fibers: dict, n: int, max_k: int) -> None:
    """Print detailed results for the kn subcommand."""
    sorted_fibers = sorted(
        all_fibers.values(),
        key=lambda f: (f["nedges"], -max(f["fiber_sz"].values(), default=0)),
    )

    # ---- per-fiber detail ----
    print(f"\n{'=' * 80}")
    print(f"  Fiber coefficients for L^k(K_{n})")
    print(f"{'=' * 80}")

    for fiber in sorted_fibers:
        name = fiber["name"]
        ne = fiber["nedges"]
        aut = fiber["aut"]
        orbit = fiber["orbit_sz"]

        print(
            f"\n  {name}  ({ne} edges, |Aut|={aut}, orbit_sz={orbit}, "
            f"count_in_K{n}={orbit})"
        )
        print(
            f"    {'k':>4s} {'coeff_k':>12s} {'fiber_sz':>12s} "
            f"{'#orbits':>10s} {'ratio':>10s}"
        )

        prev_coeff = None
        for k in sorted(fiber["coeff"]):
            c = fiber["coeff"][k]
            fs = fiber["fiber_sz"][k]
            no = fiber["n_orbits"][k]
            ratio = c / prev_coeff if prev_coeff and prev_coeff > 0 else None
            print(
                f"    {k:>4d} {c:>12d} {fs:>12d} {no:>10d} "
                f"{f'{ratio:.2f}' if ratio else '---':>10s}"
            )
            prev_coeff = c

    # ---- summary table ----
    print(f"\n{'=' * 80}")
    print(f"  Summary: coeff_k(tau) for all fibers")
    print(f"{'=' * 80}")

    all_grades: set[int] = set()
    for f in sorted_fibers:
        all_grades.update(f["coeff"].keys())
    grades = sorted(all_grades)

    hdr = f"  {'Tree':>16s} {'|e|':>4s} {'|Aut|':>5s}"
    for k in grades:
        hdr += f" {'k=' + str(k):>10s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for fiber in sorted_fibers:
        row = f"  {fiber['name']:>16s} {fiber['nedges']:>4d} {fiber['aut']:>5d}"
        for k in grades:
            c = fiber["coeff"].get(k)
            row += f" {c:>10d}" if c is not None else f" {'':>10s}"
        print(row)

    # ---- growth rates ----
    print(f"\n{'=' * 80}")
    print(f"  Growth rates: coeff_k / coeff_{{k-1}}")
    print(f"{'=' * 80}")

    hdr = f"  {'Tree':>16s}"
    for k in grades[1:]:
        hdr += f" {'k=' + str(k):>10s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for fiber in sorted_fibers:
        row = f"  {fiber['name']:>16s}"
        for k in grades[1:]:
            c = fiber["coeff"].get(k)
            cp = fiber["coeff"].get(k - 1)
            if c is not None and cp is not None and cp > 0:
                row += f" {c / cp:>10.2f}"
            else:
                row += f" {'':>10s}"
        print(row)

    # ---- fiber fractions ----
    print(f"\n{'=' * 80}")
    print(f"  Fiber fractions: fiber_sz / |V(L^k)|")
    print(f"{'=' * 80}")

    hdr = f"  {'Tree':>16s}"
    for k in grades:
        hdr += f" {'k=' + str(k):>10s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    totals = {}
    for k in grades:
        totals[k] = sum(f["fiber_sz"].get(k, 0) for f in sorted_fibers)

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
                row += f" {fs / totals[k]:>10.4f}"
            else:
                row += f" {'':>10s}"
        print(row)


# ===================================================================
#  Subcommand: bootstrap  (Mobius inversion over tree types)
# ===================================================================

def _prufer_to_edges(seq: list[int]) -> list[tuple[int, int]]:
    """Convert Prufer sequence to edge list."""
    n = len(seq) + 2
    degree = [1] * n
    for v in seq:
        degree[v] += 1
    edges = []
    for v in seq:
        for u in range(n):
            if degree[u] == 1:
                edges.append((min(u, v), max(u, v)))
                degree[u] -= 1
                degree[v] -= 1
                break
    last = [u for u in range(n) if degree[u] == 1]
    edges.append((min(last[0], last[1]), max(last[0], last[1])))
    return edges


def _canonical(edges: list[tuple[int, int]]) -> object:
    """Canonical form using nauty if available, brute-force otherwise."""
    if not edges:
        return ()
    if nauty_available():
        return canonical_graph_nauty(edges)
    return canonical_graph_bruteforce(edges)


def _generate_all_trees(max_edges: int) -> dict[object, tuple[int, list[tuple[int, int]], int, int]]:
    """Generate all non-isomorphic trees with 1..max_edges edges.

    Returns dict of canonical_form -> (1, edges, n_verts, n_edges) compatible
    with the all_types format expected by compute_all_coefficients.
    """
    all_types: dict[object, tuple[int, list[tuple[int, int]], int, int]] = {}

    for ne in range(1, max_edges + 1):
        nv = ne + 1
        count = 0

        if nv == 2:
            edges = [(0, 1)]
            canon = _canonical(edges)
            if canon not in all_types:
                all_types[canon] = (1, edges, nv, ne)
                count += 1
        elif nv == 3:
            edges = [(0, 1), (1, 2)]
            canon = _canonical(edges)
            if canon not in all_types:
                all_types[canon] = (1, edges, nv, ne)
                count += 1
        else:
            for seq in product(range(nv), repeat=nv - 2):
                edges = _prufer_to_edges(list(seq))
                canon = _canonical(edges)
                if canon not in all_types:
                    all_types[canon] = (1, edges, nv, ne)
                    count += 1

        print(f"  {ne}-edge trees: {count} types")

    return all_types


def cmd_bootstrap(args: argparse.Namespace) -> None:
    """Compute fiber coefficients via Mobius inversion bootstrapping."""
    max_edges = args.max_edges
    max_k = args.max_k
    max_line_edges = args.max_line_edges

    print(f"{'=' * 70}")
    print(f"  Bootstrap: Mobius inversion for trees up to {max_edges} edges, max_k={max_k}")
    print(f"{'=' * 70}")

    # Step 1: enumerate all tree types
    print(f"\nEnumerating tree types...")
    all_types = _generate_all_trees(max_edges)
    print(f"  Total: {len(all_types)} tree types")

    # Step 2: compute coefficients via Mobius inversion
    print(f"\nComputing coefficients by Mobius inversion...")
    gammas, coeffs, subtypes = compute_all_coefficients(
        all_types, max_k, max_edges=max_line_edges, verbose=True,
    )

    # Step 3: print results
    _print_bootstrap_results(all_types, coeffs, gammas, max_k)


def _print_bootstrap_results(
    all_types: dict,
    coeffs: dict,
    gammas: dict,
    max_k: int,
) -> None:
    """Print coefficient table and growth rates for bootstrap results."""
    # Build sorted list of types by edge count
    sorted_canons = sorted(all_types.keys(), key=lambda c: all_types[c][3])

    # Collect all grades where at least one type has a nonzero coefficient
    all_grades: set[int] = set()
    for canon in sorted_canons:
        for k, c in enumerate(coeffs.get(canon, [])):
            if c is not None and c != 0 and k > 0:
                all_grades.add(k)
    grades = sorted(all_grades)
    if not grades:
        print("\nNo nonzero coefficients found.")
        return

    # Coefficient table
    print(f"\n{'=' * 70}")
    print(f"  Coefficient table: coeff_k(tau)")
    print(f"{'=' * 70}")

    hdr = f"  {'Type':>20s} {'|e|':>4s}"
    for k in grades:
        hdr += f" {'k=' + str(k):>12s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for canon in sorted_canons:
        _, edges, nv, ne = all_types[canon]
        name = describe_graph(edges)
        cvec = coeffs.get(canon, [])
        row = f"  {name:>20s} {ne:>4d}"
        for k in grades:
            c = cvec[k] if k < len(cvec) else None
            if c is not None and c != 0:
                row += f" {c:>12d}"
            else:
                row += f" {'':>12s}"
        print(row)

    # Growth rates
    print(f"\n{'=' * 70}")
    print(f"  Growth rates: coeff_k / coeff_{{k-1}}")
    print(f"{'=' * 70}")

    hdr = f"  {'Type':>20s}"
    for k in grades[1:]:
        hdr += f" {'k=' + str(k):>12s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for canon in sorted_canons:
        _, edges, nv, ne = all_types[canon]
        name = describe_graph(edges)
        cvec = coeffs.get(canon, [])
        row = f"  {name:>20s}"
        for k in grades[1:]:
            c = cvec[k] if k < len(cvec) else None
            cp = cvec[k - 1] if (k - 1) < len(cvec) else None
            if c is not None and cp is not None and cp > 0:
                row += f" {c / cp:>12.2f}"
            else:
                row += f" {'':>12s}"
        print(row)

    # Log2 analysis
    print(f"\n{'=' * 70}")
    print(f"  Log2 analysis")
    print(f"{'=' * 70}")

    hdr = f"  {'Type':>20s}"
    for k in grades:
        hdr += f" {'k=' + str(k):>12s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for canon in sorted_canons:
        _, edges, nv, ne = all_types[canon]
        name = describe_graph(edges)
        cvec = coeffs.get(canon, [])
        row = f"  {name:>20s}"
        for k in grades:
            c = cvec[k] if k < len(cvec) else None
            if c is not None and c > 0:
                row += f" {math.log2(c):>12.2f}"
            else:
                row += f" {'':>12s}"
        print(row)


# ===================================================================
#  Subcommand: matrix  (hardcoded n=6 bitmask-based analysis)
# ===================================================================

def _edges_of_kn(n: int) -> list[tuple[int, int]]:
    """All edges of K_n as (i, j) with i < j."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _edge_index(i: int, j: int, n: int) -> int:
    """Map edge (i,j) with i<j to its index in K_n."""
    if i > j:
        i, j = j, i
    return i * n - i * (i + 1) // 2 + (j - i - 1)


def _base_to_edge_list(
    base_mask: int, edge_list: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Convert bitmask to list of edges."""
    return [edge_list[i] for i in range(len(edge_list)) if base_mask & (1 << i)]


def _classify_tree_from_mask(
    base_mask: int, edge_list: list[tuple[int, int]], n_vertices: int
) -> int:
    """Classify a tree type from its base edge bitmask.

    Returns type index 0-5 (for the 6 trees on 6 vertices) or -1.
    """
    edges = _base_to_edge_list(base_mask, edge_list)
    ne = len(edges)
    if ne != 5:
        return -1

    verts: set[int] = set()
    for u, v in edges:
        verts.add(u)
        verts.add(v)
    if len(verts) != 6:
        return -1

    adj: dict[int, set[int]] = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    start = next(iter(verts))
    visited = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for u in adj[node]:
            if u not in visited:
                visited.add(u)
                stack.append(u)
    if len(visited) != len(verts):
        return -1

    deg_seq = tuple(sorted(len(adj[v]) for v in verts))

    if deg_seq == (1, 1, 2, 2, 2, 2):
        return 0  # P_6
    if deg_seq == (1, 1, 1, 1, 1, 5):
        return 5  # K_{1,5}
    if deg_seq == (1, 1, 1, 1, 3, 3):
        return 4  # Double star
    if deg_seq == (1, 1, 1, 1, 2, 4):
        return 3  # Spider
    if deg_seq == (1, 1, 1, 2, 2, 3):
        v3 = [v for v in verts if len(adj[v]) == 3][0]
        nbr_degs = sorted(len(adj[u]) for u in adj[v3])
        if nbr_degs == [1, 2, 2]:
            return 1  # Caterpillar A
        if nbr_degs == [1, 1, 2]:
            return 2  # Caterpillar B
    return -1


def _count_labeled_copies_n6(n: int) -> list[int]:
    """Count sub(tau, K_6) for each of the 6 tree types on 6 vertices."""
    edge_list = _edges_of_kn(n)
    ne = len(edge_list)
    counts = [0] * 6
    for combo in combinations(range(ne), 5):
        mask = 0
        for i in combo:
            mask |= 1 << i
        t = _classify_tree_from_mask(mask, edge_list, n)
        if t >= 0:
            counts[t] += 1
    return counts


def _build_line_graph_level(
    n_prev: int,
    adj_prev: list[set[int]],
    base_prev: list[int],
) -> tuple[int, list[set[int]], list[int]]:
    """Build next line graph level using bitmask base tracking."""
    new_edges: list[tuple[int, int]] = []
    for v in range(n_prev):
        for u in adj_prev[v]:
            if u > v:
                new_edges.append((v, u))

    n_new = len(new_edges)
    base_new = [base_prev[v] | base_prev[u] for v, u in new_edges]

    incident: list[list[int]] = [[] for _ in range(n_prev)]
    for idx, (v, u) in enumerate(new_edges):
        incident[v].append(idx)
        incident[u].append(idx)

    adj_new: list[set[int]] = [set() for _ in range(n_new)]
    for v_prev in range(n_prev):
        inc = incident[v_prev]
        for i in range(len(inc)):
            for j in range(i + 1, len(inc)):
                a, b = inc[i], inc[j]
                adj_new[a].add(b)
                adj_new[b].add(a)

    return n_new, adj_new, base_new


def cmd_matrix(args: argparse.Namespace) -> None:
    """Hardcoded coefficient matrix analysis for trees on 6 vertices in K_6."""
    import numpy as np

    n = 6
    edge_list = _edges_of_kn(n)
    ne = len(edge_list)  # 15
    tree_names_list = ["P_6", "Cat_A", "Cat_B", "Spider", "DblStar", "K_{1,5}"]
    n_types = 6

    print(f"=== Coefficient matrix for trees on {n} vertices in K_{n} ===\n")

    print("Counting labeled copies of each tree type in K_6...")
    t0 = time.time()
    sub_counts = _count_labeled_copies_n6(n)
    print(
        f"  sub(tau, K_6): {list(zip(tree_names_list, sub_counts))} "
        f"({time.time() - t0:.1f}s)"
    )
    print(f"  Total tree edge-subsets: {sum(sub_counts)}")

    # Build L^1(K_6)
    print(f"\nBuilding iterated line graphs...")

    adj1: list[set[int]] = [set() for _ in range(ne)]
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

    max_grade = 10
    coeffs = np.zeros((n_types, max_grade - 4), dtype=np.int64)

    for grade in range(2, max_grade + 1):
        t0 = time.time()
        cur_n, cur_adj, cur_base = _build_line_graph_level(cur_n, cur_adj, cur_base)
        elapsed = time.time() - t0
        print(f"  L^{grade}(K_{n}): {cur_n} vertices ({elapsed:.1f}s)", flush=True)

        if grade >= 5:
            type_counts = [0] * n_types
            for v in range(cur_n):
                t = _classify_tree_from_mask(cur_base[v], edge_list, n)
                if t >= 0:
                    type_counts[t] += 1

            col = grade - 5
            for t in range(n_types):
                if sub_counts[t] > 0:
                    assert type_counts[t] % sub_counts[t] == 0, (
                        f"Non-integer coefficient at grade {grade}, "
                        f"type {tree_names_list[t]}: {type_counts[t]}/{sub_counts[t]}"
                    )
                    coeffs[t][col] = type_counts[t] // sub_counts[t]

            print(f"    Fiber counts: {list(zip(tree_names_list, type_counts))}")
            print(
                f"    Coefficients: "
                f"{list(zip(tree_names_list, coeffs[:, col].tolist()))}"
            )

        mem_mb = sys.getsizeof(cur_adj) / 1e6
        print(f"    (approx adj mem: {mem_mb:.0f} MB)", flush=True)

        if cur_n > 5_000_000:
            print(f"    Too large, stopping at grade {grade}")
            break

    # Print coefficient matrix
    n_cols = coeffs.shape[1]
    print(
        f"\n=== Coefficient Matrix "
        f"(rows=types, cols=grades 5..{4 + n_cols}) ==="
    )
    header = "".ljust(12) + "".join(
        f"k={k}".rjust(12) for k in range(5, 5 + n_cols)
    )
    print(header)
    for t in range(n_types):
        row = tree_names_list[t].ljust(12) + "".join(
            f"{coeffs[t][c]}".rjust(12) for c in range(n_cols)
        )
        print(row)

    # Rank analysis
    M = coeffs.T  # shape (n_grades, n_types)
    print(f"\n=== Rank analysis ===")
    print(f"Matrix shape: {M.shape} (grades x types)")

    Mf = M.astype(np.float64)
    rank = np.linalg.matrix_rank(Mf)
    print(f"Rank: {rank} (out of {n_types} types)")

    if rank == n_types:
        print("FULL RANK -- coefficient vectors are linearly independent!")
    else:
        print(f"RANK DEFICIENT -- rank {rank} < {n_types}")

    sv = np.linalg.svd(Mf, compute_uv=False)
    print(f"Singular values: {sv}")
    if sv[-1] > 0:
        print(f"Condition number: {sv[0] / sv[-1]:.2e}")
    else:
        print("Infinite condition number")

    print(f"\nCumulative rank by grade:")
    for k in range(1, n_cols + 1):
        r = np.linalg.matrix_rank(M[:k, :].astype(np.float64))
        print(f"  Grades 5..{4 + k}: rank {r}")


# ===================================================================
#  CLI entry point
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fiber coefficient computation and analysis for iterated line graphs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- kn --
    p_kn = subparsers.add_parser(
        "kn",
        help="Compute fiber coefficients from L^k(K_n)",
    )
    p_kn.add_argument("--n", type=int, default=5, help="Complete graph K_n (default: 5)")
    p_kn.add_argument("--max-k", type=int, default=7, help="Maximum grade (default: 7)")

    # -- bootstrap --
    p_boot = subparsers.add_parser(
        "bootstrap",
        help="Compute coefficients via Mobius inversion bootstrapping",
    )
    p_boot.add_argument(
        "--max-edges", type=int, default=5,
        help="Maximum tree edge count to enumerate (default: 5)",
    )
    p_boot.add_argument("--max-k", type=int, default=10, help="Maximum grade (default: 10)")
    p_boot.add_argument(
        "--max-line-edges", type=int, default=5_000_000,
        help="Edge cap for line graph iteration (default: 5000000)",
    )

    # -- matrix --
    subparsers.add_parser(
        "matrix",
        help="Hardcoded coefficient matrix analysis for n=6 trees",
    )

    args = parser.parse_args()

    if args.command is None or args.command == "kn":
        if args.command is None:
            if not hasattr(args, "n"):
                args.n = 5
            if not hasattr(args, "max_k"):
                args.max_k = 7
        cmd_kn(args)
    elif args.command == "bootstrap":
        cmd_bootstrap(args)
    elif args.command == "matrix":
        cmd_matrix(args)


if __name__ == "__main__":
    main()
