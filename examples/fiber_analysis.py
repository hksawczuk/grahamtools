"""
Fiber analysis of iterated line graphs L^k(K_n).

Subcommands:
  degrees   -- Compute pre-quotient fiber graph for each tree type in L^k(K_n)
               and analyze degree distributions to assess regularity.
  iterate   -- Iterate fiber coefficients forward using orbit-level line graph
               structure: classify up to a given grade, then build transition
               tables for higher grades.

Usage:
  python3 fiber_analysis.py degrees [--n 5] [--max-k 6]
  python3 fiber_analysis.py iterate [--n 5] [--classify-k 7] [--max-k 12]
"""
import argparse
import math
import time
from collections import defaultdict, Counter

from grahamtools.kn.levels import generate_levels_Kn_ids
from grahamtools.kn.expand import expand_to_simple_base_edges_id
from grahamtools.kn.classify import canon_key, iso_classes_with_stats
from grahamtools.utils.automorphisms import aut_size_edges
from grahamtools.utils.naming import tree_name


# ---------------------------------------------------------------------------
# Subcommand: degrees
# ---------------------------------------------------------------------------

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

        # -- Group vertices by canonical tree type --
        vertex_to_key = {}
        key_to_vertices = defaultdict(list)

        for v in Vk:
            edges = expand_to_simple_base_edges_id(v, k, ep)
            key = canon_key(edges, n)
            vertex_to_key[v] = key
            key_to_vertices[key].append(v)

        # Get iso class stats for naming
        iso_stats = iso_classes_with_stats(Vk, n, k, ep)
        key_to_info = {s['key']: s for s in iso_stats}

        # -- Build adjacency at grade k --
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

        # -- For each fiber, compute intra-fiber degree distribution --
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

        # -- Summary: total edges in each fiber graph --
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


# ---------------------------------------------------------------------------
# Subcommand: iterate
# ---------------------------------------------------------------------------

def classify_grade(Vk, n, k, ep):
    """Classify all grade-k vertices into orbits with fiber type info."""
    # Group vertices by canonical key
    buckets = {}
    vertex_to_key = {}

    for v in Vk:
        edges = expand_to_simple_base_edges_id(v, k, ep)
        key = canon_key(edges, n)
        vertex_to_key[v] = key
        if key not in buckets:
            buckets[key] = {"edges": edges, "vertices": []}
        buckets[key]["vertices"].append(v)

    # Compute orbit info
    orbit_info = {}  # key -> {name, aut, orbit_sz, edges, vertices}
    for key, bucket in buckets.items():
        edges = bucket["edges"]
        aut = aut_size_edges(edges, n)
        orbit_sz = math.factorial(n) // aut
        name = tree_name(edges, n)
        n_orbits = len(bucket["vertices"]) // orbit_sz
        orbit_info[key] = {
            "name": name,
            "aut": aut,
            "orbit_sz": orbit_sz,
            "edges": edges,
            "n_vertices": len(bucket["vertices"]),
            "n_orbits": n_orbits,
            "nedges": len(edges),
        }

    return vertex_to_key, orbit_info


def build_orbit_adjacency(Vk, ep_k, vertex_to_key, orbit_info, n):
    """
    Build orbit-level adjacency at grade k.

    For each pair of adjacent vertices (u, v) at grade k, record
    the pair (key_u, key_v) and determine the fiber type of the
    grade-(k+1) vertex they produce.

    Returns:
      edge_counts: dict mapping (key_u, key_v) -> count
        where count = number of labeled edges between orbits of type key_u and key_v
    """
    # Build adjacency at grade k
    inc = defaultdict(list)
    for v in Vk:
        a, b = ep_k[v]
        inc[a].append(v)
        inc[b].append(v)

    # For each adjacent pair, determine parent keys and child type
    # But we can't easily determine child type without going to grade k+1
    # Instead, just count edges between orbit types

    edge_counts = defaultdict(int)  # (key_u, key_v) -> count (key_u <= key_v)

    seen_edges = set()
    for parent, children in inc.items():
        for i in range(len(children)):
            for j in range(i + 1, len(children)):
                u, v = children[i], children[j]
                if u > v:
                    u, v = v, u
                if (u, v) not in seen_edges:
                    seen_edges.add((u, v))
                    ku = vertex_to_key[u]
                    kv = vertex_to_key[v]
                    pair = (min(ku, kv), max(ku, kv))
                    edge_counts[pair] += 1

    return edge_counts


def iterate_from_classified(n, classify_k, max_k):
    """Main computation for the iterate subcommand."""

    print(f"Generating L^k(K_{n}) for k=0..{classify_k} (trees only)...")
    t0 = time.time()
    V_by_level, ep = generate_levels_Kn_ids(n, classify_k + 1, prune_cycles=True)
    t1 = time.time()
    print(f"  Done in {t1-t0:.1f}s")

    for k in range(classify_k + 2):
        if k in V_by_level:
            print(f"  Grade {k}: {len(V_by_level[k])} tree vertices")

    # -- Classify at each grade up to classify_k+1 --
    # We classify classify_k and classify_k+1 to verify our iteration

    fiber_coeffs = defaultdict(dict)  # key -> {k: coeff}
    fiber_names = {}  # key -> name
    fiber_info = {}  # key -> orbit info

    for k in range(1, min(classify_k + 2, len(V_by_level))):
        Vk = V_by_level.get(k, [])
        if not Vk:
            continue

        t0 = time.time()
        vtk, oi = classify_grade(Vk, n, k, ep)
        t1 = time.time()

        for key, info in oi.items():
            coeff = info["n_vertices"] // info["orbit_sz"]
            fiber_coeffs[key][k] = coeff
            fiber_names[key] = info["name"]
            fiber_info[key] = info

        if t1 - t0 > 0.5:
            print(f"    Grade {k}: classified into {len(oi)} fibers in {t1-t0:.1f}s")

    # -- Build orbit-level structure at classify_k --
    k0 = classify_k
    Vk0 = V_by_level[k0]

    print(f"\n  Building orbit adjacency at grade {k0}...")
    t0 = time.time()
    vtk0, oi0 = classify_grade(Vk0, n, k0, ep)
    edge_counts = build_orbit_adjacency(Vk0, ep[k0], vtk0, oi0, n)
    t1 = time.time()
    print(f"  Done in {t1-t0:.1f}s")

    # Print edge structure
    total_edges = sum(edge_counts.values())
    print(f"  Total edges at grade {k0}: {total_edges}")
    print(f"  Edge types:")
    for (k1, k2), count in sorted(edge_counts.items(), key=lambda x: -x[1]):
        n1 = fiber_names.get(k1, f"key={k1}")
        n2 = fiber_names.get(k2, f"key={k2}")
        print(f"    {n1} -- {n2}: {count} edges")

    # -- Now iterate forward --
    # At grade k0+1, each vertex = edge at grade k0
    # For fiber counting, we need: for each edge (u,v) at grade k0,
    # what is the fiber type of the grade-(k0+1) vertex?
    #
    # The fiber type depends on the UNION of base edges of u and v.
    # Two vertices in the same orbit have the same base-edge type,
    # but different labeled base edges. The union depends on the
    # specific labels, not just the types.
    #
    # However, for counting how many edges between fiber A and fiber B
    # produce children in fiber C, we can compute this from the
    # classified grade k0+1 data (if available).

    # Check if we have grade k0+1 classified
    k1 = k0 + 1
    if k1 in V_by_level and V_by_level[k1]:
        Vk1 = V_by_level[k1]
        print(f"\n  Verifying with classified grade {k1}...")
        vtk1, oi1 = classify_grade(Vk1, n, k1, ep)

        # For each grade-k1 vertex, its parents are ep[k1][v] = (a, b)
        # a, b are grade-k0 vertices
        # child fiber = vtk1[v], parent fibers = vtk0[a], vtk0[b]

        # Build transition table: (parent_key_a, parent_key_b) -> {child_key: count}
        transition = defaultdict(lambda: defaultdict(int))
        for v in Vk1:
            a, b = ep[k1][v]
            ka = vtk0[a]
            kb = vtk0[b]
            kc = vtk1[v]
            pair = (min(ka, kb), max(ka, kb))
            transition[pair][kc] += 1

        print(f"\n  Transition table (parent pair -> child fiber):")
        for pair in sorted(transition, key=lambda p: -sum(transition[p].values())):
            na = fiber_names.get(pair[0], f"key={pair[0]}")
            nb = fiber_names.get(pair[1], f"key={pair[1]}")
            total = sum(transition[pair].values())
            print(f"    {na} x {nb} ({total} edges):")
            for ck, cnt in sorted(transition[pair].items(), key=lambda x: -x[1]):
                nc = fiber_names.get(ck, f"key={ck}")
                oi_c = fiber_info.get(ck, oi1.get(ck, {}))
                orb = oi_c.get("orbit_sz", "?")
                print(f"      -> {nc} (orbit={orb}): {cnt} labeled, "
                      f"{cnt // orb if isinstance(orb, int) else '?'} orbits")

        # -- Can we use this transition table to iterate further? --
        # The transition table tells us: for each pair of parent fiber types,
        # how many labeled children of each child type are produced.
        #
        # But the counts depend on the SPECIFIC edge structure between fibers,
        # not just the fiber sizes. So we need to track edge counts between
        # fibers at each grade.
        #
        # State at grade k: for each pair (fiber_A, fiber_B), the number of
        # edges between them. Then:
        #   children_in_C_from_(A,B) = transition_rate(A,B->C) * edges(A,B)
        #
        # But the transition rates may NOT be constant -- they depend on the
        # internal structure of the fibers, not just their sizes.
        #
        # Let's check: is the transition rate = children / edges constant?

        print(f"\n  Transition RATES (children per edge):")
        for pair in sorted(transition, key=lambda p: -sum(transition[p].values())):
            na = fiber_names.get(pair[0], f"key={pair[0]}")
            nb = fiber_names.get(pair[1], f"key={pair[1]}")
            n_edges = edge_counts.get(pair, 0)
            if n_edges == 0:
                continue
            total_children = sum(transition[pair].values())
            print(f"    {na} x {nb}: {n_edges} edges -> {total_children} children "
                  f"(rate={total_children/n_edges:.2f})")
            for ck, cnt in sorted(transition[pair].items(), key=lambda x: -x[1]):
                nc = fiber_names.get(ck, f"key={ck}")
                print(f"      -> {nc}: rate = {cnt/n_edges:.4f}")

    # -- Print coefficient summary --
    print(f"\n{'='*80}")
    print(f"  Coefficient summary")
    print(f"{'='*80}")

    all_grades = set()
    for data in fiber_coeffs.values():
        all_grades.update(data.keys())
    grades = sorted(all_grades)

    sorted_keys = sorted(fiber_coeffs.keys(),
                         key=lambda k: (fiber_info.get(k, {}).get("nedges", 0),
                                        -max(fiber_coeffs[k].values())))

    hdr = f"  {'Tree':>16s} {'|e|':>4s}"
    for k in grades:
        hdr += f" {'k='+str(k):>10s}"
    print(hdr)
    print("  " + "-" * (len(hdr)))

    for key in sorted_keys:
        name = fiber_names.get(key, f"key={key}")
        ne = fiber_info.get(key, {}).get("nedges", "?")
        row = f"  {name:>16s} {str(ne):>4s}"
        for k in grades:
            c = fiber_coeffs[key].get(k, None)
            if c is not None:
                row += f" {c:>10d}"
            else:
                row += f" {'':>10s}"
        print(row)

    # Log analysis
    print(f"\n  Log2 analysis:")
    print(f"  {'Tree':>16s}", end="")
    for k in grades:
        print(f" {'k='+str(k):>8s}", end="")
    print()

    for key in sorted_keys:
        data = fiber_coeffs[key]
        ks = sorted(data.keys())
        if len(ks) < 2 or all(data[k] == data[ks[0]] for k in ks):
            continue  # skip constant fibers
        logs = {k: math.log2(data[k]) for k in ks}
        deltas = {ks[i]: logs[ks[i]] - logs[ks[i-1]] for i in range(1, len(ks))}
        ddeltas = {ks[i]: deltas[ks[i]] - deltas[ks[i-1]]
                   for i in range(2, len(ks)) if ks[i] in deltas and ks[i-1] in deltas}

        name = fiber_names.get(key, f"key={key}")
        print(f"\n  {name:>16s}")
        print(f"    {'k':>4s} {'coeff':>10s} {'log2':>8s} {'D':>8s} {'DD':>8s}")
        for k in ks:
            c = data[k]
            lg = logs[k]
            d = f"{deltas[k]:.3f}" if k in deltas else "-"
            dd = f"{ddeltas[k]:.3f}" if k in ddeltas else "-"
            print(f"    {k:>4d} {c:>10d} {lg:>8.3f} {d:>8s} {dd:>8s}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fiber analysis of iterated line graphs L^k(K_n).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- degrees subcommand --
    p_deg = subparsers.add_parser(
        "degrees",
        help="Compute fiber graph degree distributions for each tree type.",
    )
    p_deg.add_argument("--n", type=int, default=5,
                       help="Number of vertices in K_n (default: 5)")
    p_deg.add_argument("--max-k", type=int, default=6,
                       help="Maximum grade to analyze (default: 6)")

    # -- iterate subcommand --
    p_iter = subparsers.add_parser(
        "iterate",
        help="Iterate fiber coefficients using orbit-level line graph structure.",
    )
    p_iter.add_argument("--n", type=int, default=5,
                        help="Number of vertices in K_n (default: 5)")
    p_iter.add_argument("--classify-k", type=int, default=6,
                        help="Grade to fully classify (default: 6)")
    p_iter.add_argument("--max-k", type=int, default=10,
                        help="Target grade (default: 10)")

    args = parser.parse_args()

    if args.command == "degrees":
        fiber_degree_analysis(args.n, args.max_k)
    elif args.command == "iterate":
        iterate_from_classified(args.n, args.classify_k, args.max_k)


if __name__ == "__main__":
    main()
