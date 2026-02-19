#!/usr/bin/env python3
"""Consolidated fiber analysis for iterated line graphs of K_n.

Subcommands
-----------
fork-fiber   Compute the fiber graph of the "fork" tree in L^4(K_5).
k16-degree   Compute d_0(K_{1,6}) via parent analysis in L^5(K_6).

Usage
-----
    python kn_fiber_compute.py fork-fiber
    python kn_fiber_compute.py k16-degree
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from grahamtools.kn.levels import generate_levels_Kn_ids, Endpoints
from grahamtools.kn.expand import expand_to_simple_base_edges_id
from grahamtools.utils.connectivity import is_connected_edges, connected_components_edges
from grahamtools.utils.naming import tree_name

# ---------------------------------------------------------------------------
# Shared helpers that wrap package primitives
# ---------------------------------------------------------------------------

def _build_adjacency(
    V_by_level: Dict[int, List[int]],
    endpoints_by_level: Dict[int, List[Endpoints]],
    level: int,
) -> Dict[int, set[int]]:
    """Build an adjacency dict for vertices at *level*.

    Two vertices at level k are adjacent iff they share an endpoint at
    level k-1 (i.e. the corresponding edges in L^{k-1}(K_n) are incident).
    """
    adj: Dict[int, set[int]] = defaultdict(set)
    if level < 1:
        return adj

    eps = endpoints_by_level[level]
    # Build incidence: for each level-(k-1) vertex, which level-k vertices
    # have it as an endpoint?
    incidence: Dict[int, List[int]] = defaultdict(list)
    for v_id, (a, b) in enumerate(eps):
        incidence[a].append(v_id)
        incidence[b].append(v_id)

    for inc_list in incidence.values():
        for i in range(len(inc_list)):
            for j in range(i + 1, len(inc_list)):
                u, v = inc_list[i], inc_list[j]
                adj[u].add(v)
                adj[v].add(u)

    return adj


def _base_edges_for_level(
    V_by_level: Dict[int, List[int]],
    endpoints_by_level: Dict[int, List[Endpoints]],
    level: int,
) -> List[List[Tuple[int, int]]]:
    """Return the base-edge set for every vertex at *level*.

    Returns a list indexed by vertex ID, each entry a sorted list of (i, j)
    base edges.
    """
    Vk = V_by_level[level]
    return [
        expand_to_simple_base_edges_id(v, level, endpoints_by_level)
        for v in Vk
    ]


def _base_edges_frozensets(
    V_by_level: Dict[int, List[int]],
    endpoints_by_level: Dict[int, List[Endpoints]],
    level: int,
) -> List[frozenset[Tuple[int, int]]]:
    """Return frozenset base-edge sets for every vertex at *level*."""
    Vk = V_by_level[level]
    return [
        frozenset(expand_to_simple_base_edges_id(v, level, endpoints_by_level))
        for v in Vk
    ]


def _classify_base_edges(
    base_edges: list[tuple[int, int]],
    n: int,
) -> str | None:
    """Classify a base-edge set as a named tree, or return None if not a tree.

    Uses ``tree_name`` from the grahamtools package for naming and
    ``is_connected_edges`` for connectivity checking.
    """
    if not base_edges:
        return None

    verts: set[int] = set()
    for u, v in base_edges:
        verts.add(u)
        verts.add(v)

    ne = len(base_edges)
    nv = len(verts)

    # Tree check: connected and |E| == |V| - 1
    if ne != nv - 1:
        return None
    if not is_connected_edges(base_edges, verts):
        return None

    return tree_name(base_edges, n)


def _star_info(
    base_edges: list[tuple[int, int]],
) -> tuple[int, int] | None:
    """If *base_edges* form a star K_{1,r}, return (r, center). Else None."""
    if not base_edges:
        return None

    deg: dict[int, int] = defaultdict(int)
    for u, v in base_edges:
        deg[u] += 1
        deg[v] += 1

    ne = len(base_edges)
    for vertex, d in deg.items():
        if d == ne and all(dd == 1 for vv, dd in deg.items() if vv != vertex):
            return (ne, vertex)
    return None


# ---------------------------------------------------------------------------
# Subcommand: fork-fiber
# ---------------------------------------------------------------------------

def cmd_fork_fiber(_args: argparse.Namespace) -> None:
    """Compute the fiber graph of the 'fork' tree in L^4(K_5)."""

    n, max_k = 5, 4
    print("=== Fork fiber graph in L^4(K_5) ===\n")

    t0 = time.time()
    V, eps = generate_levels_Kn_ids(n, max_k)
    for lvl in range(1, max_k + 1):
        print(f"  L^{lvl}(K_{n}): {len(V[lvl])}", flush=True)
    print(f"  (built in {time.time() - t0:.1f}s)")

    # Build adjacency and base-edge maps at level 4
    a4 = _build_adjacency(V, eps, 4)
    b4 = _base_edges_frozensets(V, eps, 4)
    b4_lists = _base_edges_for_level(V, eps, 4)
    v4 = V[4]

    print(f"\nGrade 4: {len(v4)} vertices")

    # Classify all grade-4 elements
    type_counts: dict[str, int] = defaultdict(int)
    fork_elements: list[int] = []

    for v in v4:
        typ = _classify_base_edges(b4_lists[v], n)
        if typ:
            type_counts[typ] += 1
        if typ == "fork":
            fork_elements.append(v)

    print("\nType distribution at grade 4:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    print(f"\nFork fiber: {len(fork_elements)} elements")
    print(
        f"coeff_4(fork) = {len(fork_elements)} "
        "(in K_5, divide by nothing since we count all)"
    )

    # Group by labeled support
    by_support: dict[frozenset, list[int]] = defaultdict(list)
    for v in fork_elements:
        by_support[b4[v]].append(v)

    print(f"Distinct fork supports: {len(by_support)}")
    for sup, vs in sorted(by_support.items(), key=lambda x: sorted(x[0])):
        edges_labeled = sorted(sup)
        print(f"  {edges_labeled}: {len(vs)} elements")

    # Build fiber graph
    fork_set = set(fork_elements)
    fork_adj: dict[int, set[int]] = defaultdict(set)
    for v in fork_elements:
        for u in a4[v]:
            if u in fork_set:
                fork_adj[v].add(u)

    degrees = [len(fork_adj[v]) for v in fork_elements]
    deg_dist: dict[int, int] = defaultdict(int)
    for d in degrees:
        deg_dist[d] += 1

    n_edges = sum(degrees) // 2

    print("\nFork fiber graph:")
    print(f"  Vertices: {len(fork_elements)}")
    print(f"  Edges: {n_edges}")
    print(f"  Degree distribution: {dict(sorted(deg_dist.items()))}")

    # Check if regular
    if len(deg_dist) == 1:
        print(f"  Regular: {list(deg_dist.keys())[0]}")
    else:
        print("  Not regular!")

    # Check decomposition by support
    cross_support = 0
    intra_support = 0
    for v in fork_elements:
        for u in fork_adj[v]:
            if b4[v] == b4[u]:
                intra_support += 1
            else:
                cross_support += 1

    print(f"\n  Intra-support edges: {intra_support // 2}")
    print(f"  Cross-support edges: {cross_support // 2}")

    # Analyze cross-support edges: what's the union type?
    cross_union_types: dict[str, int] = defaultdict(int)
    for v in fork_elements:
        for u in fork_adj[v]:
            if u > v and b4[v] != b4[u]:
                union_base = list(b4[v] | b4[u])
                typ = _classify_base_edges(union_base, n)
                if typ:
                    cross_union_types[typ] += 1
                else:
                    # Not a tree -- describe by vertex/edge counts
                    vs_u: set[int] = set()
                    for a, b in union_base:
                        vs_u.add(a)
                        vs_u.add(b)
                    cross_union_types[
                        f"non-tree_{len(vs_u)}v_{len(union_base)}e"
                    ] += 1

    print("\n  Cross-support union types:")
    for t, c in sorted(cross_union_types.items()):
        print(f"    {t}: {c}")

    # Check connectivity of the fiber graph using package utility
    fiber_edge_list: list[tuple[int, int]] = []
    for v in fork_elements:
        for u in fork_adj[v]:
            if u > v:
                fiber_edge_list.append((v, u))

    if fiber_edge_list:
        components_raw = connected_components_edges(fiber_edge_list)
        # Include isolated vertices not covered by edges
        covered = set()
        for vs_set, _ in components_raw:
            covered |= vs_set
        components: list[set[int]] = [vs_set for vs_set, _ in components_raw]
        for v in fork_elements:
            if v not in covered:
                components.append({v})
    else:
        components = [{v} for v in fork_elements]

    print(f"\n  Connected components: {len(components)}")
    comp_sizes = sorted(len(c) for c in components)
    print(f"  Component sizes: {comp_sizes}")

    # Properties per component
    for ci, comp in enumerate(components):
        comp_list = sorted(comp)
        comp_degs = [len(fork_adj[v] & comp) for v in comp_list]
        comp_deg_dist: dict[int, int] = defaultdict(int)
        for d in comp_degs:
            comp_deg_dist[d] += 1
        comp_edges = sum(comp_degs) // 2

        supports_in_comp = {b4[v] for v in comp_list}

        print(f"\n  Component {ci}: {len(comp)} vertices, {comp_edges} edges")
        print(f"    Degrees: {dict(sorted(comp_deg_dist.items()))}")
        print(f"    Supports: {len(supports_in_comp)}")

    # Girth, diameter, triangles for full fiber graph
    print("\nGraph properties:")

    # Bipartite check
    color: dict[int, int] = {}
    is_bip = True
    for start in fork_elements:
        if start in color:
            continue
        stack = [(start, 0)]
        while stack:
            v, c = stack.pop()
            if v in color:
                if color[v] != c:
                    is_bip = False
                continue
            color[v] = c
            for u in fork_adj[v]:
                if u not in color:
                    stack.append((u, 1 - c))
    print(f"  Bipartite: {is_bip}")

    # Girth (sampled)
    min_girth = float("inf")
    for sv in fork_elements[:50]:
        dist = {sv: 0}
        parent = {sv: -1}
        queue = [sv]
        qi = 0
        while qi < len(queue):
            vi = queue[qi]
            qi += 1
            for u in fork_adj[vi]:
                if u not in dist:
                    dist[u] = dist[vi] + 1
                    parent[u] = vi
                    queue.append(u)
                elif parent[vi] != u and parent[u] != vi:
                    min_girth = min(min_girth, dist[vi] + dist[u] + 1)
    print(f"  Girth: {min_girth}")

    # Diameter (sampled)
    max_dist = 0
    for sv in fork_elements[:50]:
        dist = {sv: 0}
        queue = [sv]
        qi = 0
        while qi < len(queue):
            vi = queue[qi]
            qi += 1
            for u in fork_adj[vi]:
                if u not in dist:
                    dist[u] = dist[vi] + 1
                    queue.append(u)
        max_dist = max(max_dist, max(dist.values()))
    print(f"  Diameter (sampled): {max_dist}")

    # Triangle count
    n_tri = 0
    fork_list = sorted(fork_elements)
    for v in fork_list:
        for u in fork_adj[v]:
            if u > v:
                for w in fork_adj[u]:
                    if w > u and w in fork_adj[v]:
                        n_tri += 1
    print(f"  Triangles: {n_tri}")


# ---------------------------------------------------------------------------
# Subcommand: k16-degree
# ---------------------------------------------------------------------------

def cmd_k16_degree(_args: argparse.Namespace) -> None:
    """Compute d_0(K_{1,6}) via parent analysis in L^5(K_6)."""

    n, max_k = 6, 5
    print("=== Computing d_0(K_{1,6}) via parent analysis ===\n")

    t0 = time.time()
    V, eps = generate_levels_Kn_ids(n, max_k)
    for lvl in range(1, max_k + 1):
        print(f"  L^{lvl}(K_{n}): {len(V[lvl])}", flush=True)
    print(f"  (built in {time.time() - t0:.1f}s)")

    # Build base-edge lists at levels 4 and 5
    b5_lists = _base_edges_for_level(V, eps, 5)
    b4_lists = _base_edges_for_level(V, eps, 4)
    v5 = V[5]
    v4 = V[4]

    # Parents at level 5: each level-5 vertex has two endpoint IDs at level 4
    parents5 = eps[5]  # parents5[v] = (p1, p2)

    # K_{1,5} fiber, center=0
    all_leaves = frozenset({1, 2, 3, 4, 5})
    fib5: list[int] = []
    for v in v5:
        info = _star_info(b5_lists[v])
        if info and info[0] == 5 and info[1] == 0:
            fib5.append(v)
    print(f"\nK_{{1,5}} fiber (center=0): {len(fib5)} elements")

    # K_{1,4} fiber, center=0, indexed by leaf set
    fib4_by_leaves: dict[frozenset[int], list[int]] = defaultdict(list)
    for v in v4:
        info = _star_info(b4_lists[v])
        if info and info[0] == 4 and info[1] == 0:
            ls: set[int] = set()
            for u_e, v_e in b4_lists[v]:
                ls.add(u_e)
                ls.add(v_e)
            ls.discard(0)
            fib4_by_leaves[frozenset(ls)].append(v)

    print(f"K_{{1,4}} leaf sets (center=0): {len(fib4_by_leaves)}")
    for ls in sorted(fib4_by_leaves, key=sorted):
        print(f"  leaves {sorted(ls)}: {len(fib4_by_leaves[ls])} elements")

    # For each K_{1,5} element, find its K_{1,4} parents
    fib4_set: set[int] = set()
    for vs in fib4_by_leaves.values():
        fib4_set.update(vs)

    # Build reverse lookup: fib4 vertex -> leaf set
    fib4_to_leaves: dict[int, frozenset[int]] = {}
    for ls, vs in fib4_by_leaves.items():
        for v in vs:
            fib4_to_leaves[v] = ls

    parent_map: dict[int, dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    element_cross_count: dict[int, int] = defaultdict(int)

    for v in fib5:
        p1, p2 = parents5[v]
        for p in [p1, p2]:
            if p in fib4_set:
                leaves_p = fib4_to_leaves.get(p)
                if leaves_p is not None:
                    missing = all_leaves - leaves_p
                    ml = next(iter(missing))
                    parent_map[ml][p].append(v)

    # Count cross-edges per element
    for ml in sorted(parent_map):
        for p, vs in parent_map[ml].items():
            for v in vs:
                element_cross_count[v] += len(vs)

    print("\nCross-edge structure per missing leaf:")
    total_cross_from_component = 0
    for ml in sorted(parent_map):
        parents = parent_map[ml]
        n_parents = len(parents)
        counts = sorted(len(vs) for vs in parents.values())
        n_cross = sum(c * c for c in counts)
        total_cross_from_component += n_cross
        print(
            f"  ml={ml}: {n_parents} K_{{1,4}} parents, "
            f"elements/parent: {counts}, cross-edges: {n_cross}"
        )

    print(f"Total cross-edges from this component: {total_cross_from_component}")

    # Degree of L(cross-edges)
    all_degrees: list[int] = []
    for ml in sorted(parent_map):
        for p, vs in parent_map[ml].items():
            for a in vs:
                for b_mirror in vs:
                    deg = (
                        (element_cross_count[a] - 1)
                        + (element_cross_count[b_mirror] - 1)
                    )
                    all_degrees.append(deg)

    deg_dist: dict[int, int] = defaultdict(int)
    for d in all_degrees:
        deg_dist[d] += 1

    print("\n=== Result ===")
    print(f"|Gamma_6(K_{{1,6}})| from this component: {len(all_degrees)}")
    print(f"Degree distribution: {dict(sorted(deg_dist.items()))}")

    if len(deg_dist) == 1:
        d0 = list(deg_dist.keys())[0]
        print(f"\n*** d_0(K_{{1,6}}) = {d0} ***")
    else:
        print("\nNot regular from this component's perspective.")
        print(f"Min degree: {min(deg_dist)}, Max degree: {max(deg_dist)}")

    print("\nSummary & pattern check (d_0 = 2^(r-1) - 2?):")
    known = [(3, 2), (4, 6), (5, 14)]
    d6 = list(deg_dist.keys())[0] if len(deg_dist) == 1 else max(deg_dist)
    known.append((6, d6))
    for r, d in known:
        pred = 2 ** (r - 1) - 2
        sl = math.log2(d - 2) if d > 2 else 0
        print(
            f"  K_{{1,{r}}}: d_0={d:4d}, predicted={pred:4d}, "
            f"match={pred == d}, sub-leading=j*{sl:.4f}"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fiber analysis for iterated line graphs of K_n.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "fork-fiber",
        help="Compute the fiber graph of the fork tree in L^4(K_5).",
    )
    sub.add_parser(
        "k16-degree",
        help="Compute d_0(K_{1,6}) via parent analysis in L^5(K_6).",
    )

    args = parser.parse_args()

    dispatch = {
        "fork-fiber": cmd_fork_fiber,
        "k16-degree": cmd_k16_degree,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
