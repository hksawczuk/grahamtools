from __future__ import annotations

import argparse
from itertools import combinations, permutations
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Tuple, Set, Iterable, Optional

from collections import Counter

import math

# Reuse from earlier:
# - expand_to_simple_base_edges_id(v, k, endpoints_by_level)
# - canon_key_ir_bitset(edges, n)
# - equitable_partition(adj) and _edges_to_adj_bitsets(edges, n)

def aut_size_via_color_classes(edges: List[Tuple[int, int]], n: int) -> int:
    """
    Count automorphisms of the simple graph on vertices {0..n-1}.
    Uses WL equitable partition to restrict permutations to within color classes.
    """
    adj = _edges_to_adj_bitsets(edges, n)
    colors = equitable_partition(adj)          # tuple length n
    classes = _color_classes(colors)           # list of lists of vertices (old vertices)

    # For automorphisms, each vertex must map to same-color vertex
    # So we only permute within each class and take the product action.
    # Build a permutation p: old -> old (bijection).
    # Check if p preserves adjacency using bitsets.

    # Precompute adjacency matrix access via bitsets:
    # edge(u,v) iff (adj[u] >> v) & 1 == 1

    classes_perms = [list(permutations(cls)) for cls in classes]  # can still be big if a class is large

    count = 0
    p = [-1] * n  # mapping old -> old

    def backtrack(i: int):
        nonlocal count
        if i == len(classes):
            # verify automorphism
            # condition: for all u, v, edge(u,v) == edge(p[u], p[v])
            for u in range(n):
                pu = p[u]
                au = adj[u]
                # iterate neighbors v>u in original
                neigh = au
                while neigh:
                    lsb = neigh & -neigh
                    v = lsb.bit_length() - 1
                    neigh ^= lsb
                    if v <= u:
                        continue
                    if ((adj[pu] >> p[v]) & 1) == 0:
                        return
                # also ensure non-edges map to non-edges:
                # easiest: compare full neighbor sets under mapping
                # Build mapped neighbor bitset and compare:
                mapped = 0
                neigh2 = au
                while neigh2:
                    lsb2 = neigh2 & -neigh2
                    v2 = lsb2.bit_length() - 1
                    neigh2 ^= lsb2
                    mapped |= 1 << p[v2]
                if mapped != adj[pu]:
                    return
            count += 1
            return

        cls = classes[i]
        for perm in classes_perms[i]:
            # set mapping for this class: cls[j] -> perm[j]
            ok = True
            for a, b in zip(cls, perm):
                if p[a] != -1:
                    ok = False
                    break
                p[a] = b
            if ok:
                backtrack(i + 1)
            # undo
            for a in cls:
                p[a] = -1

    backtrack(0)
    return count


def orbit_size_under_Sn(edges: List[Tuple[int, int]], n: int) -> int:
    """
    Orbit size of the labeled graph under S_n action = n! / |Aut(G)|.
    """
    aut = aut_size_via_color_classes(edges, n)
    return math.factorial(n) // aut


# ---------- bit indexing for K_n edges ----------
@lru_cache(maxsize=None)
def _pair_to_idx(n: int) -> Dict[Tuple[int, int], int]:
    d = {}
    t = 0
    for i in range(n):
        for j in range(i + 1, n):
            d[(i, j)] = t
            t += 1
    return d

def _edges_to_adj_bitsets(edges: List[Tuple[int, int]], n: int) -> List[int]:
    """adj[u] is an n-bit bitset of neighbors of u."""
    adj = [0] * n
    for u, v in edges:
        if u == v:
            continue
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def _encode_bitset_for_order(adj: List[int], order: List[int]) -> int:
    """
    Given an order = [v0,v1,...,v_{n-1}] meaning new label i corresponds to old vertex order[i],
    return the edge-bitset integer in the canonical edge-indexing (i<j).
    """
    n = len(order)
    pair_to_idx = _pair_to_idx(n)

    # inv maps old vertex -> new label
    inv = [0] * n
    for new, old in enumerate(order):
        inv[old] = new

    bits = 0
    for old_u in range(n):
        new_u = inv[old_u]
        neigh = adj[old_u]
        # iterate neighbors old_v > old_u to avoid double counting
        while neigh:
            lsb = neigh & -neigh
            old_v = (lsb.bit_length() - 1)
            neigh ^= lsb
            if old_v <= old_u:
                continue
            new_v = inv[old_v]
            if new_u < new_v:
                i, j = new_u, new_v
            else:
                i, j = new_v, new_u
            bits |= 1 << pair_to_idx[(i, j)]
    return bits

# ---------- 1-WL color refinement (equitable partition) ----------
def _refine_colors(adj: List[int], colors: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    One round of color refinement: newcolor(u) = hash( oldcolor(u), multiset{oldcolor(v): v~u} ).
    Implemented deterministically via sorting tuples.
    """
    n = len(adj)
    sigs = []
    for u in range(n):
        neigh = adj[u]
        cnt = Counter()
        while neigh:
            lsb = neigh & -neigh
            v = lsb.bit_length() - 1
            neigh ^= lsb
            cnt[colors[v]] += 1
        # signature: (own_color, sorted neighbor-color counts)
        sig = (colors[u], tuple(sorted(cnt.items())))
        sigs.append(sig)

    # relabel signatures to 0..c-1 deterministically
    uniq = {sig: i for i, sig in enumerate(sorted(set(sigs)))}
    return tuple(uniq[sigs[u]] for u in range(n))

def equitable_partition(adj: List[int], initial: Optional[Tuple[int, ...]] = None) -> Tuple[int, ...]:
    """
    Iterate refinement to a fixed point.
    """
    n = len(adj)
    if initial is None:
        # start with degrees (good cheap start)
        initial = tuple(int(adj[u].bit_count()) for u in range(n))

    colors = initial
    while True:
        newc = _refine_colors(adj, colors)
        if newc == colors:
            return colors
        colors = newc

def _color_classes(colors: Tuple[int, ...]) -> List[List[int]]:
    classes: Dict[int, List[int]] = {}
    for v, c in enumerate(colors):
        classes.setdefault(c, []).append(v)
    # sort by (size, then members) for determinism
    cls = list(classes.values())
    cls.sort(key=lambda L: (len(L), L))
    return cls


@lru_cache(maxsize=None)
def _pair_to_idx_list(n: int):
    """Return a 2D table idx[i][j] = bit position for 0<=i<j<n, else -1."""
    idx = [[-1]*n for _ in range(n)]
    t = 0
    for i in range(n):
        for j in range(i+1, n):
            idx[i][j] = t
            t += 1
    return idx

@lru_cache(maxsize=None)
def _all_perms(n: int):
    return list(permutations(range(n)))

def canon_key_bruteforce_bitset(edges: list[tuple[int,int]], n: int) -> int:
    """
    True canonical key under full S_n: min over all permutations.
    edges: list of unique unordered edges (u<v), 0-based.
    """
    idx = _pair_to_idx_list(n)
    perms = _all_perms(n)

    best = None
    for p in perms:
        bits = 0
        for u, v in edges:
            pu, pv = p[u], p[v]
            if pu > pv:
                pu, pv = pv, pu
            bits |= 1 << idx[pu][pv]
        if best is None or bits < best:
            best = bits
    return best or 0

@lru_cache(maxsize=None)
def _idx_to_pair(n: int) -> List[Tuple[int, int]]:
    """Inverse of your edge indexing: bit index -> (i,j) with 0<=i<j<n."""
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    return pairs

def _is_forest_edgebit(bits: int, n: int) -> bool:
    """
    Decide whether the simple graph on {0..n-1} encoded by edge-bitset `bits`
    is acyclic (a forest). Union-find cycle detection.
    """
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False  # cycle edge
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    pairs = _idx_to_pair(n)
    while bits:
        lsb = bits & -bits
        idx = lsb.bit_length() - 1
        bits ^= lsb
        u, v = pairs[idx]
        if not union(u, v):
            return False
    return True


# ----------------------------
# Internal representation
# ----------------------------
# Level 0: base vertices are 0..n-1
# Level t>=1: vertices are 0..(len(V_t)-1)
# endpoints_by_level[t][v] = (a,b) where a,b are vertex-IDs from level t-1
Endpoints = Tuple[int, int]


def canon_pair_int(a: int, b: int) -> Endpoints:
    return (a, b) if a <= b else (b, a)


# ----------------------------
# Generate levels for L^k(K_n)
# ----------------------------

def generate_levels_Kn_ids(n: int, k: int, prune_cycles: bool = False) -> Tuple[Dict[int, List[int]], Dict[int, List[Endpoints]]]:
    """
    Build levels 0..k of vertices of L^t(K_n) by incidence recursion.
    Returns:
      V_by_level[t] = list of vertex IDs [0..|V_t|-1]
      endpoints_by_level[t] = list indexed by vertex ID, storing endpoints (IDs from level t-1) for t>=1
    """
    if n <= 0:
        return {0: []}, {}

    V_by_level: Dict[int, List[int]] = {0: list(range(n))}
    endpoints_by_level: Dict[int, List[Endpoints]] = {}

    if k == 0:
        return V_by_level, endpoints_by_level

    # level 1 vertices = edges of K_n
    ep1: List[Endpoints] = []
    bit1: List[int] = []
    idx = _pair_to_idx_list(n)
    for i, j in combinations(range(n), 2):
        ep1.append((i, j))
        bit1.append(1 << idx[i][j])
    V_by_level[1] = list(range(len(ep1)))
    endpoints_by_level[1] = ep1

    # Track base-edge bitsets for each vertex at each level (local, just for pruning)
    bits_prev = bit1
    ok_prev = [True] * len(bits_prev)  # level 1 always forests

    for level in range(2, k + 1):
        ep_prev = endpoints_by_level[level - 1]

        # incidence built only from "ok" prev vertices if pruning
        incidence: Dict[int, List[int]] = defaultdict(list)
        for e_id, (a, b) in enumerate(ep_prev):
            if prune_cycles and not ok_prev[e_id]:
                continue
            incidence[a].append(e_id)
            incidence[b].append(e_id)

        next_pairs: Set[Endpoints] = set()
        bits_next_map: Dict[Endpoints, int] = {}

        for inc_edges in incidence.values():
            for e1, e2 in combinations(inc_edges, 2):
                p = canon_pair_int(e1, e2)
                if p in next_pairs:
                    continue

                if prune_cycles:
                    b = bits_prev[e1] | bits_prev[e2]
                    if not _is_forest_edgebit(b, n):
                        continue
                    bits_next_map[p] = b

                next_pairs.add(p)

        ep_next = sorted(next_pairs)
        endpoints_by_level[level] = ep_next
        V_by_level[level] = list(range(len(ep_next)))

        if prune_cycles:
            # align bits_next and ok_next with ep_next ordering
            bits_next = [bits_next_map[p] for p in ep_next]
            ok_next = [True] * len(bits_next)  # by construction forests
        else:
            # still need to keep lists sized correctly for the next loop
            bits_next = [0] * len(ep_next)
            ok_next = [True] * len(ep_next)

        bits_prev, ok_prev = bits_next, ok_next

    return V_by_level, endpoints_by_level


# ----------------------------
# Human-readable label formatting (lazy + memoized)
# ----------------------------

def format_label(v: int, level: int, endpoints_by_level: Dict[int, List[Endpoints]], sep_after_level: int = 1) -> str:
    """
    Convert internal ID at given level to your recursive string label.
    sep_after_level: concatenate through this level, then separate with '|'
    """
    @lru_cache(maxsize=None)
    def rec(v_id: int, lvl: int) -> str:
        if lvl == 0:
            return str(v_id + 1)  # match your 1..n base labels
        a, b = endpoints_by_level[lvl][v_id]
        sa, sb = rec(a, lvl - 1), rec(b, lvl - 1)
        if sa <= sb:
            left, right = sa, sb
        else:
            left, right = sb, sa
        if lvl <= sep_after_level:
            return f"{left}{right}"
        return f"{left}|{right}"

    return rec(v, level)


# ----------------------------
# Expand label to base edges (memoized)
# ----------------------------

def expand_to_simple_base_edges_id(
    v: int,
    level: int,
    endpoints_by_level: Dict[int, List[Endpoints]],
) -> List[Tuple[int, int]]:
    """
    Expand internal ID at iterate `level` down to unique base edges of K_n (0-based),
    then return sorted list of (i,j) with i<j (still 0-based).
    """
    @lru_cache(maxsize=None)
    def rec(v_id: int, lvl: int) -> frozenset[Tuple[int, int]]:
        if lvl == 0:
            return frozenset()
        if lvl == 1:
            a, b = endpoints_by_level[1][v_id]
            return frozenset({(a, b) if a < b else (b, a)})
        a, b = endpoints_by_level[lvl][v_id]
        return rec(a, lvl - 1) | rec(b, lvl - 1)

    return sorted(rec(v, level))




# ----------------------------
# Canonical form under S_n (much faster bitset encoding)
# ----------------------------

def _pair_index_map(n: int) -> Dict[Tuple[int, int], int]:
    """
    Map unordered pair (i,j), 0<=i<j<n, to bit position in [0, C(n,2)).
    """
    idx = {}
    t = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx[(i, j)] = t
            t += 1
    return idx


def canon_simple_graph_key_bitset(edges: List[Tuple[int, int]], n: int, perms: List[Tuple[int, ...]]) -> int:
    """
    Canonical key for a simple graph under relabeling of {0..n-1}.
    Returns the minimal bitset integer over all vertex permutations.
    """
    pair_to_idx = _pair_index_map(n)

    # edge list already unique unordered (i<j), 0-based
    best = None
    for p in perms:
        # p maps old vertex i -> new vertex p[i]
        bits = 0
        for (u, v) in edges:
            pu, pv = p[u], p[v]
            if pu > pv:
                pu, pv = pv, pu
            bits |= 1 << pair_to_idx[(pu, pv)]
        if best is None or bits < best:
            best = bits
    return best if best is not None else 0


def iso_classes_with_stats(
    Vk: List[int],
    n: int,
    k: int,
    endpoints_by_level,
) -> List[dict]:
    """
    Returns a list of dicts, one per isomorphism class:
      {
        "key": <canonical key>,
        "rep": <vertex id representative>,
        "freq": <# labels in this class>,
        "orbit": <size of S_n orbit>,
        "aut": <|Aut(H)|>   (optional but useful)
      }
    """
    buckets: Dict[int, dict] = {}

    for v in Vk:
        edges = expand_to_simple_base_edges_id(v, k, endpoints_by_level)
        key = canon_key_bruteforce_bitset(edges, n)

        if key not in buckets:
            buckets[key] = {"key": key, "rep": v, "freq": 0, "edges_rep": edges}
        buckets[key]["freq"] += 1

    # compute orbit/aut per class (once per bucket)
    out = []
    for b in buckets.values():
        edges = b["edges_rep"]
        aut = aut_size_via_color_classes(edges, n)
        orbit = math.factorial(n) // aut
        freq = b["freq"]
        coeff = int(freq / orbit) if orbit else 0

        out.append({
            "key": b["key"],
            "rep": b["rep"],
            "freq": freq,
            "aut": aut,
            "orbit": orbit,
            "coeff": coeff,
        })


    # deterministic order: sort by orbit desc then freq desc then key
    out.sort(key=lambda d: (-d["orbit"], -d["freq"], d["key"]))
    return out

def reps_by_graph_iso_ids(Vk: List[int], n: int, k: int, endpoints_by_level) -> List[int]:
    seen: Set[int] = set()
    reps: List[int] = []

    for v in Vk:
        edges = expand_to_simple_base_edges_id(v, k, endpoints_by_level)
        key = canon_key_bruteforce_bitset(edges, n)

        if key not in seen:
            seen.add(key)
            reps.append(v)

    return reps

def dump_fiber_for_labeled_edges(n: int, k: int, endpoints_by_level, target_edges_1based, sep_after: int):
    # normalize target to 0-based sorted unique
    target01 = []
    for u, v in target_edges_1based:
        u -= 1; v -= 1
        if u > v: u, v = v, u
        target01.append((u, v))
    target01 = sorted(set(target01))

    hits = []
    for v in range(len(endpoints_by_level[k])):
        edges01 = expand_to_simple_base_edges_id(v, k, endpoints_by_level)
        if edges01 == target01:
            hits.append(v)

    print(f"labeled fiber size = {len(hits)}  target={target_edges_1based}")
    for v in hits:
        lab = format_label(v, k, endpoints_by_level, sep_after_level=sep_after)
        a, b = endpoints_by_level[k][v]
        la = format_label(a, k-1, endpoints_by_level, sep_after_level=sep_after)
        lb = format_label(b, k-1, endpoints_by_level, sep_after_level=sep_after)
        print(f"v={v:5d}  {lab}   parents: {la}  +  {lb}")


import math
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from typing import Optional, List, Tuple

def _normalize_target_1based(target_edges_1based: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    # normalize target to 0-based sorted unique, i<j
    target01 = []
    for u, v in target_edges_1based:
        u -= 1; v -= 1
        if u > v: u, v = v, u
        target01.append((u, v))
    return sorted(set(target01))
def plot_fiber_multigraph_sum_separate_windows(
    n: int,
    k: int,
    endpoints_by_level,
    target_edges_1based: List[Tuple[int,int]],
    sep_after: int = 1,
    *,
    base_width: float = 2.0,
    node_size: int = 350,
    show_node_labels: bool = True,
    max_windows: Optional[int] = None,
):
    assert k >= 1, "k must be >= 1"

    target01 = _normalize_target_1based(target_edges_1based)

    # Find hits by underlying simple support (your current fiber definition)
    hits = []
    for v in range(len(endpoints_by_level[k])):
        edges01 = expand_to_simple_base_edges_id(v, k, endpoints_by_level)
        if edges01 == target01:
            hits.append(v)

    if max_windows is not None:
        hits = hits[:max_windows]

    print(f"labeled fiber size = {len(hits)}  target={target_edges_1based}")
    if not hits:
        return

    # fixed base positions
    base_nodes = list(range(1, n + 1))
    pos = nx.circular_layout(base_nodes)

    for v in hits:
        a, b = endpoints_by_level[k][v]

        child = expand_to_base_edge_multiset_id(v, k, endpoints_by_level)
        left  = expand_to_base_edge_multiset_id(a, k-1, endpoints_by_level)
        right = expand_to_base_edge_multiset_id(b, k-1, endpoints_by_level)
        ok = (left + right) == child

        lab_v = format_label(v, k, endpoints_by_level, sep_after_level=sep_after)
        lab_a = format_label(a, k-1, endpoints_by_level, sep_after_level=sep_after)
        lab_b = format_label(b, k-1, endpoints_by_level, sep_after_level=sep_after)

        print(f"v={v:5d}  {lab_v}   parents: {lab_a}  +  {lab_b}   (ok={ok})")

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        fig.suptitle(f"v={v}   {lab_a} + {lab_b} = {lab_v}   (ok={ok})", fontsize=11)

        _draw_weighted_simple_graph_on_base(
            axes[0], n, left, pos,
            title=f"Left parent\n{lab_a}",
            node_size=node_size, base_width=base_width, show_node_labels=show_node_labels
        )
        _draw_weighted_simple_graph_on_base(
            axes[1], n, right, pos,
            title=f"Right parent\n{lab_b}",
            node_size=node_size, base_width=base_width, show_node_labels=show_node_labels
        )
        _draw_weighted_simple_graph_on_base(
            axes[2], n, child, pos,
            title=f"Child\n{lab_v}",
            node_size=node_size, base_width=base_width, show_node_labels=show_node_labels
        )

        plt.tight_layout()

    # This will display all created figures as separate windows (GUI backend).
    plt.show()


def _draw_weighted_simple_graph_on_base(
    ax,
    n: int,
    edge_counts: Counter[Tuple[int,int]],
    pos,
    title: str,
    *,
    node_size: int = 350,
    base_width: float = 2.0,
    show_node_labels: bool = True,
):
    """
    Draws a weighted simple graph on base vertices 1..n, where multiplicity controls linewidth.
    """
    base_nodes = list(range(1, n + 1))

    G = nx.Graph()
    G.add_nodes_from(base_nodes)

    # Add weighted edges
    edges = []
    widths = []
    for (u0, v0), c in sorted(edge_counts.items()):
        u, v = u0 + 1, v0 + 1  # to 1-based for display
        G.add_edge(u, v, weight=c)
        edges.append((u, v))
        widths.append(base_width * max(1, c))

    ax.axis("off")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, linewidths=1.0, edgecolors="black")
    if edges:
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, width=widths)
    if show_node_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)
    ax.set_title(title, fontsize=10)

def plot_fiber_multigraph_sum(
    n: int,
    k: int,
    endpoints_by_level,
    target_edges_1based: List[Tuple[int,int]],
    sep_after: int = 1,
    *,
    out_path: Optional[str] = None,
    base_width: float = 2.0,
    node_size: int = 350,
    show_node_labels: bool = True,
    max_rows: Optional[int] = None,
):
    """
    Find the labeled fiber over the given target (by underlying SIMPLE base-edge set),
    then for each fiber element v at level k, plot:

        (parent a at level k-1)  +  (parent b at level k-1)  =  (child v at level k)

    as multigraphs on the base vertex set {1..n}, with linewidth proportional to multiplicity.
    """
    assert k >= 1, "k must be >= 1"

    target01 = _normalize_target_1based(target_edges_1based)

    # Find hits by underlying simple support (your current definition of the labeled fiber)
    hits = []
    for v in range(len(endpoints_by_level[k])):
        edges01 = expand_to_simple_base_edges_id(v, k, endpoints_by_level)
        if edges01 == target01:
            hits.append(v)

    if max_rows is not None:
        hits = hits[:max_rows]

    print(f"labeled fiber size = {len(hits)}  target={target_edges_1based}")

    if not hits:
        return

    # Fixed base positions for consistent comparison
    base_nodes = list(range(1, n + 1))
    pos = nx.circular_layout(base_nodes)

    rows = len(hits)
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.6 * rows))
    if rows == 1:
        axes = [axes]  # make it indexable as axes[row][col]

    for r, v in enumerate(hits):
        a, b = endpoints_by_level[k][v]

        # multiset expansions
        child = expand_to_base_edge_multiset_id(v, k, endpoints_by_level)
        left  = expand_to_base_edge_multiset_id(a, k-1, endpoints_by_level)
        right = expand_to_base_edge_multiset_id(b, k-1, endpoints_by_level)

        # sanity check: left + right == child
        ok = (left + right) == child
        if not ok:
            print(f"[WARN] multiplicity mismatch at v={v}: (left+right)!=child")

        # labels
        lab_v = format_label(v, k, endpoints_by_level, sep_after_level=sep_after)
        lab_a = format_label(a, k-1, endpoints_by_level, sep_after_level=sep_after)
        lab_b = format_label(b, k-1, endpoints_by_level, sep_after_level=sep_after)

        axL, axR, axC = axes[r][0], axes[r][1], axes[r][2]

        _draw_weighted_simple_graph_on_base(
            axL, n, left, pos,
            title=f"Left parent\n{lab_a}",
            node_size=node_size, base_width=base_width, show_node_labels=show_node_labels
        )
        _draw_weighted_simple_graph_on_base(
            axR, n, right, pos,
            title=f"Right parent\n{lab_b}",
            node_size=node_size, base_width=base_width, show_node_labels=show_node_labels
        )
        _draw_weighted_simple_graph_on_base(
            axC, n, child, pos,
            title=f"Child (ok={ok})\n{lab_v}",
            node_size=node_size, base_width=base_width, show_node_labels=show_node_labels
        )

        # also print the identity in text
        print(f"v={v:5d}  {lab_v}   parents: {lab_a}  +  {lab_b}   (ok={ok})")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()


def plot_reps_grid(
    classes: List[dict],
    n: int,
    k: int,
    endpoints_by_level,
    cols: int = 4,
    sep_after: int = 1,
    show_label: bool = False,
    show_stats_title: bool = True,
    out_path: Optional[str] = None,
):
    import math
    import networkx as nx
    import matplotlib.pyplot as plt

    if cols <= 0:
        cols = 1
    rows = math.ceil(len(classes) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    # axes can be a single Axes if rows=cols=1
    if rows == 1 and cols == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes.flat)

    # Fixed vertex positions across all subplots for easy comparison
    base_nodes = list(range(1, n + 1))
    pos = nx.circular_layout(base_nodes)  # deterministic layout (no randomness)

    for ax in axes_list[len(classes):]:
        ax.axis("off")

    for ax, c in zip(axes_list, classes):
        # build representative edge list (1-based)
        edges01 = expand_to_simple_base_edges_id(c["rep"], k, endpoints_by_level)
        edges = [(u + 1, v + 1) for (u, v) in edges01]

        G = nx.Graph()
        G.add_nodes_from(base_nodes)   # keep isolates for consistent pictures
        G.add_edges_from(edges)

        ax.axis("off")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=350, linewidths=1.0, edgecolors="black")
        nx.draw_networkx_edges(G, pos, ax=ax, width=2.0)

        # labels (optional)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10)

        if show_stats_title:
            title = f"coeff={c['coeff']}  freq={c['freq']}"
            if show_label:
                lab = format_label(c["rep"], k, endpoints_by_level, sep_after_level=sep_after)
                title = f"{title}\n{lab}"
            ax.set_title(title, fontsize=10)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()

from collections import Counter
from functools import lru_cache
from typing import Tuple, Dict, List

def expand_to_base_edge_multiset_id(
    v: int,
    level: int,
    endpoints_by_level: Dict[int, List[Endpoints]],
) -> Counter[Tuple[int, int]]:
    """
    Expand internal ID at iterate `level` down to a multiset of base edges of K_n (0-based),
    returned as Counter[(i,j)] with i<j. Multiplicities are preserved by addition.
    """
    @lru_cache(maxsize=None)
    def rec(v_id: int, lvl: int) -> Tuple[Tuple[Tuple[int, int], int], ...]:
        # Return a hashable representation: sorted tuple of ((u,v), count).
        if lvl == 0:
            return tuple()
        if lvl == 1:
            a, b = endpoints_by_level[1][v_id]
            if a > b:
                a, b = b, a
            return (((a, b), 1),)

        a, b = endpoints_by_level[lvl][v_id]
        A = Counter(dict(rec(a, lvl - 1)))
        B = Counter(dict(rec(b, lvl - 1)))
        C = A + B
        return tuple(sorted(C.items()))

    return Counter(dict(rec(v, level)))

def fiber_graph_stats_for_target_support(
    n: int,
    k: int,
    endpoints_by_level,
    target_edges_1based: List[Tuple[int, int]],
):
    """
    Build the fiber graph Γ_k(G) where G is the labeled SIMPLE support specified by target_edges_1based.
    Vertices: level-k IDs v with expand_to_simple_base_edges_id(v,k)==target
    Edges: v--w iff their level-(k-1) endpoint pairs share exactly one endpoint (line-graph adjacency)
    """
    assert k >= 1

    # normalize target to 0-based unique sorted (i<j)
    target01 = _normalize_target_1based(target_edges_1based)

    # collect fiber vertices (level-k vertex IDs)
    hits = []
    for v in range(len(endpoints_by_level[k])):
        edges01 = expand_to_simple_base_edges_id(v, k, endpoints_by_level)
        if edges01 == target01:
            hits.append(v)

    m = len(hits)
    print(f"[fiber] n={n} k={k}  |F_k|={m}  target={target_edges_1based}")
    if m == 0:
        return

    # map level-k vertex id -> 0..m-1
    idx_of = {v: i for i, v in enumerate(hits)}

    # incidence from level-(k-1) vertex -> list of fiber-vertex indices incident to it
    inc: Dict[int, List[int]] = defaultdict(list)
    for v in hits:
        a, b = endpoints_by_level[k][v]
        i = idx_of[v]
        inc[a].append(i)
        inc[b].append(i)

    # build adjacency sets inside fiber
    nbrs = [set() for _ in range(m)]
    for lst in inc.values():
        # any two fiber vertices sharing this endpoint are adjacent
        for i, j in combinations(lst, 2):
            nbrs[i].add(j)
            nbrs[j].add(i)

    degs = [len(nbrs[i]) for i in range(m)]
    deg_counter = Counter(degs)
    print(f"[fiber] degree multiset: {dict(sorted(deg_counter.items()))}")

    # regularity check
    is_regular = (len(deg_counter) == 1)
    if is_regular:
        d = degs[0]
        print(f"[fiber] regular: YES (d={d})")
    else:
        print("[fiber] regular: NO")

    # connectivity check (BFS)
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in nbrs[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    print(f"[fiber] connected: {'YES' if len(seen)==m else 'NO'}  (component size {len(seen)}/{m})")

    # optional: sanity check edge count
    E = sum(degs) // 2
    print(f"[fiber] |E(Γ_k)|={E}")

    return {
        "m": m,
        "deg_counter": deg_counter,
        "is_regular": is_regular,
        "d": degs[0] if is_regular else None,
        "connected": (len(seen) == m),
        "E": E,
        "hits": hits,  # level-k IDs in the fiber
    }

def fiber_parent_support_breakdown(
    n: int,
    k: int,
    endpoints_by_level,
    target_edges_1based: List[Tuple[int, int]],
    *,
    sep_after: int = 1,
    show_examples: int = 10,
):
    """
    For target support G (specified by simple base edges), analyze fiber elements u at grade k:
      u in F_k(G)  =>  u has parents (a,b) at grade k-1.
    Report how many parents are also in the same fiber F_{k-1}(G), i.e. closure diagnostics.

    This is the exact check you want for "is the fiber closed under factorization at this grade?"
    """
    assert k >= 1

    target01 = _normalize_target_1based(target_edges_1based)

    # helper: support key of any level-(t) vertex id = sorted tuple of 0-based base edges
    @lru_cache(maxsize=None)
    def support_key(v: int, t: int) -> Tuple[Tuple[int, int], ...]:
        return tuple(expand_to_simple_base_edges_id(v, t, endpoints_by_level))

    target_key = tuple(target01)

    # collect fiber vertices at level k and level k-1
    Fk = []
    for v in range(len(endpoints_by_level[k])):
        if support_key(v, k) == target_key:
            Fk.append(v)

    Fkm1 = set()
    if k - 1 >= 1:
        for v in range(len(endpoints_by_level[k - 1])):
            if support_key(v, k - 1) == target_key:
                Fkm1.add(v)
    else:
        # grade 0 has no support edges, so F_0(target) is empty unless target is empty
        Fkm1 = set()

    print(f"[fiber-check] n={n} k={k}  target={target_edges_1based}")
    print(f"[fiber-check] |F_k|={len(Fk)}   |F_(k-1)|={len(Fkm1)}")

    # breakdown counts
    c2 = 0  # both parents in F_{k-1}(target)
    c1 = 0  # exactly one parent in
    c0 = 0  # neither parent in

    # for examples
    ex2, ex1, ex0 = [], [], []

    for v in Fk:
        a, b = endpoints_by_level[k][v]
        in_a = a in Fkm1
        in_b = b in Fkm1
        s = int(in_a) + int(in_b)

        if s == 2:
            c2 += 1
            if len(ex2) < show_examples:
                ex2.append((v, a, b))
        elif s == 1:
            c1 += 1
            if len(ex1) < show_examples:
                ex1.append((v, a, b))
        else:
            c0 += 1
            if len(ex0) < show_examples:
                ex0.append((v, a, b))

    print(f"[fiber-check] parents in same fiber: 2->{c2}  1->{c1}  0->{c0}")
    print(f"[fiber-check] closed at grade k? {'YES' if c1==0 and c0==0 else 'NO'}")

    def _print_examples(tag: str, triples: List[Tuple[int,int,int]]):
        if not triples:
            return
        print(f"\n[{tag}] showing up to {len(triples)} examples")
        for (v, a, b) in triples:
            lab_v = format_label(v, k, endpoints_by_level, sep_after_level=sep_after)
            lab_a = format_label(a, k-1, endpoints_by_level, sep_after_level=sep_after)
            lab_b = format_label(b, k-1, endpoints_by_level, sep_after_level=sep_after)

            sa = support_key(a, k-1)
            sb = support_key(b, k-1)

            # convert supports to 1-based for readability
            sa1 = [(x+1, y+1) for (x,y) in sa]
            sb1 = [(x+1, y+1) for (x,y) in sb]

            print(f"v={v:5d}  {lab_v}")
            print(f"   parents: a={a:5d} {lab_a}   supp(a)={sa1}")
            print(f"            b={b:5d} {lab_b}   supp(b)={sb1}")

    if show_examples > 0:
        _print_examples("both parents in fiber", ex2)
        _print_examples("exactly one parent in fiber", ex1)
        _print_examples("no parents in fiber", ex0)

    return {
        "Fk_size": len(Fk),
        "Fkm1_size": len(Fkm1),
        "both": c2,
        "one": c1,
        "none": c0,
    }

def fiber_degree_by_parent_type(
    n: int,
    k: int,
    endpoints_by_level,
    target_edges_1based: List[Tuple[int, int]],
):
    """
    For each vertex v in F_k(target), classify v by whether its two parents
    are in F_{k-1}(target) (type SS), exactly one is (type SC), or none (type CC),
    and report degree stats within Γ_k(target) by type.
    """
    assert k >= 1

    target01 = _normalize_target_1based(target_edges_1based)
    target_key = tuple(target01)

    @lru_cache(maxsize=None)
    def support_key(v: int, t: int) -> Tuple[Tuple[int, int], ...]:
        return tuple(expand_to_simple_base_edges_id(v, t, endpoints_by_level))

    # fiber at level k
    hits = []
    for v in range(len(endpoints_by_level[k])):
        if support_key(v, k) == target_key:
            hits.append(v)

    m = len(hits)
    print(f"[fiber-type-deg] n={n} k={k} |F_k|={m}")

    if m == 0:
        return

    idx_of = {v: i for i, v in enumerate(hits)}

    # fiber at level k-1
    Fkm1 = set()
    for v in range(len(endpoints_by_level[k - 1])):
        if support_key(v, k - 1) == target_key:
            Fkm1.add(v)

    # build Γ_k adjacency via incidence
    inc = defaultdict(list)
    for v in hits:
        a, b = endpoints_by_level[k][v]
        i = idx_of[v]
        inc[a].append(i)
        inc[b].append(i)

    nbrs = [set() for _ in range(m)]
    for lst in inc.values():
        for i, j in combinations(lst, 2):
            nbrs[i].add(j)
            nbrs[j].add(i)

    # classify and collect degrees
    deg_SS, deg_SC, deg_CC = [], [], []
    for v in hits:
        i = idx_of[v]
        a, b = endpoints_by_level[k][v]
        s = int(a in Fkm1) + int(b in Fkm1)
        d = len(nbrs[i])
        if s == 2:
            deg_SS.append(d)
        elif s == 1:
            deg_SC.append(d)
        else:
            deg_CC.append(d)

    def summarize(name: str, L: list[int]):
        if not L:
            print(f"[fiber-type-deg] {name}: none")
            return
        c = Counter(L)
        print(f"[fiber-type-deg] {name}: count={len(L)}  degrees={dict(sorted(c.items()))}")

    summarize("SS (both parents in fiber)", deg_SS)
    summarize("SC (exactly one parent in fiber)", deg_SC)
    summarize("CC (no parents in fiber)", deg_CC)

    return {
        "SS": Counter(deg_SS),
        "SC": Counter(deg_SC),
        "CC": Counter(deg_CC),
        "counts": (len(deg_SS), len(deg_SC), len(deg_CC)),
    }

def fiber_SC_other_parent_support_hist(
    n: int,
    k: int,
    endpoints_by_level,
    target_edges_1based: List[Tuple[int, int]],
    *,
    sep_after: int = 1,
    show_examples_per_type: int = 3,
):
    """
    For SC-type vertices v in F_k(target): exactly one parent in F_{k-1}(target),
    tally the support of the OTHER parent (the one not in the fiber).
    Prints a histogram over those supports (as 1-based edge lists).
    """
    assert k >= 1

    target01 = _normalize_target_1based(target_edges_1based)
    target_key = tuple(target01)

    @lru_cache(maxsize=None)
    def support_key(v: int, t: int) -> Tuple[Tuple[int, int], ...]:
        return tuple(expand_to_simple_base_edges_id(v, t, endpoints_by_level))

    # fiber at level k
    Fk = []
    for v in range(len(endpoints_by_level[k])):
        if support_key(v, k) == target_key:
            Fk.append(v)

    # fiber at level k-1
    Fkm1 = set()
    for v in range(len(endpoints_by_level[k - 1])):
        if support_key(v, k - 1) == target_key:
            Fkm1.add(v)

    print(f"[SC-hist] n={n} k={k}  target={target_edges_1based}")
    print(f"[SC-hist] |F_k|={len(Fk)}   |F_(k-1)|={len(Fkm1)}")

    hist = Counter()
    examples: Dict[Tuple[Tuple[int,int],...], List[Tuple[int,int,int]]] = defaultdict(list)
    sc_count = 0

    for v in Fk:
        a, b = endpoints_by_level[k][v]
        in_a = a in Fkm1
        in_b = b in Fkm1
        if int(in_a) + int(in_b) != 1:
            continue  # not SC

        sc_count += 1
        other = b if in_a else a
        other_supp = support_key(other, k - 1)
        hist[other_supp] += 1
        if show_examples_per_type > 0 and len(examples[other_supp]) < show_examples_per_type:
            examples[other_supp].append((v, a, b))

    print(f"[SC-hist] SC count = {sc_count}")
    print(f"[SC-hist] distinct other-parent supports = {len(hist)}")

    # pretty print histogram as 1-based edges
    def to1(supp0: Tuple[Tuple[int,int],...]) -> List[Tuple[int,int]]:
        return [(u+1, v+1) for (u,v) in supp0]

    for supp0, cnt in hist.most_common():
        print(f"  count={cnt:6d}  other_support={to1(supp0)}")

    # optional witnesses
    if show_examples_per_type > 0:
        for supp0, cnt in hist.most_common():
            print(f"\n[SC-hist] examples for other_support={to1(supp0)} (count={cnt})")
            for (v, a, b) in examples[supp0]:
                lab_v = format_label(v, k, endpoints_by_level, sep_after_level=sep_after)
                lab_a = format_label(a, k-1, endpoints_by_level, sep_after_level=sep_after)
                lab_b = format_label(b, k-1, endpoints_by_level, sep_after_level=sep_after)

                sa = to1(support_key(a, k-1))
                sb = to1(support_key(b, k-1))

                print(f"v={v:5d}  {lab_v}")
                print(f"   a={a:5d} {lab_a}  supp(a)={sa}")
                print(f"   b={b:5d} {lab_b}  supp(b)={sb}")

    return {
        "sc_count": sc_count,
        "hist": hist,
    }

def fiber_SS_jf_support_hist(
    n: int,
    k: int,
    endpoints_by_level,
    target_edges_1based: List[Tuple[int, int]],
    *,
    sep_after: int = 1,
    show_examples_per_type: int = 3,
):
    """
    For SS-type vertices v in F_k(target): both parents in F_{k-1}(target),
    tally the support of the joining factor jf(a,b) in level k-2, where
        fac(v) = {a,b} and fac(a) ∩ fac(b) = {x} with x = jf(a,b).

    Prints a histogram over supp(x) (as 1-based edge lists).
    """
    assert k >= 2, "need k>=2 so parents are at k-1 and joining factor is at k-2"

    target01 = _normalize_target_1based(target_edges_1based)
    target_key = tuple(target01)

    @lru_cache(maxsize=None)
    def support_key(v: int, t: int) -> Tuple[Tuple[int, int], ...]:
        return tuple(expand_to_simple_base_edges_id(v, t, endpoints_by_level))

    def to1(supp0: Tuple[Tuple[int,int],...]) -> List[Tuple[int,int]]:
        return [(u+1, v+1) for (u,v) in supp0]

    # Build fiber membership at level k and k-1
    Fk: List[int] = []
    for v in range(len(endpoints_by_level[k])):
        if support_key(v, k) == target_key:
            Fk.append(v)

    Fkm1 = set()
    for v in range(len(endpoints_by_level[k - 1])):
        if support_key(v, k - 1) == target_key:
            Fkm1.add(v)

    print(f"[SS-jf-hist] n={n} k={k}  target={target_edges_1based}")
    print(f"[SS-jf-hist] |F_k|={len(Fk)}   |F_(k-1)|={len(Fkm1)}")

    hist = Counter()
    examples: Dict[Tuple[Tuple[int,int],...], List[Tuple[int,int,int,int]]] = defaultdict(list)
    ss_count = 0

    for v in Fk:
        a, b = endpoints_by_level[k][v]

        in_a = a in Fkm1
        in_b = b in Fkm1
        if not (in_a and in_b):
            continue  # not SS

        # Joining factor at level k-2:
        # a and b are vertices of L^{k-1}; each has endpoints in L^{k-2}
        a0, a1 = endpoints_by_level[k - 1][a]
        b0, b1 = endpoints_by_level[k - 1][b]

        common = None
        if a0 == b0 or a0 == b1:
            common = a0
        elif a1 == b0 or a1 == b1:
            common = a1

        if common is None:
            # This should never happen if (a,b) really form an edge in L^{k-1}
            # but keep it robust.
            continue

        ss_count += 1
        jf_supp = support_key(common, k - 2)
        hist[jf_supp] += 1

        if show_examples_per_type > 0 and len(examples[jf_supp]) < show_examples_per_type:
            examples[jf_supp].append((v, a, b, common))

    print(f"[SS-jf-hist] SS count = {ss_count}")
    print(f"[SS-jf-hist] distinct jf supports = {len(hist)}")

    for supp0, cnt in hist.most_common():
        print(f"  count={cnt:6d}  jf_support={to1(supp0)}   |supp|={len(supp0)}")

    if show_examples_per_type > 0:
        for supp0, cnt in hist.most_common():
            print(f"\n[SS-jf-hist] examples for jf_support={to1(supp0)} (count={cnt})")
            for (v, a, b, x) in examples[supp0]:
                lab_v = format_label(v, k, endpoints_by_level, sep_after_level=sep_after)
                lab_a = format_label(a, k-1, endpoints_by_level, sep_after_level=sep_after)
                lab_b = format_label(b, k-1, endpoints_by_level, sep_after_level=sep_after)
                lab_x = format_label(x, k-2, endpoints_by_level, sep_after_level=sep_after)

                sa = to1(support_key(a, k-1))
                sb = to1(support_key(b, k-1))
                sx = to1(support_key(x, k-2))

                print(f"v={v:5d}  {lab_v}")
                print(f"   a={a:5d} {lab_a}  supp(a)={sa}")
                print(f"   b={b:5d} {lab_b}  supp(b)={sb}")
                print(f"  jf={x:5d} {lab_x}  supp(jf)={sx}")

    return {
        "ss_count": ss_count,
        "hist": hist,
    }




# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g-reps", action="store_true", help="print one representative per simple-graph isomorphism class on {1..n}")
    ap.add_argument("n", type=int, help="n in K_n")
    ap.add_argument("k", type=int, help="iterate depth k (labels for L^k(K_n))")
    ap.add_argument("--sep-after", type=int, default=1, help="concatenate up to this level; then use '|'")
    ap.add_argument("--simple-edges", action="store_true", help="when printing labels, also print induced simple base-edge set (no multiplicity)")
    ap.add_argument("--class-stats", action="store_true", help="print one line per isomorphism class: rep, freq, orbit size")
    ap.add_argument("--full-labels", action="store_true", help="print recursive vertex labels like '12|34|56'")
    ap.add_argument("--plot-grid", action="store_true", help="plot representative graphs in a matplotlib grid")
    ap.add_argument("--plot-cols", type=int, default=4, help="number of columns in the plot grid")
    ap.add_argument("--plot-out", type=str, default=None, help="if set, save plot to this path instead of showing")
    ap.add_argument("--plot-title", action="store_true", help="include freq/coeff in subplot titles")
    ap.add_argument("--prune-cycles", action="store_true", help="prune vertices whose expanded base simple graph contains a cycle (forest-only)")
    ap.add_argument("--rep-label", action="store_true", help="print the full recursive label for the representative of each class")


    # after generating Vk:
    

    args = ap.parse_args()

    n, k = args.n, args.k

    V_by_level, endpoints_by_level = generate_levels_Kn_ids(n, k, prune_cycles=args.prune_cycles)
    Vk = V_by_level.get(k, [])

#     plot_fiber_multigraph_sum_separate_windows(
#         n, k,
#         endpoints_by_level,
#         target_edges_1based=[(1,2),(1,3),(1,4),(1,5)],
#         sep_after=args.sep_after,
#         # max_windows=20,  # optional
# )   

    # fiber_graph_stats_for_target_support(
    #         n=n,
    #         k=k,
    #         endpoints_by_level=endpoints_by_level,
    #         target_edges_1based=[(1,2),(1,3),(1,4),(1,5),(1,6)],
    #     )

    # fiber_parent_support_breakdown(
    #     n=n,
    #     k=k,  # check closure from 4 -> 5
    #     endpoints_by_level=endpoints_by_level,
    #     target_edges_1based=[(1,2),(1,3),(1,4),(1,5),(1,6)],
    #     sep_after=args.sep_after,
    #     show_examples=5,
    # )

    fiber_degree_by_parent_type(
        n=n, k=k,
        endpoints_by_level=endpoints_by_level,
        target_edges_1based=[(1,2),(1,3),(1,4),(1,5),(1,6),(1,7)],
    )

    # fiber_SC_other_parent_support_hist(
    #     n=n,
    #     k=k,
    #     endpoints_by_level=endpoints_by_level,
    #     target_edges_1based=[(1,2),(1,3),(1,4),(1,5)],
    #     sep_after=args.sep_after,
    #     show_examples_per_type=2,
    # )

    # fiber_SS_jf_support_hist(
    #     n=n, k=k,
    #     endpoints_by_level=endpoints_by_level,
    #     target_edges_1based=[(1,2),(1,3),(1,4),(1,5),(1,6)],
    #     sep_after=args.sep_after,
    #     show_examples_per_type=2,
    # )

    return

    if args.g_reps and not args.class_stats and k >= 1:
        Vk = reps_by_graph_iso_ids(Vk, n, k, endpoints_by_level)

    if args.class_stats:
        classes = iso_classes_with_stats(Vk, n, k, endpoints_by_level)
        if args.plot_grid:
            plot_reps_grid(
                classes,
                n=n,
                k=k,
                endpoints_by_level=endpoints_by_level,
                cols=args.plot_cols,
                sep_after=args.sep_after,
                show_label=args.full_labels,
                show_stats_title=args.plot_title,
                out_path=args.plot_out,
            )


    header_cols = ["coeff", "freq"]
    if args.rep_label:
        header_cols.append("rep_label")
    if args.simple_edges:
        header_cols.append("edges")

    print(
        f"{header_cols[0]:>6}  "
        f"{header_cols[1]:>6}"
        + (f"  {'rep_label':<30}" if args.rep_label else "")
        + (f"  edges" if args.simple_edges else "")
    )


    print(
        f"{header_cols[0]:>6}  "
        f"{header_cols[1]:>6}  "
        # f"{header_cols[2]:>6}  "
        # f"{header_cols[3]:>6}  "
        + (f"  {header_cols[4]:<20}" if args.full_labels else "")
        + (f"  {header_cols[-1]}" if args.simple_edges else "")
    )

    classes = []  # <-- add this default

    if args.g_reps and not args.class_stats and k >= 1:
        Vk = reps_by_graph_iso_ids(Vk, n, k, endpoints_by_level)

    if args.class_stats:
        classes = iso_classes_with_stats(Vk, n, k, endpoints_by_level)

        if args.plot_grid:
            plot_reps_grid(
                classes,
                n=n,
                k=k,
                endpoints_by_level=endpoints_by_level,
                cols=args.plot_cols,
                sep_after=args.sep_after,
                show_label=args.full_labels,
                show_stats_title=args.plot_title,
                out_path=args.plot_out,
            )

    # ----- only print class table if we actually computed classes -----
    if args.class_stats:
        header_cols = ["coeff", "freq"]
        if args.rep_label:
            header_cols.append("rep_label")
        if args.simple_edges:
            header_cols.append("edges")

        print(
            f"{header_cols[0]:>6}  "
            f"{header_cols[1]:>6}"
            + (f"  {'rep_label':<30}" if args.rep_label else "")
            + (f"  edges" if args.simple_edges else "")
        )

        for c in classes:
            row = (
                f"{c['coeff']:>6d}  "
                f"{c['freq']:>6d}"
            )

            if args.rep_label:
                rep_lab = format_label(
                    c["rep"],
                    k,
                    endpoints_by_level,
                    sep_after_level=args.sep_after
                )
                row += f"  {rep_lab:<30}"

            if args.simple_edges:
                edges01 = expand_to_simple_base_edges_id(c["rep"], k, endpoints_by_level)
                edges = [(i + 1, j + 1) for (i, j) in edges01]
                row += f"  {edges}"

            print(row)



if __name__ == "__main__":
    main()
