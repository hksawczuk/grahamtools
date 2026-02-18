#!/usr/bin/env python3
"""
Quick patch: run the same bootstrap but print ALL nonzero coeff_6 values,
and for the 6-edge type with coeff_6=80, reconstruct its edge list and degree sequence.
"""

import time
from itertools import combinations
from collections import defaultdict

def tree_canonical(adj, n):
    if n == 1: return ('V',)
    if n == 2: return ('E',)
    degree = {v: len(adj[v]) for v in adj}
    leaves = [v for v in adj if degree[v] == 1]
    removed = set()
    remaining = n
    while remaining > 2:
        new_leaves = []
        for leaf in leaves:
            removed.add(leaf)
            remaining -= 1
            for w in adj[leaf]:
                if w not in removed:
                    degree[w] -= 1
                    if degree[w] == 1: new_leaves.append(w)
        leaves = new_leaves
    centers = [v for v in adj if v not in removed]
    def encode(root, parent):
        cc = []
        for w in adj[root]:
            if w != parent: cc.append(encode(w, root))
        cc.sort()
        return tuple(cc)
    if len(centers) == 1:
        return ('C', encode(centers[0], -1))
    else:
        c0, c1 = centers
        e0, e1 = encode(c0, c1), encode(c1, c0)
        return ('B', min((e0, e1), (e1, e0)))

def edges_to_adj(el):
    adj = defaultdict(set)
    for u, v in el:
        adj[u].add(v); adj[v].add(u)
    return dict(adj)

def canonical_of_edges(el):
    if not el: return ('EMPTY',)
    adj = edges_to_adj(el)
    return tree_canonical(adj, len(adj))

def is_connected(el):
    if not el: return True
    adj = defaultdict(set)
    for u, v in el:
        adj[u].add(v); adj[v].add(u)
    verts = set(adj)
    start = next(iter(verts))
    vis = {start}; stack = [start]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in vis: vis.add(w); stack.append(w)
    return vis == verts

def relabel(el):
    verts = sorted(set(v for e in el for v in e))
    lab = {v: i for i, v in enumerate(verts)}
    return [(lab[u], lab[v]) for u, v in el], len(verts)

def line_graph_edges(edges):
    m = len(edges)
    if m == 0: return [], 0
    incident = defaultdict(list)
    for idx, (u, v) in enumerate(edges):
        incident[u].append(idx); incident[v].append(idx)
    new_edges = set()
    for v_inc in incident.values():
        for i in range(len(v_inc)):
            for j in range(i + 1, len(v_inc)):
                a, b = v_inc[i], v_inc[j]
                if a > b: a, b = b, a
                new_edges.add((a, b))
    return list(new_edges), m

def graham_value_at_k(edges, n, k, max_edges=30_000_000):
    cur = list(edges); cur_n = n
    for step in range(k):
        new_e, new_n = line_graph_edges(cur)
        if len(new_e) > max_edges: return None
        cur = new_e; cur_n = new_n
        if new_n == 0: return 0
    return cur_n

def enumerate_subtree_counts(edges):
    counts = defaultdict(int)
    reps = {}
    m = len(edges)
    for size in range(1, m + 1):
        for combo in combinations(range(m), size):
            sub = [edges[i] for i in combo]
            if is_connected(sub):
                c = canonical_of_edges(sub)
                counts[c] += 1
                if c not in reps: reps[c] = sub[:]
    return dict(counts), reps

def degree_sequence(edges):
    deg = defaultdict(int)
    for u, v in edges:
        deg[u] += 1; deg[v] += 1
    return sorted(deg.values(), reverse=True)

# ─── Main ───
edges_a = [
    (0,1),(0,2),(0,3),(0,4),(0,5),
    (1,6),(1,7),(1,8),(1,9),
    (2,10),(2,11),(3,12),(3,13),
    (6,14),(7,15),(8,16),(9,17),
]
edges_b = [
    (0,1),(0,2),(0,3),(0,4),(0,5),
    (1,6),(1,7),(1,8),(1,9),
    (2,10),(2,11),(3,12),(4,13),
    (6,14),(6,15),(7,16),(8,17),
]

print("Enumerating subtrees...", flush=True)
counts_a, reps_a = enumerate_subtree_counts(edges_a)
counts_b, reps_b = enumerate_subtree_counts(edges_b)
all_types = set(counts_a) | set(counts_b)
reps = {**reps_b, **reps_a}

type_data = {}
for c in all_types:
    rel, nv = relabel(reps[c])
    type_data[c] = {'edges': rel, 'n': nv, 'ne': len(rel)}

sorted_types = sorted(all_types, key=lambda c: (type_data[c]['ne'], str(c)))

print(f"Bootstrapping coeff_6 for {len(sorted_types)} types...", flush=True)
coeff6 = {}
gamma6 = {}

for idx, canon in enumerate(sorted_types):
    td = type_data[canon]
    gval = graham_value_at_k(td['edges'], td['n'], 6)
    if gval is None: continue
    gamma6[canon] = gval
    
    if td['ne'] <= 1:
        sub_counts = {canon: 1}
    else:
        sub_counts, _ = enumerate_subtree_counts(td['edges'])
    
    correction = sum(coeff6.get(sc, 0) * cnt for sc, cnt in sub_counts.items() if sc != canon and sc in coeff6)
    coeff6[canon] = gval - correction

print(f"\nAll nonzero coeff_6 values:")
print(f"{'edges':>5s} {'coeff₆':>10s} {'deg_seq':>20s} {'canonical':>50s}")
print("-" * 90)
for canon in sorted_types:
    if canon in coeff6 and coeff6[canon] != 0:
        td = type_data[canon]
        ds = degree_sequence(td['edges'])
        print(f"{td['ne']:>5d} {coeff6[canon]:>10d} {str(ds):>20s} {str(canon):>50s}")

# Also print the specific 6-edge type(s)
print("\n\nDetails of 6-edge types:")
for canon in sorted_types:
    if canon in coeff6 and type_data[canon]['ne'] == 6:
        td = type_data[canon]
        ds = degree_sequence(td['edges'])
        ca = counts_a.get(canon, 0)
        cb = counts_b.get(canon, 0)
        print(f"  canon={canon}")
        print(f"  edges={td['edges']}")
        print(f"  deg_seq={ds}")
        print(f"  γ₆={gamma6[canon]}, coeff₆={coeff6[canon]}")
        print(f"  count(τ,A)={ca}, count(τ,B)={cb}, diff={ca-cb}")
        print()