#!/usr/bin/env python3
"""Check whether the non-regularity comes from cross-support edges."""

from collections import defaultdict

def build_lg(n, max_k):
    edges = [(i,j) for i in range(n) for j in range(i+1,n)]
    m = len(edges)
    base = [frozenset({i}) for i in range(m)]
    adj = defaultdict(set)
    for i in range(m):
        for j in range(i+1,m):
            if edges[i][0] in edges[j] or edges[i][1] in edges[j]:
                adj[i].add(j); adj[j].add(i)
    cn, ca, cb = m, adj, base
    levels = [(list(range(m)), adj, base)]
    for k in range(2, max_k+1):
        el = []
        for v in range(cn):
            for u in ca[v]:
                if u > v: el.append((v, u))
        nm = len(el)
        nb = [cb[v] | cb[u] for v, u in el]
        inc = defaultdict(list)
        for i, (v, u) in enumerate(el): inc[v].append(i); inc[u].append(i)
        na = defaultdict(set)
        for vp, il in inc.items():
            for i in range(len(il)):
                for j in range(i+1, len(il)):
                    na[il[i]].add(il[j]); na[il[j]].add(il[i])
        cn, ca, cb = nm, na, nb
        levels.append((list(range(nm)), na, nb))
    return levels, edges

def classify_tree(bs, e1):
    edges = [e1[i] for i in bs]
    verts = set()
    for u, v in edges: verts.add(u); verts.add(v)
    ne, nv = len(edges), len(verts)
    if ne != nv - 1: return None
    adj = defaultdict(set)
    for u, v in edges: adj[u].add(v); adj[v].add(u)
    visited = set()
    stack = [next(iter(verts))]
    while stack:
        v = stack.pop()
        if v in visited: continue
        visited.add(v)
        for u in adj[v]: stack.append(u)
    if visited != verts: return None
    deg = defaultdict(int)
    for u, v in edges: deg[u] += 1; deg[v] += 1
    deg_seq = tuple(sorted(deg.values()))
    if deg_seq == (1, 1, 1, 2, 3): return "fork"
    if deg_seq == (1, 1, 1, 1, 4): return "K_{1,4}"
    if deg_seq == (1, 1, 2, 2, 2): return "P_5"
    return f"tree_{deg_seq}"

levels, e1 = build_lg(5, 4)
v4, a4, b4 = levels[3]

fork_elements = []
for v in v4:
    if classify_tree(b4[v], e1) == "fork":
        fork_elements.append(v)

fork_set = set(fork_elements)
fork_adj = defaultdict(set)
for v in fork_elements:
    for u in a4[v]:
        if u in fork_set:
            fork_adj[v].add(u)

# For each element, split degree into intra-support and cross-support
print("degree  intra  cross  | count")
combos = defaultdict(int)
for v in fork_elements:
    intra = sum(1 for u in fork_adj[v] if b4[u] == b4[v])
    cross = sum(1 for u in fork_adj[v] if b4[u] != b4[v])
    total = intra + cross
    combos[(total, intra, cross)] += 1

for (t, i, c), n in sorted(combos.items()):
    print(f"  {t:3d}    {i:3d}    {c:3d}    | {n}")