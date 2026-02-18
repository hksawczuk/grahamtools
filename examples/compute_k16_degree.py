#!/usr/bin/env python3
"""
Compute d_0(K_{1,6}) without building L^5(K_7).

In K_7, the K_{1,5} fiber has C(7,1)*C(6,5) = 42 components.
Each component is isomorphic to the one we can compute in K_6.

Two components (same center, leaves differ by 1 element) have cross-edges:
element a in A is adjacent to element b in B iff they share a K_{1,4} parent
whose leaves are the 4-leaf intersection.

Plan:
1. Build L^5(K_6), extract K_{1,5} fiber (center=0, 480 elements)
2. For each fiber element, find its grade-4 parents
3. Identify which parents are K_{1,4} with which 4-leaf subset
4. Count cross-edges and compute L(cross) degree
"""

from collections import defaultdict
import time, math

def build_lg_with_parents(n, max_k):
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
    all_parents = [None]
    print(f'  L^1(K_{n}): {m}', flush=True)
    for k in range(2, max_k+1):
        t0 = time.time()
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
        all_parents.append(el)
        print(f'  L^{k}(K_{n}): {nm} ({time.time()-t0:.1f}s)', flush=True)
    return levels, edges, all_parents

def star_check(bs, e1):
    eds = [e1[i] for i in bs]
    vs = set()
    for u, v in eds: vs.add(u); vs.add(v)
    ne, nv = len(eds), len(vs)
    if ne != nv - 1: return None
    deg = defaultdict(int)
    for u, v in eds: deg[u] += 1; deg[v] += 1
    md = max(deg.values())
    if md == ne:
        c = [v for v, d in deg.items() if d == md][0]
        return (ne, c)
    return None

print('=== Computing d_0(K_{1,6}) via parent analysis ===\n')
levels, e1, all_parents = build_lg_with_parents(6, 5)

v5, a5, b5 = levels[4]
v4, a4, b4 = levels[3]
parents5 = all_parents[4]  # parents5[v] = (p1, p2) for grade-5 vertex v

# K_{1,5} fiber, center=0
all_leaves = frozenset({1, 2, 3, 4, 5})
fib5 = []
for v in v5:
    r = star_check(b5[v], e1)
    if r and r[0] == 5 and r[1] == 0:
        fib5.append(v)
print(f'\nK_{{1,5}} fiber (center=0): {len(fib5)} elements')

# K_{1,4} fiber, center=0, indexed by leaf set
fib4_by_leaves = defaultdict(list)
for v in v4:
    r = star_check(b4[v], e1)
    if r and r[0] == 4 and r[1] == 0:
        ls = set()
        for i in b4[v]: ls.add(e1[i][0]); ls.add(e1[i][1])
        ls.discard(0)
        fib4_by_leaves[frozenset(ls)].append(v)

print(f'K_{{1,4}} leaf sets (center=0): {len(fib4_by_leaves)}')
for ls in sorted(fib4_by_leaves, key=sorted):
    print(f'  leaves {sorted(ls)}: {len(fib4_by_leaves[ls])} elements')

# For each K_{1,5} element, find its K_{1,4} parents
# parent_map[missing_leaf][parent] = [list of K_{1,5} elements with this parent]
fib4_set = set()
for vs in fib4_by_leaves.values():
    fib4_set.update(vs)

parent_map = defaultdict(lambda: defaultdict(list))
element_cross_count = defaultdict(int)

for v in fib5:
    p1, p2 = parents5[v]
    for p in [p1, p2]:
        if p in fib4_set:
            leaves_p = None
            for ls, vs in fib4_by_leaves.items():
                if p in vs:
                    leaves_p = ls
                    break
            if leaves_p is not None:
                missing = all_leaves - leaves_p
                ml = next(iter(missing))
                parent_map[ml][p].append(v)

# Count cross-edges per element
# Through parent p (missing leaf ml), element a gets len(by_parent[p]) cross-edges
# (one to each mirror element on the other side)
for ml in sorted(parent_map):
    for p, vs in parent_map[ml].items():
        for v in vs:
            element_cross_count[v] += len(vs)

print(f'\nCross-edge structure per missing leaf:')
total_cross_from_component = 0
for ml in sorted(parent_map):
    parents = parent_map[ml]
    n_parents = len(parents)
    counts = sorted([len(vs) for vs in parents.values()])
    n_cross = sum(c*c for c in counts)
    total_cross_from_component += n_cross
    print(f'  ml={ml}: {n_parents} K_{{1,4}} parents, elements/parent: {counts}, cross-edges: {n_cross}')

print(f'Total cross-edges from this component: {total_cross_from_component}')

# Degree of L(cross-edges)
# Cross-edge (a in A, b in B_ml via parent p):
#   degree = (element_cross_count[a] - 1) + (element_cross_count[b_mirror] - 1)
# b_mirror has same cross-count as the corresponding element on our side.
# Through parent p with elements [v1,...,vk], b_idx j corresponds to vj.

all_degrees = []
for ml in sorted(parent_map):
    for p, vs in parent_map[ml].items():
        for a in vs:
            for b_mirror in vs:
                deg = (element_cross_count[a] - 1) + (element_cross_count[b_mirror] - 1)
                all_degrees.append(deg)

deg_dist = defaultdict(int)
for d in all_degrees:
    deg_dist[d] += 1

print(f'\n=== Result ===')
print(f'|Gamma_6(K_{{1,6}})| from this component: {len(all_degrees)}')
print(f'Degree distribution: {dict(sorted(deg_dist.items()))}')

if len(deg_dist) == 1:
    d0 = list(deg_dist.keys())[0]
    print(f'\n*** d_0(K_{{1,6}}) = {d0} ***')
else:
    print(f'\nNot regular from this component\'s perspective.')
    print(f'Min degree: {min(deg_dist)}, Max degree: {max(deg_dist)}')

print(f'\nSummary & pattern check (d_0 = 2^(r-1) - 2?):')
known = [(3, 2), (4, 6), (5, 14)]
d6 = list(deg_dist.keys())[0] if len(deg_dist) == 1 else max(deg_dist)
known.append((6, d6))
for r, d in known:
    pred = 2**(r-1) - 2
    sl = math.log2(d - 2) if d > 2 else 0
    print(f'  K_{{1,{r}}}: d_0={d:4d}, predicted={pred:4d}, match={pred==d}, sub-leading=j*{sl:.4f}')