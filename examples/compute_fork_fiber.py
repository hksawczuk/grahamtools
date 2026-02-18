#!/usr/bin/env python3
"""
Compute the fiber graph of the "fork" tree in L^4(K_5).

The fork = unique tree on 5 vertices with degree sequence (2,2,1,1,1).
It's a path of length 2 with one extra leaf at an interior vertex:

    a - b - c - d
        |
        e

Equivalently: vertex b has degree 3? No, that's K_{1,3} plus an edge.
Wait, degree sequence (2,2,1,1,1) on 5 vertices, 4 edges.

Let me enumerate: trees on 5 vertices with 4 edges:
- P_5: path, degree seq (1,1,2,2,2) -- no wait, P_5 has deg seq (1,2,2,2,1)
  Actually P_5: 1-2-3-4-5, degrees: 1,2,2,2,1
- K_{1,4}: star, degree seq (4,1,1,1,1)  
- Fork/caterpillar: 
    1-2-3-4
      |
      5
  degrees: 1,3,1,1,1 -- that's the star K_{1,3} with one edge extended. 
  Wait no: vertex 2 has degree 3 (connected to 1,3,5), vertex 3 has degree 2 (connected to 2,4).
  Degree seq: (1,3,2,1,1) = sorted: (1,1,1,2,3)

Actually there are 3 trees on 5 vertices:
1. P_5: degrees (1,2,2,2,1)
2. K_{1,4}: degrees (4,1,1,1,1)
3. The "fork" or "Y-tree": one vertex of degree 3, one of degree 2, three of degree 1
   Shape: 1-2-3, 2-4, 3-5 => degrees (1,2,2,1,1)? No...
   
   Let me just list them:
   a) 1-2, 2-3, 3-4, 4-5 => P_5, degrees 1,2,2,2,1
   b) 1-2, 1-3, 1-4, 1-5 => K_{1,4}, degrees 4,1,1,1,1
   c) 1-2, 2-3, 2-4, 3-5 => degrees: 1 has deg 1, 2 has deg 3, 3 has deg 2, 4 has deg 1, 5 has deg 1
      sorted: (1,1,1,2,3)

So the fork has degree sequence (1,1,1,2,3). Let me verify this is the only other tree.
Trees on 5 vertices: there are exactly 3 (P_5, fork, K_{1,4}).

The fork: one vertex of degree 3 adjacent to one vertex of degree 2 and two leaves.
The degree-2 vertex has one more leaf.
"""

from collections import defaultdict
import time

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
        print(f'  L^{k}(K_{n}): {nm} ({time.time()-t0:.1f}s)', flush=True)
    return levels, edges

def classify_tree(bs, e1):
    """Classify the tree type of a base edge set. Returns type name and canonical info."""
    edges = [e1[i] for i in bs]
    verts = set()
    for u, v in edges: verts.add(u); verts.add(v)
    ne, nv = len(edges), len(verts)
    
    if ne != nv - 1:
        return None, None  # not a tree
    
    # Check connectivity
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)
    
    visited = set()
    stack = [next(iter(verts))]
    while stack:
        v = stack.pop()
        if v in visited: continue
        visited.add(v)
        for u in adj[v]: stack.append(u)
    if visited != verts:
        return None, None  # disconnected
    
    deg = defaultdict(int)
    for u, v in edges: deg[u] += 1; deg[v] += 1
    deg_seq = tuple(sorted(deg.values()))
    
    if ne == 4:
        if deg_seq == (1, 1, 1, 1, 4):
            center = [v for v in verts if deg[v] == 4][0]
            return "K_{1,4}", {"center": center}
        elif deg_seq == (1, 1, 1, 2, 3):
            # Fork: vertex of degree 3 and vertex of degree 2
            v3 = [v for v in verts if deg[v] == 3][0]
            v2 = [v for v in verts if deg[v] == 2][0]
            return "fork", {"deg3": v3, "deg2": v2}
        elif deg_seq == (1, 1, 2, 2, 2):
            return "P_5", {}
        else:
            return f"tree_4e_{deg_seq}", {}
    elif ne == 3:
        if deg_seq == (1, 1, 1, 3):
            center = [v for v in verts if deg[v] == 3][0]
            return "K_{1,3}", {"center": center}
        elif deg_seq == (1, 1, 2, 2):
            return "P_4", {}
        else:
            return f"tree_3e_{deg_seq}", {}
    elif ne == 2:
        if deg_seq == (1, 1, 2):
            return "P_3", {}
        elif deg_seq == (1, 1, 1, 0):  # shouldn't happen
            pass
        return f"tree_2e_{deg_seq}", {}
    else:
        return f"tree_{ne}e_{deg_seq}", {}


print('=== Fork fiber graph in L^4(K_5) ===\n')
levels, e1 = build_lg_with_parents(5, 4)

v4, a4, b4 = levels[3]
print(f'\nGrade 4: {len(v4)} vertices')

# Classify all grade-4 elements
type_counts = defaultdict(int)
fork_elements = []
fork_info = {}

for v in v4:
    typ, info = classify_tree(b4[v], e1)
    if typ:
        type_counts[typ] += 1
    if typ == "fork":
        fork_elements.append(v)
        fork_info[v] = info

print(f'\nType distribution at grade 4:')
for t, c in sorted(type_counts.items()):
    print(f'  {t}: {c}')

print(f'\nFork fiber: {len(fork_elements)} elements')
print(f'coeff_4(fork) = {len(fork_elements)} (in K_5, divide by nothing since we count all)')

# Group by labeled support
by_support = defaultdict(list)
for v in fork_elements:
    # Support = set of base edges
    by_support[b4[v]].append(v)

print(f'Distinct fork supports: {len(by_support)}')
for sup, vs in sorted(by_support.items(), key=lambda x: sorted(x[0])):
    edges_labeled = [e1[i] for i in sorted(sup)]
    print(f'  {edges_labeled}: {len(vs)} elements')

# Build fiber graph
fork_set = set(fork_elements)
fork_adj = defaultdict(set)
for v in fork_elements:
    for u in a4[v]:
        if u in fork_set:
            fork_adj[v].add(u)

degrees = [len(fork_adj[v]) for v in fork_elements]
deg_dist = defaultdict(int)
for d in degrees: deg_dist[d] += 1

n_edges = sum(degrees) // 2

print(f'\nFork fiber graph:')
print(f'  Vertices: {len(fork_elements)}')
print(f'  Edges: {n_edges}')
print(f'  Degree distribution: {dict(sorted(deg_dist.items()))}')

# Check if regular
if len(deg_dist) == 1:
    print(f'  Regular: {list(deg_dist.keys())[0]}')
else:
    print(f'  Not regular!')

# Check decomposition by support
cross_support = 0
intra_support = 0
for v in fork_elements:
    for u in fork_adj[v]:
        if b4[v] == b4[u]:
            intra_support += 1
        else:
            cross_support += 1

print(f'\n  Intra-support edges: {intra_support // 2}')
print(f'  Cross-support edges: {cross_support // 2}')

# Analyze cross-support edges: what's the union type?
cross_union_types = defaultdict(int)
for v in fork_elements:
    for u in fork_adj[v]:
        if u > v and b4[v] != b4[u]:
            union_base = b4[v] | b4[u]
            typ, info = classify_tree(union_base, e1)
            if typ:
                cross_union_types[typ] += 1
            else:
                # Not a tree
                edges_u = [e1[i] for i in union_base]
                ne = len(union_base)
                vs = set()
                for a, b in edges_u: vs.add(a); vs.add(b)
                cross_union_types[f"non-tree_{len(vs)}v_{ne}e"] += 1

print(f'\n  Cross-support union types:')
for t, c in sorted(cross_union_types.items()):
    print(f'    {t}: {c}')

# Check connectivity of the fiber graph
visited = set()
components = []
for start in fork_elements:
    if start in visited: continue
    comp = set()
    stack = [start]
    while stack:
        v = stack.pop()
        if v in comp: continue
        comp.add(v)
        for u in fork_adj[v]: stack.append(u)
    visited |= comp
    components.append(comp)

print(f'\n  Connected components: {len(components)}')
comp_sizes = sorted([len(c) for c in components])
print(f'  Component sizes: {comp_sizes}')

# Properties per component
for ci, comp in enumerate(components):
    comp_list = sorted(comp)
    comp_degs = [len(fork_adj[v] & comp) for v in comp_list]
    comp_deg_dist = defaultdict(int)
    for d in comp_degs: comp_deg_dist[d] += 1
    comp_edges = sum(comp_degs) // 2
    
    # Get support of elements in this component
    supports_in_comp = set()
    for v in comp_list:
        supports_in_comp.add(b4[v])
    
    print(f'\n  Component {ci}: {len(comp)} vertices, {comp_edges} edges')
    print(f'    Degrees: {dict(sorted(comp_deg_dist.items()))}')
    print(f'    Supports: {len(supports_in_comp)}')

# Girth, diameter, triangles for full fiber graph
print(f'\nGraph properties:')

# Bipartite check
color = {}
is_bip = True
for start in fork_elements:
    si = start
    if si in color: continue
    stack = [(si, 0)]
    while stack:
        v, c = stack.pop()
        if v in color:
            if color[v] != c: is_bip = False
            continue
        color[v] = c
        for u in fork_adj[v]:
            if u not in color: stack.append((u, 1 - c))
print(f'  Bipartite: {is_bip}')

# Girth
min_girth = float('inf')
for sv in fork_elements[:50]:  # sample
    dist = {sv: 0}
    parent = {sv: -1}
    queue = [sv]
    qi = 0
    while qi < len(queue):
        vi = queue[qi]; qi += 1
        for u in fork_adj[vi]:
            if u not in dist:
                dist[u] = dist[vi] + 1
                parent[u] = vi
                queue.append(u)
            elif parent[vi] != u and parent[u] != vi:
                min_girth = min(min_girth, dist[vi] + dist[u] + 1)
print(f'  Girth: {min_girth}')

# Diameter (sample)
max_dist = 0
for sv in fork_elements[:50]:
    dist = {sv: 0}
    queue = [sv]
    qi = 0
    while qi < len(queue):
        vi = queue[qi]; qi += 1
        for u in fork_adj[vi]:
            if u not in dist:
                dist[u] = dist[vi] + 1
                queue.append(u)
    max_dist = max(max_dist, max(dist.values()))
print(f'  Diameter (sampled): {max_dist}')

# Triangle count
n_tri = 0
fork_list = sorted(fork_elements)
fork_idx = {v: i for i, v in enumerate(fork_list)}
for v in fork_list:
    for u in fork_adj[v]:
        if u > v:
            for w in fork_adj[u]:
                if w > u and w in fork_adj[v]:
                    n_tri += 1
print(f'  Triangles: {n_tri}')