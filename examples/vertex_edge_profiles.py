"""
Vertex Edge-Type Profiles on Iterated Line Graphs of T_{2,5}.

For the complete binary tree T_{2,5} (depth 5), compute iterated line graphs
L^0 through L^5. At each iteration, compute each vertex's "local edge-type
profile" -- the sorted tuple of edge types incident to that vertex, where each
edge type is the sorted degree pair (deg(u), deg(v)) of the edge's endpoints.

Group vertices by profile and report statistics.
"""

import sys
from collections import Counter

sys.path.insert(0, "/Users/hamiltonsawczuk/grahamtools/.claude/worktrees/amazing-kilby/src")

import networkx as nx
from grahamtools.linegraph.adjlist import line_graph_adj, edges_from_adj


def nx_to_adj(G):
    """Convert a NetworkX graph to a sorted adjacency list (list of lists)."""
    n = G.number_of_nodes()
    # Ensure nodes are labeled 0..n-1
    mapping = {node: i for i, node in enumerate(sorted(G.nodes()))}
    adj = [[] for _ in range(n)]
    for u, v in G.edges():
        adj[mapping[u]].append(mapping[v])
        adj[mapping[v]].append(mapping[u])
    # Sort neighbor lists for consistency
    for i in range(n):
        adj[i].sort()
    return adj


def compute_profiles(adj):
    """
    For each vertex v, compute its local edge-type profile.

    The profile is the sorted tuple of edge types incident to v,
    where each edge type is (min(deg(v), deg(u)), max(deg(v), deg(u)))
    for each neighbor u of v.
    """
    n = len(adj)
    degrees = [len(adj[v]) for v in range(n)]

    profiles = {}
    for v in range(n):
        dv = degrees[v]
        edge_types = []
        for u in adj[v]:
            du = degrees[u]
            edge_type = (min(dv, du), max(dv, du))
            edge_types.append(edge_type)
        edge_types.sort()
        profiles[v] = tuple(edge_types)

    return profiles


def profile_summary(profiles):
    """Group vertices by profile and return a Counter of profile -> count."""
    return Counter(profiles.values())


def format_profile(prof):
    """Format a profile tuple for display."""
    if len(prof) <= 12:
        return str(prof)
    # Summarize long profiles: show edge-type counts
    edge_counts = Counter(prof)
    parts = []
    for etype, count in sorted(edge_counts.items()):
        parts.append(f"{etype}x{count}")
    return "{" + ", ".join(parts) + "}"


def main():
    # Build the complete binary tree T_{2,5}
    depth = 5
    T = nx.balanced_tree(r=2, h=depth)
    adj = nx_to_adj(T)

    print("=" * 72)
    print("Vertex Edge-Type Profiles on Iterated Line Graphs of T_{2,5}")
    print("=" * 72)
    print(f"\nBase tree: T_{{2,5}} (complete binary tree, depth {depth})")
    print(f"  Vertices: {len(adj)}")
    print(f"  Edges:    {len(edges_from_adj(adj))}")

    current_adj = adj

    for iteration in range(6):
        n = len(current_adj)
        m = len(edges_from_adj(current_adj))

        print(f"\n{'=' * 72}")
        print(f"L^{iteration}:  |V| = {n},  |E| = {m}")
        print("=" * 72)

        profiles = compute_profiles(current_adj)
        counts = profile_summary(profiles)

        num_distinct = len(counts)
        print(f"Distinct profiles: {num_distinct}")

        # Sort profiles by count (descending), then by profile tuple
        sorted_profiles = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

        # Identify the bulk profile
        bulk_profile, bulk_count = sorted_profiles[0]
        bulk_fraction = bulk_count / n

        if iteration <= 3:
            # Print ALL profiles for small graphs
            print(f"\nAll profiles (sorted by count, descending):")
            print(f"{'Count':>8}  {'Fraction':>10}  Profile")
            print(f"{'-'*8}  {'-'*10}  {'-'*40}")
            for prof, cnt in sorted_profiles:
                frac = cnt / n
                print(f"{cnt:>8}  {frac:>10.4f}  {format_profile(prof)}")
        else:
            # Summary for large graphs
            print(f"\nTop 5 most common profiles:")
            print(f"{'Count':>8}  {'Fraction':>10}  Profile")
            print(f"{'-'*8}  {'-'*10}  {'-'*40}")
            for prof, cnt in sorted_profiles[:5]:
                frac = cnt / n
                print(f"{cnt:>8}  {frac:>10.4f}  {format_profile(prof)}")
            if num_distinct > 5:
                remaining = sum(c for _, c in sorted_profiles[5:])
                print(f"  ... and {num_distinct - 5} more profiles "
                      f"covering {remaining} vertices")

        print(f"\nBulk profile: {format_profile(bulk_profile)}")
        print(f"  Vertices in bulk: {bulk_count} / {n} = {bulk_fraction:.4f}")
        has_dominant = bulk_fraction > 0.5
        print(f"  Dominant (>50%)? {'YES' if has_dominant else 'NO'}")

        # Compute next line graph (unless this is the last iteration)
        if iteration < 5:
            current_adj = line_graph_adj(current_adj)


if __name__ == "__main__":
    main()
