from __future__ import annotations

from typing import List
import networkx as nx


def strip_graph6_header(g6: str) -> str:
    """
    Remove optional '>>graph6<<' header and whitespace.
    """
    s = g6.strip()
    if s.startswith(">>graph6<<"):
        s = s[len(">>graph6<<") :].strip()
    return s


def g6_to_nx(g6: str) -> nx.Graph:
    """
    Parse a graph6 string into a simple undirected NetworkX Graph.
    """
    s = strip_graph6_header(g6)
    G = nx.from_graph6_bytes(s.encode("ascii"))
    # graph6 is simple by design, but guard anyway
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        G = nx.Graph(G)
    return G


def g6_to_adjlist(g6: str) -> List[List[int]]:
    """
    Parse a graph6 string into a 0..n-1 adjacency list.

    Returns:
      adj[u] = sorted list of neighbors of u
    """
    G = g6_to_nx(g6)
    n = G.number_of_nodes()
    adj = [[] for _ in range(n)]
    for u in range(n):
        adj[u] = sorted(G.neighbors(u))
    return adj
