from __future__ import annotations

from typing import List, Sequence
import networkx as nx

from .adjlist import line_graph_adj


def iterated_line_graphs_adj(adj0: Sequence[Sequence[int]], k: int) -> List[List[List[int]]]:
    """
    Return adjlists [L^0(adj0), ..., L^k(adj0)].

    Note: L^0(adj0) is copied into list as a list-of-lists.
    """
    out: List[List[List[int]]] = [[list(neigh) for neigh in adj0]]
    cur = out[0]
    for _ in range(k):
        cur = line_graph_adj(cur)
        out.append(cur)
        if len(cur) == 0:
            break
    return out


def iterated_line_graphs_nx(G0: nx.Graph, k: int) -> List[nx.Graph]:
    """
    Return NetworkX graphs [L^0(G0), ..., L^k(G0)] using nx.line_graph.
    """
    out = [G0]
    cur = G0
    for _ in range(k):
        if cur.number_of_edges() == 0:
            break
        cur = nx.line_graph(cur)
        out.append(cur)
    return out
