from dataclasses import dataclass
from typing import Optional, List

import networkx as nx

from grahamtools.io.graph6 import g6_to_nx
from grahamtools.linegraph.iterate import iterated_line_graphs_nx


@dataclass
class GrahamOptions:
    k_max: int = 10
    n_cap: Optional[int] = None
    m_cap: Optional[int] = None


def graham_sequence_from_graph(G: nx.Graph, opts: GrahamOptions) -> List[int]:
    seq: List[int] = []
    H = G
    for _k in range(opts.k_max + 1):
        n = H.number_of_nodes()
        m = H.number_of_edges()
        seq.append(n)

        if opts.n_cap is not None and n > opts.n_cap:
            break
        if opts.m_cap is not None and m > opts.m_cap:
            break
        if m == 0:
            break

        H = nx.line_graph(H)
    return seq


if __name__ == "__main__":
    g6 = "DsC"  # replace
    opts = GrahamOptions(k_max=9, n_cap=2_000_000, m_cap=5_000_000)

    G = g6_to_nx(g6)
    seq = graham_sequence_from_graph(G, opts)
    print("g6:", g6)
    print("Graham sequence:", seq)

    # Quick sanity: package function (adjlist-based) for the first few
    from grahamtools.invariants.graham import graham_sequence_g6
    print("Package prefix:", graham_sequence_g6(g6, 6))
