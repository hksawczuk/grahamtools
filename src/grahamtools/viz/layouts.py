from __future__ import annotations

import networkx as nx


def base_layout(G: nx.Graph, seed: int = 7):
    """
    Choose a reasonable base layout:
      - planar_layout if planar and succeeds
      - otherwise spring_layout
    """
    try:
        is_planar, _ = nx.check_planarity(G)
        if is_planar:
            return nx.planar_layout(G)
    except Exception:
        pass
    return nx.spring_layout(G, seed=seed, iterations=300)


def layout_line_graph_from_prev(
    H_prev: nx.Graph,
    pos_prev: dict,
    H_line: nx.Graph,
    seed: int = 7,
    iterations: int = 150,
):
    """
    Use midpoint positions from H_prev as initial positions for L(H_prev),
    then refine with a few spring iterations.

    NetworkX line_graph nodes are edges of H_prev (2-tuples).
    """
    init = {}
    for e in H_line.nodes():
        u, v = e
        if u in pos_prev and v in pos_prev:
            init[e] = 0.5 * (pos_prev[u] + pos_prev[v])
        else:
            init[e] = (0.0, 0.0)

    pos = nx.spring_layout(H_line, seed=seed, pos=init, iterations=iterations)
    return pos


def iterated_line_graphs_with_layout(G0: nx.Graph, k: int, seed: int = 7):
    """
    Returns:
      graphs[i] = L^i(G0)
      pos[i]    = layout for graphs[i], inherited from previous iterate
    """
    graphs = [G0]
    pos = [base_layout(G0, seed=seed)]

    H = G0
    for _i in range(1, k + 1):
        if H.number_of_edges() == 0:
            break
        H_next = nx.line_graph(H)
        pos_next = layout_line_graph_from_prev(H, pos[-1], H_next, seed=seed)
        graphs.append(H_next)
        pos.append(pos_next)
        H = H_next

    return graphs, pos
