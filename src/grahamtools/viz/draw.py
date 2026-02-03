from __future__ import annotations

import networkx as nx
import matplotlib.pyplot as plt

from grahamtools.io.graph6 import g6_to_nx
from .layouts import iterated_line_graphs_with_layout


def draw_iterate_pair(
    g6A: str,
    g6B: str,
    *,
    k: int,
    seed: int = 7,
    node_size: int = 140,
    edge_width: float = 1.2,
    max_nodes_to_draw: int = 600,
    save_prefix: str | None = None,
):
    """
    Draw side-by-side the iterates L^i(A) and L^i(B) for i=0..k.
    Uses inherited layouts for visual stability.

    If save_prefix is set, saves PNG files:
      {save_prefix}_L0.png, ..., {save_prefix}_Lk.png
    """
    GA0 = g6_to_nx(g6A)
    GB0 = g6_to_nx(g6B)

    A_graphs, A_pos = iterated_line_graphs_with_layout(GA0, k, seed=seed)
    B_graphs, B_pos = iterated_line_graphs_with_layout(GB0, k, seed=seed)

    countsA = [H.number_of_nodes() for H in A_graphs]
    countsB = [H.number_of_nodes() for H in B_graphs]

    # Check agreement up to available iterates
    t = min(len(countsA), len(countsB), k + 1)
    for i in range(t):
        if countsA[i] != countsB[i]:
            print(f"Mismatch at i={i}: |V(L^{i}(A))|={countsA[i]} vs |V(L^{i}(B))|={countsB[i]}")
            break
    else:
        print(f"Verified: |V(L^i(A))|=|V(L^i(B))| for i=0..{t-1}")
        print("Counts:", countsA[:t])

    for i in range(0, min(k + 1, len(A_graphs), len(B_graphs))):
        HA, HB = A_graphs[i], B_graphs[i]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axA, axB = axes

        axA.set_title(f"A: L^{i}   |V|={HA.number_of_nodes()}  |E|={HA.number_of_edges()}")
        axB.set_title(f"B: L^{i}   |V|={HB.number_of_nodes()}  |E|={HB.number_of_edges()}")

        for ax in axes:
            ax.set_axis_off()

        if HA.number_of_nodes() <= max_nodes_to_draw:
            nx.draw_networkx(
                HA,
                pos=A_pos[i],
                ax=axA,
                with_labels=False,
                node_size=node_size,
                width=edge_width,
            )
        else:
            axA.text(
                0.5,
                0.5,
                f"Too large to draw\n(|V|={HA.number_of_nodes()})",
                ha="center",
                va="center",
                transform=axA.transAxes,
            )

        if HB.number_of_nodes() <= max_nodes_to_draw:
            nx.draw_networkx(
                HB,
                pos=B_pos[i],
                ax=axB,
                with_labels=False,
                node_size=node_size,
                width=edge_width,
            )
        else:
            axB.text(
                0.5,
                0.5,
                f"Too large to draw\n(|V|={HB.number_of_nodes()})",
                ha="center",
                va="center",
                transform=axB.transAxes,
            )

        plt.tight_layout()

        if save_prefix:
            plt.savefig(f"{save_prefix}_L{i}.png", dpi=200)
            plt.close(fig)
        else:
            plt.show()

    return countsA, countsB
