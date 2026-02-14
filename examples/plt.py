import matplotlib.pyplot as plt
import networkx as nx
from grahamtools.io.graph6 import g6_to_nx
from grahamtools.viz.layouts import base_layout

g6 = "FlCKG"
g6 = "FxCGg"
G = g6_to_nx(g6)
pos = base_layout(G, seed=7)

nx.draw_networkx(G, pos=pos, with_labels=True)
plt.axis("off")
plt.show()
