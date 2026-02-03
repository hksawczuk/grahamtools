from grahamtools.io.graph6 import g6_to_nx
from grahamtools.wl.wlk_aligned import wlk_distinguishable_aligned

# Your graphs (canonical graph6)
g6A = "F`AZO"
g6B = "F`BHo"

GA = g6_to_nx(g6A)
GB = g6_to_nx(g6B)

dist, result = wlk_distinguishable_aligned(GA, GB, k=3, max_iter=50)
print("WL-3 distinguishable?", dist)
print("Rounds:", result.rounds)
print("Witness color:", result.witness_color)
print("Hist A:", result.histA)
print("Hist B:", result.histB)
