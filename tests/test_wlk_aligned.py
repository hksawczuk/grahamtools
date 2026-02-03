from grahamtools.io.graph6 import g6_to_nx
from grahamtools.wl.wlk_aligned import wlk_distinguishable_aligned

def test_wlk_aligned_runs():
    GA = g6_to_nx("EEj_")
    GB = g6_to_nx("EQjO")
    dist, result = wlk_distinguishable_aligned(GA, GB, k=2, max_iter=10)
    assert isinstance(dist, bool)
    assert result.k == 2
    assert result.rounds >= 1
