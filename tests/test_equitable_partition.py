"""Tests for WL-1 equitable partition."""
from grahamtools.wl.equitable_partition import equitable_partition_bitset, color_classes


def _edges_to_adj(edges, n):
    """Helper: convert edge list to bitset adjacency."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj


def test_equitable_partition_complete():
    # K_4: all vertices should have the same color
    adj = _edges_to_adj([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], 4)
    colors = equitable_partition_bitset(adj)
    assert len(set(colors)) == 1


def test_equitable_partition_path():
    # P4: 0-1-2-3
    # Degrees: 1, 2, 2, 1
    # After refinement: endpoints same color, middle vertices same color
    adj = _edges_to_adj([(0, 1), (1, 2), (2, 3)], 4)
    colors = equitable_partition_bitset(adj)
    assert colors[0] == colors[3]  # endpoints
    assert colors[1] == colors[2]  # middle
    assert colors[0] != colors[1]


def test_equitable_partition_star():
    # K_{1,3}: center has degree 3, leaves have degree 1
    adj = _edges_to_adj([(0, 1), (0, 2), (0, 3)], 4)
    colors = equitable_partition_bitset(adj)
    assert colors[1] == colors[2] == colors[3]
    assert colors[0] != colors[1]


def test_equitable_partition_fixpoint():
    # Ensure the algorithm reaches a fixpoint (converges)
    adj = _edges_to_adj([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], 5)
    colors = equitable_partition_bitset(adj)
    # C5 is vertex-transitive -> all same color
    assert len(set(colors)) == 1


def test_color_classes():
    colors = (0, 1, 0, 2, 1)
    classes = color_classes(colors)
    assert len(classes) == 3
    # Sorted by (len, vertices): singleton first, then pairs
    assert classes[0] == [3]       # color 2, length 1
    assert classes[1] == [0, 2]    # color 0, length 2
    assert classes[2] == [1, 4]    # color 1, length 2


def test_equitable_partition_initial_coloring():
    # Test with initial coloring
    adj = _edges_to_adj([(0, 1), (1, 2), (2, 3), (3, 0)], 4)
    # Without initial: C4 -> 1 color class (all degree 2)
    colors_no_init = equitable_partition_bitset(adj)
    assert len(set(colors_no_init)) == 1

    # With initial coloring breaking symmetry
    colors_with_init = equitable_partition_bitset(adj, initial=(0, 0, 1, 1))
    # Vertices 0,1 start as color 0, vertices 2,3 start as color 1
    # After refinement: still 2 classes (0,1) and (2,3)
    assert colors_with_init[0] == colors_with_init[1]
    assert colors_with_init[2] == colors_with_init[3]
