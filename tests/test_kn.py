"""Tests for grahamtools.kn module."""
from grahamtools.kn.levels import generate_levels_Kn_ids, canon_pair_int
from grahamtools.kn.expand import expand_to_simple_base_edges_id, expand_to_base_edge_multiset_id
from grahamtools.kn.classify import canon_key


def test_canon_pair_int():
    assert canon_pair_int(3, 1) == (1, 3)
    assert canon_pair_int(2, 2) == (2, 2)
    assert canon_pair_int(0, 5) == (0, 5)


def test_generate_levels_k4_level0():
    V, ep = generate_levels_Kn_ids(4, 0)
    assert V[0] == [0, 1, 2, 3]
    assert ep == {}


def test_generate_levels_k4_level1():
    V, ep = generate_levels_Kn_ids(4, 1)
    # K_4 has C(4,2) = 6 edges
    assert len(V[1]) == 6
    assert len(ep[1]) == 6
    # All endpoints should be pairs from {0,1,2,3}
    for a, b in ep[1]:
        assert 0 <= a < b < 4


def test_generate_levels_k4_level2():
    V, ep = generate_levels_Kn_ids(4, 2)
    # L(K_4) has |V| = 6, and L(K_4) is the octahedral graph with 12 edges
    # So level 2 should have 12 vertices
    assert len(V[2]) == 12


def test_generate_levels_k5_prune_cycles():
    V, ep = generate_levels_Kn_ids(5, 3, prune_cycles=True)
    # With pruning, only tree-type vertices survive
    assert len(V[1]) == 10  # all edges of K_5 are single edges (trees)
    # Level 2: L(K_5) vertices whose base edge set is a forest (path P3)
    # Each level-2 vertex is a pair of adjacent edges = path of length 2
    assert len(V[2]) > 0
    assert len(V[3]) > 0


def test_expand_simple_level1():
    V, ep = generate_levels_Kn_ids(4, 1)
    for v in V[1]:
        edges = expand_to_simple_base_edges_id(v, 1, ep)
        assert len(edges) == 1
        a, b = edges[0]
        assert a < b


def test_expand_simple_level2():
    V, ep = generate_levels_Kn_ids(4, 2)
    for v in V[2]:
        edges = expand_to_simple_base_edges_id(v, 2, ep)
        # Level 2: each vertex is a pair of incident edges = 2 base edges (path P3)
        assert len(edges) == 2


def test_expand_multiset_level2():
    V, ep = generate_levels_Kn_ids(4, 2)
    for v in V[2]:
        ms = expand_to_base_edge_multiset_id(v, 2, ep)
        # At level 2, the multiset should have 2 edges (shared endpoint counted once each)
        assert sum(ms.values()) == 2


def test_canon_key_consistent():
    # Two isomorphic base-edge sets should have the same canonical key
    edges_a = [(0, 1), (1, 2)]  # P3 on vertices 0,1,2
    edges_b = [(2, 3), (3, 4)]  # P3 on vertices 2,3,4
    assert canon_key(edges_a, 5) == canon_key(edges_b, 5)


def test_canon_key_different():
    edges_p3 = [(0, 1), (1, 2)]
    edges_k2 = [(0, 1)]
    assert canon_key(edges_p3, 3) != canon_key(edges_k2, 3)
