"""Tests for expanded nauty integration."""
import pytest

from grahamtools.external.nauty import (
    nauty_available,
    dreadnaut_available,
    edgelist_to_g6,
    canon_g6,
    aut_size_g6,
    canon_label_g6,
    geng_g6,
    geng_connected_subcubic_g6,
)


# --- edgelist_to_g6 (always works, uses networkx) ---

def test_edgelist_to_g6_triangle():
    g6 = edgelist_to_g6([(0, 1), (1, 2), (0, 2)], 3)
    assert isinstance(g6, str)
    assert len(g6) > 0


def test_edgelist_to_g6_empty():
    g6 = edgelist_to_g6([], 3)
    assert isinstance(g6, str)


def test_edgelist_to_g6_k4():
    edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    g6 = edgelist_to_g6(edges, 4)
    assert isinstance(g6, str)


# --- canon_g6 (requires nauty) ---

@pytest.mark.skipif(not nauty_available(), reason="nauty not available")
def test_canon_g6_triangle():
    g6_a = edgelist_to_g6([(0, 1), (1, 2), (0, 2)], 3)
    g6_b = edgelist_to_g6([(0, 2), (1, 2), (0, 1)], 3)
    assert canon_g6(g6_a) == canon_g6(g6_b)


@pytest.mark.skipif(not nauty_available(), reason="nauty not available")
def test_canon_g6_different_graphs():
    # P3 vs K3
    p3 = edgelist_to_g6([(0, 1), (1, 2)], 3)
    k3 = edgelist_to_g6([(0, 1), (1, 2), (0, 2)], 3)
    assert canon_g6(p3) != canon_g6(k3)


# --- aut_size_g6 (requires dreadnaut) ---

@pytest.mark.skipif(not dreadnaut_available(), reason="dreadnaut not available")
def test_aut_size_triangle():
    g6 = edgelist_to_g6([(0, 1), (1, 2), (0, 2)], 3)
    assert aut_size_g6(g6) == 6  # |Aut(K3)| = 3! = 6


@pytest.mark.skipif(not dreadnaut_available(), reason="dreadnaut not available")
def test_aut_size_k4():
    edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    g6 = edgelist_to_g6(edges, 4)
    assert aut_size_g6(g6) == 24  # |Aut(K4)| = 4! = 24


@pytest.mark.skipif(not dreadnaut_available(), reason="dreadnaut not available")
def test_aut_size_p4():
    g6 = edgelist_to_g6([(0, 1), (1, 2), (2, 3)], 4)
    assert aut_size_g6(g6) == 2  # P4 has one non-trivial aut (reflection)


# --- canon_label_g6 ---

@pytest.mark.skipif(
    not (nauty_available() and dreadnaut_available()),
    reason="nauty or dreadnaut not available",
)
def test_canon_label_g6_triangle():
    g6 = edgelist_to_g6([(0, 1), (1, 2), (0, 2)], 3)
    canon, aut = canon_label_g6(g6)
    assert isinstance(canon, str)
    assert aut == 6


# --- geng_g6 ---

@pytest.mark.skipif(not nauty_available(), reason="nauty not available")
def test_geng_connected_graphs_on_4():
    graphs = list(geng_g6(4, connected=True))
    assert len(graphs) == 6  # 6 connected graphs on 4 vertices


@pytest.mark.skipif(not nauty_available(), reason="nauty not available")
def test_geng_connected_subcubic_4():
    graphs = list(geng_connected_subcubic_g6(4))
    # Connected graphs on 4 vertices with max degree <= 3
    assert len(graphs) > 0
    assert all(isinstance(g, str) for g in graphs)


@pytest.mark.skipif(not nauty_available(), reason="nauty not available")
def test_geng_triangle_free():
    graphs = list(geng_g6(4, connected=True, triangle_free=True))
    # K4 is excluded, so fewer graphs
    all_connected = list(geng_g6(4, connected=True))
    assert len(graphs) < len(all_connected)
