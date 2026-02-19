"""Tests for grahamtools.utils module."""
import math
from fractions import Fraction

from grahamtools.utils.connectivity import is_connected_edges, connected_components_edges
from grahamtools.utils.linegraph_edgelist import line_graph_edgelist, gamma_sequence_edgelist
from grahamtools.utils.canonical import canonical_graph_bruteforce
from grahamtools.utils.subgraphs import enumerate_connected_subgraphs
from grahamtools.utils.naming import tree_name, describe_graph
from grahamtools.utils.linalg import row_reduce_fraction, exact_rank
from grahamtools.utils.automorphisms import aut_size_edges, orbit_size_under_Sn


# --- connectivity ---

def test_is_connected_empty():
    assert is_connected_edges([]) is True


def test_is_connected_single_vertex():
    assert is_connected_edges([], vertices={0}) is True


def test_is_connected_two_isolated():
    assert is_connected_edges([], vertices={0, 1}) is False


def test_is_connected_triangle():
    assert is_connected_edges([(0, 1), (1, 2), (0, 2)]) is True


def test_is_connected_disconnected():
    assert is_connected_edges([(0, 1), (2, 3)]) is False


def test_connected_components_triangle():
    comps = connected_components_edges([(0, 1), (1, 2), (0, 2)])
    assert len(comps) == 1
    assert comps[0][0] == {0, 1, 2}


def test_connected_components_two():
    comps = connected_components_edges([(0, 1), (2, 3)])
    assert len(comps) == 2


# --- line graph (edgelist) ---

def test_line_graph_triangle():
    edges = [(0, 1), (1, 2), (0, 2)]
    new_edges, n = line_graph_edgelist(edges)
    assert n == 3  # 3 edges -> 3 vertices
    assert len(new_edges) == 3  # L(C3) = C3


def test_line_graph_star():
    # K_{1,3} has 3 edges; L(K_{1,3}) = K_3
    edges = [(0, 1), (0, 2), (0, 3)]
    new_edges, n = line_graph_edgelist(edges)
    assert n == 3
    assert len(new_edges) == 3


def test_line_graph_empty():
    new_edges, n = line_graph_edgelist([])
    assert n == 0
    assert new_edges == []


def test_gamma_sequence_p3():
    # P3: 3 vertices, 2 edges, L(P3) = K2 (1 edge, 2 vertices), L(K2) = K1 (0)
    edges = [(0, 1), (1, 2)]
    seq = gamma_sequence_edgelist(edges, 3)
    assert seq[0] == 3  # |V(P3)| = 3
    assert seq[1] == 2  # |V(L(P3))| = |E(P3)| = 2
    assert seq[2] == 1  # |V(L^2(P3))| = |E(K2)| = 1
    assert seq[3] == 0  # |V(L^3(P3))| = |E(K1)| = 0


def test_gamma_sequence_star_k14():
    # K_{1,4}: 5 vertices. L(K_{1,4}) = K_4 (4 vertices).
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    seq = gamma_sequence_edgelist(edges, 2)
    assert seq[0] == 5
    assert seq[1] == 4  # |V(K_4)| = 4
    assert seq[2] == 6  # |E(K_4)| = 6


# --- canonical form ---

def test_canonical_bruteforce_triangle():
    c1 = canonical_graph_bruteforce([(0, 1), (1, 2), (0, 2)])
    c2 = canonical_graph_bruteforce([(3, 4), (4, 5), (3, 5)])
    assert c1 == c2


def test_canonical_bruteforce_different():
    c_p3 = canonical_graph_bruteforce([(0, 1), (1, 2)])
    c_k3 = canonical_graph_bruteforce([(0, 1), (1, 2), (0, 2)])
    assert c_p3 != c_k3


# --- subgraphs ---

def test_enumerate_subgraphs_triangle():
    edges = [(0, 1), (1, 2), (0, 2)]
    subs = enumerate_connected_subgraphs(edges)
    # 3 single edges + 3 paths of length 2 + 1 triangle = 7 subsets
    # But by isomorphism: K2 (3 copies) + P3 (3 copies) + K3 (1 copy)
    # -> 3 iso classes
    assert len(subs) == 3  # K2, P3, K3


# --- naming ---

def test_tree_name_k2():
    assert tree_name([(0, 1)]) == "K2"


def test_tree_name_p4():
    assert tree_name([(0, 1), (1, 2), (2, 3)]) == "P4"


def test_tree_name_star():
    assert tree_name([(0, 1), (0, 2), (0, 3)]) == "K1,3"


def test_tree_name_fork():
    # Fork: 5 vertices, one degree-3 vertex
    assert tree_name([(0, 1), (1, 2), (2, 3), (2, 4)]) == "fork"


def test_describe_graph_cycle():
    assert describe_graph([(0, 1), (1, 2), (2, 0)]) == "C3"


def test_describe_graph_k2():
    assert describe_graph([(5, 7)]) == "K2"


# --- linalg ---

def test_row_reduce_identity():
    M = [[1, 0], [0, 1]]
    rref, pivots, rank = row_reduce_fraction(M, 2, 2)
    assert rank == 2
    assert pivots == [0, 1]


def test_exact_rank_rank_deficient():
    M = [[1, 2, 3], [2, 4, 6]]  # row 2 = 2 * row 1
    assert exact_rank(M, 2, 3) == 1


def test_exact_rank_full():
    M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert exact_rank(M, 3, 3) == 3


# --- automorphisms ---

def test_aut_size_triangle():
    edges = [(0, 1), (1, 2), (0, 2)]
    assert aut_size_edges(edges, 3) == 6


def test_aut_size_p3():
    edges = [(0, 1), (1, 2)]
    assert aut_size_edges(edges, 3) == 2


def test_orbit_size_k3():
    edges = [(0, 1), (1, 2), (0, 2)]
    # orbit = 3! / |Aut| = 6 / 6 = 1
    assert orbit_size_under_Sn(edges, 3) == 1


def test_orbit_size_p3():
    edges = [(0, 1), (1, 2)]
    # orbit = 3! / 2 = 3
    assert orbit_size_under_Sn(edges, 3) == 3
