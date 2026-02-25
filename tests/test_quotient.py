"""Tests for WL-1 quotient matrix iteration (Graham sequences)."""

from grahamtools.quotient import QuotientGraph, graham_sequence_wl1
from grahamtools.invariants.graham import graham_sequence_adj


# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------


def _edges_to_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return [sorted(neigh) for neigh in adj]


# ------------------------------------------------------------------
#  QuotientGraph construction
# ------------------------------------------------------------------


def test_quotient_empty():
    q = QuotientGraph.from_adj([])
    assert q.num_classes == 0
    assert q.num_vertices == 0
    assert q.num_edges == 0


def test_quotient_single_vertex():
    q = QuotientGraph.from_adj([[]])
    assert q.num_classes == 1
    assert q.num_vertices == 1
    assert q.num_edges == 0


def test_quotient_k2():
    adj = _edges_to_adj([(0, 1)], 2)
    q = QuotientGraph.from_adj(adj)
    assert q.num_classes == 1
    assert q.class_sizes == [2]
    assert q.matrix == [[1]]
    assert q.num_vertices == 2
    assert q.num_edges == 1


def test_quotient_k4():
    """K_4: one class of size 4, 3-regular."""
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    adj = _edges_to_adj(edges, 4)
    q = QuotientGraph.from_adj(adj)
    assert q.num_classes == 1
    assert q.class_sizes == [4]
    assert q.matrix == [[3]]
    assert q.num_vertices == 4
    assert q.num_edges == 6


def test_quotient_star_k13():
    """K_{1,3}: two classes — center and leaves."""
    adj = _edges_to_adj([(0, 1), (0, 2), (0, 3)], 4)
    q = QuotientGraph.from_adj(adj)
    assert q.num_classes == 2
    assert q.num_vertices == 4
    assert q.num_edges == 3


def test_quotient_cycle_c5():
    """C_5 is vertex-transitive: one class of size 5, 2-regular."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    adj = _edges_to_adj(edges, 5)
    q = QuotientGraph.from_adj(adj)
    assert q.num_classes == 1
    assert q.class_sizes == [5]
    assert q.matrix == [[2]]
    assert q.num_edges == 5


def test_quotient_path_p4():
    """P_4: two classes (degree-1 endpoints and degree-2 middles)."""
    adj = _edges_to_adj([(0, 1), (1, 2), (2, 3)], 4)
    q = QuotientGraph.from_adj(adj)
    assert q.num_classes == 2
    assert q.num_vertices == 4
    assert q.num_edges == 3


# ------------------------------------------------------------------
#  Line graph quotient
# ------------------------------------------------------------------


def test_line_quotient_k2():
    """L(K_2) has 1 vertex, 0 edges."""
    adj = _edges_to_adj([(0, 1)], 2)
    q = QuotientGraph.from_adj(adj)
    lq = q.line_graph_quotient()
    assert lq.num_vertices == 1
    assert lq.num_edges == 0


def test_line_quotient_triangle():
    """L(C_3) = C_3: 3 vertices, 3 edges."""
    adj = _edges_to_adj([(0, 1), (1, 2), (0, 2)], 3)
    q = QuotientGraph.from_adj(adj)
    lq = q.line_graph_quotient()
    assert lq.num_vertices == 3
    assert lq.num_edges == 3


def test_line_quotient_k4():
    """L(K_4) = octahedron: 6 vertices, 12 edges, 4-regular."""
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    adj = _edges_to_adj(edges, 4)
    q = QuotientGraph.from_adj(adj)
    lq = q.line_graph_quotient()
    assert lq.num_vertices == 6
    assert lq.num_edges == 12


def test_line_quotient_star_k13():
    """L(K_{1,3}) = K_3: 3 vertices, 3 edges."""
    adj = _edges_to_adj([(0, 1), (0, 2), (0, 3)], 4)
    q = QuotientGraph.from_adj(adj)
    lq = q.line_graph_quotient()
    assert lq.num_vertices == 3
    assert lq.num_edges == 3


def test_line_quotient_star_k14():
    """L(K_{1,4}) = K_4: 4 vertices (= 4 edges of K_{1,4}), 6 edges."""
    adj = _edges_to_adj([(0, 1), (0, 2), (0, 3), (0, 4)], 5)
    q = QuotientGraph.from_adj(adj)
    lq = q.line_graph_quotient()
    assert lq.num_vertices == 4
    assert lq.num_edges == 6


# ------------------------------------------------------------------
#  Compression
# ------------------------------------------------------------------


def test_compress_preserves_counts():
    """Compression must not change vertex or edge counts."""
    adj = _edges_to_adj([(0, 1), (1, 2), (2, 3)], 4)
    q = QuotientGraph.from_adj(adj)
    lq = q.line_graph_quotient()
    lqc = lq.compress()
    assert lqc.num_vertices == lq.num_vertices
    assert lqc.num_edges == lq.num_edges


def test_compress_star_line_gives_complete():
    """L(K_{1,4}) = K_4, vertex-transitive → compression to 1 class."""
    adj = _edges_to_adj([(0, 1), (0, 2), (0, 3), (0, 4)], 5)
    q = QuotientGraph.from_adj(adj)
    lq = q.line_graph_quotient().compress()
    assert lq.num_classes == 1
    assert lq.num_vertices == 4


def test_compress_noop_on_single_class():
    """Compression on a single-class quotient is a no-op."""
    adj = _edges_to_adj([(0, 1), (1, 2), (0, 2)], 3)
    q = QuotientGraph.from_adj(adj)
    assert q.num_classes == 1
    qc = q.compress()
    assert qc.num_classes == 1


# ------------------------------------------------------------------
#  Graham sequence correctness (vs brute-force)
# ------------------------------------------------------------------


def test_graham_wl1_single_edge():
    adj = _edges_to_adj([(0, 1)], 2)
    assert graham_sequence_wl1(adj, 3) == [2, 1, 0]


def test_graham_wl1_isolated_vertex():
    assert graham_sequence_wl1([[]], 3) == [1, 0]


def test_graham_wl1_vs_brute_triangle():
    adj = _edges_to_adj([(0, 1), (1, 2), (0, 2)], 3)
    assert graham_sequence_wl1(adj, 5) == graham_sequence_adj(adj, 5)


def test_graham_wl1_vs_brute_k4():
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    adj = _edges_to_adj(edges, 4)
    assert graham_sequence_wl1(adj, 4) == graham_sequence_adj(adj, 4)


def test_graham_wl1_vs_brute_path_p4():
    adj = _edges_to_adj([(0, 1), (1, 2), (2, 3)], 4)
    assert graham_sequence_wl1(adj, 5) == graham_sequence_adj(adj, 5)


def test_graham_wl1_vs_brute_star_k14():
    adj = _edges_to_adj([(0, 1), (0, 2), (0, 3), (0, 4)], 5)
    assert graham_sequence_wl1(adj, 4) == graham_sequence_adj(adj, 4)


def test_graham_wl1_vs_brute_cycle_c5():
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    adj = _edges_to_adj(edges, 5)
    assert graham_sequence_wl1(adj, 4) == graham_sequence_adj(adj, 4)


def test_graham_wl1_vs_brute_cycle_c6():
    edges = [(i, (i + 1) % 6) for i in range(6)]
    adj = _edges_to_adj(edges, 6)
    assert graham_sequence_wl1(adj, 4) == graham_sequence_adj(adj, 4)


def test_graham_wl1_vs_brute_petersen():
    """Petersen graph: 3-regular, 10 vertices, 15 edges."""
    outer = [(i, (i + 1) % 5) for i in range(5)]
    inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    spokes = [(i, 5 + i) for i in range(5)]
    adj = _edges_to_adj(outer + inner + spokes, 10)
    assert graham_sequence_wl1(adj, 3) == graham_sequence_adj(adj, 3)


def test_graham_wl1_vs_brute_dumbbell():
    """Two triangles joined by a bridge."""
    edges = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (3, 5)]
    adj = _edges_to_adj(edges, 6)
    assert graham_sequence_wl1(adj, 3) == graham_sequence_adj(adj, 3)


def test_graham_wl1_vs_brute_double_star_s22():
    """S_{2,2}: two hubs (each with 2 leaves) joined by an edge."""
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]
    adj = _edges_to_adj(edges, 6)
    assert graham_sequence_wl1(adj, 4) == graham_sequence_adj(adj, 4)


def test_graham_wl1_vs_brute_double_star_s33():
    """S_{3,3}: two hubs (each with 3 leaves) joined by an edge."""
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)]
    adj = _edges_to_adj(edges, 8)
    assert graham_sequence_wl1(adj, 3) == graham_sequence_adj(adj, 3)
