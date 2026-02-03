from grahamtools.linegraph.adjlist import line_graph_adj

def test_line_graph_adj_triangle():
    # triangle: 3 edges; L(C3) is also C3
    adj = [
        [1,2],
        [0,2],
        [0,1]
    ]
    ladj = line_graph_adj(adj)
    assert len(ladj) == 3
    degs = sorted(len(nei) for nei in ladj)
    assert degs == [2,2,2]
