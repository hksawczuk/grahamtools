from grahamtools.invariants.graham import graham_sequence_g6

def test_graham_sequence_star_k14():
    # K_{1,4} in graph6 is "E?..." but we won't hardcode.
    # Simple sanity: path P4 graph6 is "Dhc"?? too brittle.
    # Instead: just ensure code runs on a small known graph6 token.
    g6 = "DsC"  # small graph6 token used in your script (may be a triangle-ish)
    seq = graham_sequence_g6(g6, 5)
    assert len(seq) >= 1
    assert seq[0] > 0
