from grahamtools.invariants.graham import graham_signature_g6, graham_sequence_g6
from grahamtools.viz.draw import draw_iterate_pair

g6A = "EEj_"
g6B = "EQjO"

sigA, canA = graham_signature_g6(g6A, k_cap=6, N_cap=200_000)
sigB, canB = graham_signature_g6(g6B, k_cap=6, N_cap=200_000)

print("A canon:", canA, "sig:", sigA)
print("B canon:", canB, "sig:", sigB)
print("A gamma:", graham_sequence_g6(g6A, 6))
print("B gamma:", graham_sequence_g6(g6B, 6))

draw_iterate_pair(g6A, g6B, k=3, seed=7)
