"""
Test closed-regular fiber approximation for K_{1,4} in L^k(K_5).

If we assume the fiber is closed (no inter-fiber edges) and regular 
at grade k0, then:
  d_m = 2 + 2^m * (d0 - 2)       (degree at step m after k0)
  V_m = V0 * prod_{i=0}^{m-1} d_i / 2   (vertex count)

Compare to actual data.
"""

# Actual K_{1,4} fiber data
# Orbit counts from ALG iteration
fiber_orbits = {
    4: {'f': 2, 'h': 2},
    5: {'f': 8, 'h': 2},
    6: {'f': 74, 'h': 14},
    7: {'f': 2186, 'h': 134},
    8: {'f': 256862, 'h': 4238},
    9: {'f': 123503606, 'h': 509486},
    10: {'f': 241959379022, 'h': 246497726},
    11: {'f': 1922278172321806, 'h': None},  # h not computed
}

# Labeled vertex count = h*60 + (f-h)*120
actual_V = {}
for k, d in fiber_orbits.items():
    if d['h'] is not None:
        actual_V[k] = d['h'] * 60 + (d['f'] - d['h']) * 120

# Intra-fiber edge counts (from fiber_degrees.py output)
intra_E = {
    4: 360,
    5: 7080,
    6: 135720,
}

# Mean intra-degree = 2*E_intra / V
mean_d = {}
for k in intra_E:
    mean_d[k] = 2 * intra_E[k] / actual_V[k]

print("=" * 70)
print("  K_{1,4} fiber: actual data")
print("=" * 70)
print(f"  {'Grade':>6s} {'|V|':>16s} {'E_intra':>12s} {'mean_d':>10s}")
for k in sorted(actual_V):
    v = actual_V[k]
    e = intra_E.get(k, None)
    md = mean_d.get(k, None)
    print(f"  {k:>6d} {v:>16d} {str(e) if e else '—':>12s} "
          f"{f'{md:.2f}' if md else '—':>10s}")

# ── Closed-regular prediction ──
# Start from each grade k0 where we have mean degree

print()
print("=" * 70)
print("  Closed-regular approximation: V_pred vs V_actual")
print("=" * 70)

for k0 in sorted(mean_d):
    d0 = mean_d[k0]
    V0 = actual_V[k0]

    print(f"\n  Starting at grade {k0}: V0={V0}, d0={d0:.2f}")
    print(f"  {'Grade':>6s} {'V_pred':>16s} {'V_actual':>16s} "
          f"{'ratio':>10s} {'d_pred':>10s}")

    V_pred = V0
    d_pred = d0
    for k in range(k0, 12):
        V_act = actual_V.get(k, None)
        ratio = V_pred / V_act if V_act else None
        print(f"  {k:>6d} {V_pred:>16.0f} "
              f"{str(V_act) if V_act else '—':>16s} "
              f"{f'{ratio:.6f}' if ratio else '—':>10s} "
              f"{d_pred:>10.2f}")
        # Next step
        V_pred = V_pred * d_pred / 2
        d_pred = 2 * d_pred - 2

# ── Also try: use actual intra-degree distribution ──
print()
print("=" * 70)
print("  Degree-distribution-aware prediction")
print("=" * 70)
print("  (Uses actual degree distribution, assumes closed fiber)")
print("  |V(L(G))| = |E(G)| = sum(d_v) / 2 = V * mean_d / 2 [same as regular]")
print("  But |E(L(G))| = sum_v C(d(v), 2) depends on degree variance!")
print()

# From fiber_degrees output:
degree_dists = {
    4: {6: 120},
    5: {16: 480, 18: 360},
    6: {32: 960, 34: 7080},
}

for k0 in sorted(degree_dists):
    dd = degree_dists[k0]
    V0 = sum(dd.values())
    E0 = sum(d * count for d, count in dd.items()) // 2
    # |V(L)| = E0
    # |E(L)| = sum_v C(d(v), 2) = sum over original vertices of d(d-1)/2
    E_L = sum(count * d * (d - 1) // 2 for d, count in dd.items())
    # mean degree of L(G)
    mean_d_L = 2 * E_L / E0 if E0 > 0 else 0

    print(f"  Grade {k0}: V={V0}, E={E0}, mean_d={2*E0/V0:.2f}")
    print(f"    -> L: V={E0}, E={E_L}, mean_d={mean_d_L:.2f}")
    if k0 + 1 in actual_V:
        print(f"    -> Actual V(grade {k0+1}) = {actual_V[k0+1]}")
        print(f"    -> Closed pred V(grade {k0+1}) = {E0}")
        print(f"    -> Ratio pred/actual = {E0 / actual_V[k0+1]:.6f}")
    if k0 + 2 in actual_V:
        print(f"    -> Closed pred V(grade {k0+2}) = {E_L}")
        print(f"    -> Actual V(grade {k0+2}) = {actual_V[k0+2]}")
        print(f"    -> Ratio pred/actual = {E_L / actual_V[k0+2]:.6f}")
    print()

# ── Growth rate comparison ──
print("=" * 70)
print("  Growth rate: log2(V_k) comparison")
print("=" * 70)
import math

print(f"  {'Grade':>6s} {'log2(V)':>12s} {'Δ':>8s} {'ΔΔ':>8s}")
prev_log = None
prev_delta = None
for k in sorted(actual_V):
    v = actual_V[k]
    lg = math.log2(v)
    delta = lg - prev_log if prev_log is not None else None
    ddelta = delta - prev_delta if delta is not None and prev_delta is not None else None
    print(f"  {k:>6d} {lg:>12.2f} "
          f"{f'{delta:.2f}' if delta else '—':>8s} "
          f"{f'{ddelta:.2f}' if ddelta else '—':>8s}")
    prev_log = lg
    prev_delta = delta