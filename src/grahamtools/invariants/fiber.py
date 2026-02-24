"""Fiber coefficient computation via Möbius inversion bootstrapping.

The fiber coefficient coeff_k(τ) captures how many times the tree type τ
appears as a "fiber" in the k-th iterated line graph L^k(K_n). These
coefficients are universal — independent of the host K_n — and can be
extracted efficiently via Möbius inversion on the subgraph containment
poset.
"""
from __future__ import annotations

import time
from typing import Optional

from grahamtools.utils.linegraph_edgelist import gamma_sequence_edgelist
from grahamtools.utils.subgraphs import enumerate_connected_subgraphs
from grahamtools.utils.naming import describe_graph


def compute_all_coefficients(
    all_types: dict[object, tuple[int, list[tuple[int, int]], int, int]],
    max_k: int,
    *,
    max_edges: int = 5_000_000,
    verbose: bool = False,
) -> tuple[
    dict[object, list[int | None]],
    dict[object, list[int | None]],
    dict[object, dict[object, int]],
]:
    """Compute fiber coefficients for all types by Möbius inversion.

    Parameters
    ----------
    all_types : dict
        Mapping of canonical_form -> (count, edges, n_verts, n_edges)
        as returned by :func:`enumerate_connected_subgraphs`.
    max_k : int
        Maximum grade (number of line graph iterations).
    max_edges : int
        Safety cap on edge count during line graph iteration.
        Grades where the line graph exceeds this are recorded as None.
    verbose : bool
        If True, print progress to stdout.

    Returns
    -------
    gammas : dict
        canon -> [gamma_0, gamma_1, ..., gamma_max_k]
    coeffs : dict
        canon -> [coeff_0, coeff_1, ..., coeff_max_k]
    subtypes : dict
        canon -> {sub_canon: count} for proper connected subgraphs
    """
    sorted_types = sorted(all_types.items(), key=lambda x: x[1][3])

    gammas: dict[object, list[int | None]] = {}
    coeffs: dict[object, list[int | None]] = {}
    subtypes: dict[object, dict[object, int]] = {}

    total = len(sorted_types)

    for idx, (canon, (_count, edges, nv, ne)) in enumerate(sorted_types):
        desc = describe_graph(edges)
        if verbose:
            print(
                f"  [{idx + 1}/{total}] {desc} ({ne} edges)...",
                end="",
                flush=True,
            )

        t0 = time.time()

        # Step 1: compute gamma_k(tau) by line graph iteration
        gamma = gamma_sequence_edgelist(edges, max_k, max_edges=max_edges)
        gammas[canon] = gamma

        # Step 2: enumerate proper connected subgraphs of tau
        if ne > 1:
            sub_counts = enumerate_connected_subgraphs(edges, max_size=ne - 1)
        else:
            sub_counts = {}
        subtypes[canon] = {sc: sv[0] for sc, sv in sub_counts.items()}

        # Step 3: Mobius extraction
        # coeff_k(tau) = gamma_k(tau) - sum_{sigma ⊂ tau} coeff_k(sigma) * count(sigma, tau)
        coeff: list[int | None] = list(gamma)
        for sub_canon, sub_count in subtypes[canon].items():
            if sub_canon in coeffs:
                for k in range(min(len(coeff), len(coeffs[sub_canon]))):
                    if coeff[k] is not None and coeffs[sub_canon][k] is not None:
                        coeff[k] -= coeffs[sub_canon][k] * sub_count

        coeffs[canon] = coeff

        if verbose:
            elapsed = time.time() - t0
            nonzero = [
                (k, c)
                for k, c in enumerate(coeff)
                if c is not None and c != 0 and k > 0
            ]
            if nonzero:
                nz_str = ", ".join(f"c_{k}={c}" for k, c in nonzero[:6])
                print(f" {nz_str} ({elapsed:.2f}s)")
            else:
                print(f" all zero ({elapsed:.2f}s)")

    return gammas, coeffs, subtypes
