from __future__ import annotations

import sys
from multiprocessing import Pool, cpu_count
from typing import Dict, Iterable, List, Optional, Tuple

from grahamtools.external.nauty import geng_connected_subcubic_g6, canon_g6
from grahamtools.invariants.graham import GrahamSignature, graham_signature_canon_g6


_ITER_CACHE: Dict[str, GrahamSignature] = {}


def _worker_init() -> None:
    global _ITER_CACHE
    _ITER_CACHE = {}


def _worker(job: Tuple[str, int, int]) -> Tuple[GrahamSignature, str, str]:
    """
    Return (signature, original_g6, canonical_g6).
    """
    g6, k_cap, N_cap = job
    canon0 = canon_g6(g6)
    sig = graham_signature_canon_g6(canon0, k_cap=k_cap, N_cap=N_cap, cache=_ITER_CACHE)
    return sig, g6, canon0


def _chunked(it: Iterable[str], size: int) -> Iterable[List[str]]:
    buf: List[str] = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def find_collision_subcubic(
    *,
    n_min: int,
    n_max: int,
    k_cap: int,
    N_cap: int,
    processes: int = max(1, cpu_count() - 1),
    batch_size: int = 200,
) -> Optional[Tuple[int, GrahamSignature, Tuple[str, str], Tuple[str, str]]]:
    """
    Search n in [n_min, n_max] for the first signature collision among connected subcubic graphs.

    Returns:
      (n, signature, (orig_g6_A, canon_A), (orig_g6_B, canon_B))
    or None if not found.
    """
    for n in range(n_min, n_max + 1):
        print(f"[n={n}] geng connected Δ≤3 graphs...", file=sys.stderr)

        witness: Dict[GrahamSignature, Tuple[str, str]] = {}
        gen = geng_connected_subcubic_g6(n)

        with Pool(processes=processes, initializer=_worker_init) as pool:
            for batch in _chunked(gen, batch_size):
                jobs = [(g6, k_cap, N_cap) for g6 in batch]
                for sig, orig, canon in pool.imap_unordered(_worker, jobs, chunksize=1):
                    if sig in witness:
                        orig2, canon2 = witness[sig]
                        if canon2 != canon:
                            return (n, sig, (orig2, canon2), (orig, canon))
                    else:
                        witness[sig] = (orig, canon)

        print(f"[n={n}] no collision found.", file=sys.stderr)

    return None
