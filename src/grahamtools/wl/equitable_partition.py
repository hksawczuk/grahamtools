"""WL-1 color refinement (equitable partition) on bitset adjacency."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple


def _refine_colors(adj: List[int], colors: Tuple[int, ...]) -> Tuple[int, ...]:
    """One round of WL-1 color refinement."""
    n = len(adj)
    sigs = []
    for u in range(n):
        neigh = adj[u]
        cnt: Counter[int] = Counter()
        while neigh:
            lsb = neigh & -neigh
            v = lsb.bit_length() - 1
            neigh ^= lsb
            cnt[colors[v]] += 1
        sig = (colors[u], tuple(sorted(cnt.items())))
        sigs.append(sig)

    uniq = {sig: i for i, sig in enumerate(sorted(set(sigs)))}
    return tuple(uniq[sigs[u]] for u in range(n))


def equitable_partition_bitset(
    adj: List[int],
    initial: Optional[Tuple[int, ...]] = None,
) -> Tuple[int, ...]:
    """Iterate WL-1 refinement to a fixed point.

    Parameters
    ----------
    adj : list[int]
        Bitset adjacency: adj[u] has bit v set iff u~v.
    initial : tuple[int, ...], optional
        Starting coloring. Defaults to degree coloring.

    Returns
    -------
    tuple[int, ...]
        Stable coloring (one color id per vertex).
    """
    n = len(adj)
    if initial is None:
        initial = tuple(bin(adj[u]).count("1") for u in range(n))

    colors = initial
    while True:
        newc = _refine_colors(adj, colors)
        if newc == colors:
            return colors
        colors = newc


def color_classes(colors: Tuple[int, ...]) -> List[List[int]]:
    """Group vertices by color, sorted deterministically."""
    groups: Dict[int, List[int]] = {}
    for v, c in enumerate(colors):
        groups.setdefault(c, []).append(v)
    cls = list(groups.values())
    cls.sort(key=lambda L: (len(L), L))
    return cls
