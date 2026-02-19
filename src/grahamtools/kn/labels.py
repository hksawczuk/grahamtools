"""Human-readable label formatting for K_n iterated line graph vertices."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

from grahamtools.kn.levels import Endpoints


def format_label(
    v: int,
    level: int,
    endpoints_by_level: Dict[int, List[Endpoints]],
    sep_after_level: int = 1,
) -> str:
    """Convert an internal vertex ID at a given level to a recursive string label.

    Base vertices use 1-based labels (1..n).
    *sep_after_level*: concatenate through this level, then separate with '|'.
    """

    @lru_cache(maxsize=None)
    def rec(v_id: int, lvl: int) -> str:
        if lvl == 0:
            return str(v_id + 1)  # 1-based display
        a, b = endpoints_by_level[lvl][v_id]
        sa, sb = rec(a, lvl - 1), rec(b, lvl - 1)
        left, right = (sa, sb) if sa <= sb else (sb, sa)
        if lvl <= sep_after_level:
            return f"{left}{right}"
        return f"{left}|{right}"

    return rec(v, level)
