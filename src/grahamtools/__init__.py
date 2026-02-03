"""
grahamtools: utilities for iterated line graphs, Graham sequences, nauty enumeration,
collision search, and WL-k alignment experiments.
"""

from .io.graph6 import g6_to_nx, g6_to_adjlist
from .invariants.graham import (
    GrahamSignature,
    graham_sequence_g6,
    graham_signature_g6,
)
from .search.collisions_subcubic import find_collision_subcubic
from .wl.wlk_aligned import wlk_distinguishable_aligned
from .viz.draw import draw_iterate_pair

__all__ = [
    "g6_to_nx",
    "g6_to_adjlist",
    "GrahamSignature",
    "graham_sequence_g6",
    "graham_signature_g6",
    "find_collision_subcubic",
    "wlk_distinguishable_aligned",
    "draw_iterate_pair",
]
