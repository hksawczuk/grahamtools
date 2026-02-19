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

# Nauty wrappers
from .external.nauty import (
    nauty_available,
    dreadnaut_available,
    canon_g6,
    canon_label_g6,
    aut_size_g6,
    edgelist_to_g6,
    geng_g6,
    geng_connected_subcubic_g6,
)

# Shared utilities
from .utils.connectivity import is_connected_edges, connected_components_edges
from .utils.linegraph_edgelist import line_graph_edgelist, gamma_sequence_edgelist
from .utils.canonical import canonical_graph_nauty, canonical_graph_bruteforce
from .utils.subgraphs import enumerate_connected_subgraphs
from .utils.naming import tree_name, describe_graph
from .utils.linalg import exact_rank, row_reduce_fraction
from .utils.automorphisms import aut_size_edges, orbit_size_under_Sn

__all__ = [
    # IO
    "g6_to_nx",
    "g6_to_adjlist",
    # Invariants
    "GrahamSignature",
    "graham_sequence_g6",
    "graham_signature_g6",
    # Search
    "find_collision_subcubic",
    # WL
    "wlk_distinguishable_aligned",
    # Viz
    "draw_iterate_pair",
    # Nauty
    "nauty_available",
    "dreadnaut_available",
    "canon_g6",
    "canon_label_g6",
    "aut_size_g6",
    "edgelist_to_g6",
    "geng_g6",
    "geng_connected_subcubic_g6",
    # Utils
    "is_connected_edges",
    "connected_components_edges",
    "line_graph_edgelist",
    "gamma_sequence_edgelist",
    "canonical_graph_nauty",
    "canonical_graph_bruteforce",
    "enumerate_connected_subgraphs",
    "tree_name",
    "describe_graph",
    "exact_rank",
    "row_reduce_fraction",
    "aut_size_edges",
    "orbit_size_under_Sn",
]
