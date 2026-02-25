"""Graham sequences via WL-1 quotient matrix iteration.

Instead of building L^k(G) explicitly (exponential growth), we compress G
into its WL-1 equitable partition quotient and derive L(G)'s quotient
algebraically.  The vertex count at each step equals the total number of
edges (= sum of edge-type sizes), which is always exact.

The edge-type partition of L(G) is equitable by construction: all edges of
the same type (i, j) have endpoints in the same pair of classes, so their
per-class neighbor distributions in L(G) are identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from grahamtools.wl.equitable_partition import equitable_partition_bitset, color_classes

# An edge type is an ordered pair (class_i, class_j) with i <= j.
EdgeType = Tuple[int, int]


@dataclass
class QuotientGraph:
    """Equitable partition quotient of an undirected graph.

    Vertices are grouped into *classes* where every vertex in class i
    has exactly ``matrix[i][j]`` neighbors in class j.

    Attributes
    ----------
    class_sizes : list[int]
        ``class_sizes[i]`` is the number of vertices in class i.
    matrix : list[list[int]]
        ``matrix[i][j]`` is the number of neighbors that each vertex
        in class i has in class j.
    """

    class_sizes: list[int]
    matrix: list[list[int]]

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        """Number of equivalence classes."""
        return len(self.class_sizes)

    @property
    def num_vertices(self) -> int:
        """Total vertex count (sum of class sizes)."""
        return sum(self.class_sizes)

    @property
    def num_edges(self) -> int:
        """Total edge count derived from the quotient.

        Between classes i < j: ``class_sizes[i] * matrix[i][j]`` edges.
        Within class i: ``class_sizes[i] * matrix[i][i] // 2`` edges.
        """
        c = self.num_classes
        total = 0
        for i in range(c):
            total += self.class_sizes[i] * self.matrix[i][i] // 2
            for j in range(i + 1, c):
                total += self.class_sizes[i] * self.matrix[i][j]
        return total

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_adj(cls, adj: Sequence[Sequence[int]]) -> QuotientGraph:
        """Build a QuotientGraph from an adjacency list via WL-1.

        Parameters
        ----------
        adj : Sequence[Sequence[int]]
            ``adj[u]`` is the list of neighbors of vertex u.
        """
        n = len(adj)
        if n == 0:
            return cls(class_sizes=[], matrix=[])

        # Convert to bitset adjacency for equitable_partition_bitset.
        adj_bits: list[int] = [0] * n
        for u in range(n):
            for v in adj[u]:
                adj_bits[u] |= 1 << v

        # WL-1 refinement to stable coloring.
        colors = equitable_partition_bitset(adj_bits)
        classes = color_classes(colors)
        c = len(classes)

        # Vertex-to-class-index mapping.
        vertex_class: list[int] = [0] * n
        for ci, members in enumerate(classes):
            for v in members:
                vertex_class[v] = ci

        sizes = [len(members) for members in classes]

        # Build quotient matrix: pick one representative per class.
        matrix: list[list[int]] = [[0] * c for _ in range(c)]
        for i in range(c):
            rep = classes[i][0]
            for v in adj[rep]:
                matrix[i][vertex_class[v]] += 1

        return cls(class_sizes=sizes, matrix=matrix)

    # ------------------------------------------------------------------
    #  Edge types
    # ------------------------------------------------------------------

    def edge_types(self) -> list[tuple[EdgeType, int]]:
        """Enumerate non-empty edge types and their multiplicities.

        Returns ``[((i, j), count), ...]`` sorted by ``(i, j)``.
        """
        c = self.num_classes
        result: list[tuple[EdgeType, int]] = []
        for i in range(c):
            for j in range(i, c):
                if self.matrix[i][j] == 0:
                    continue
                if i == j:
                    count = self.class_sizes[i] * self.matrix[i][i] // 2
                else:
                    count = self.class_sizes[i] * self.matrix[i][j]
                if count > 0:
                    result.append(((i, j), count))
        return result

    # ------------------------------------------------------------------
    #  Line graph quotient derivation
    # ------------------------------------------------------------------

    def line_graph_quotient(self) -> QuotientGraph:
        """Derive the quotient of L(G) from this quotient.

        Vertices of L(G) are edges of G, partitioned by edge type (i, j)
        where i <= j are the classes of the two endpoints.  This partition
        is equitable, so the quotient fully determines the next step.

        Returns
        -------
        QuotientGraph
            Quotient of L(G).
        """
        etypes = self.edge_types()
        if not etypes:
            return QuotientGraph(class_sizes=[], matrix=[])

        etype_list = [et for et, _ in etypes]
        etype_sizes = [cnt for _, cnt in etypes]
        t = len(etype_list)

        Q = self.matrix

        def _edges_from(src: int, a: int, b: int) -> int:
            """Edges of type (a, b) incident to a vertex in class *src*."""
            # Edge from src-vertex to class-k vertex has type
            # (min(src, k), max(src, k)).  We need that to equal (a, b).
            if a == b:
                return Q[src][a] if src == a else 0
            # a < b
            if src == a:
                return Q[src][b]
            if src == b:
                return Q[src][a]
            return 0

        new_matrix: list[list[int]] = [[0] * t for _ in range(t)]
        for p, (i, j) in enumerate(etype_list):
            for q_idx, (a, b) in enumerate(etype_list):
                val = _edges_from(i, a, b) + _edges_from(j, a, b)
                if (a, b) == (i, j):
                    val -= 2  # exclude edge itself (counted once per endpoint)
                new_matrix[p][q_idx] = val

        return QuotientGraph(class_sizes=etype_sizes, matrix=new_matrix)

    # ------------------------------------------------------------------
    #  Compression (WL-1 on the quotient graph)
    # ------------------------------------------------------------------

    def compress(self) -> QuotientGraph:
        """Merge structurally identical classes.

        Two classes can be merged when they have the same size, the same
        row in the quotient matrix, and the same column.  This is
        equivalent to running WL-1 on the quotient graph.  Merging never
        changes the vertex or edge counts.
        """
        c = self.num_classes
        if c <= 1:
            return self

        # Start with a coarse initial coloring: group by class size.
        # WL-1 refines from here; starting all-distinct would be a
        # no-op since every class would already be "unique".
        size_sigs = sorted(set(self.class_sizes))
        labels = [size_sigs.index(self.class_sizes[i]) for i in range(c)]

        prev_num_colors = len(set(labels))

        while True:
            sigs: list[tuple] = []
            for i in range(c):
                row_sig = tuple(sorted(
                    (labels[j], self.matrix[i][j])
                    for j in range(c) if self.matrix[i][j] > 0
                ))
                col_sig = tuple(sorted(
                    (labels[j], self.matrix[j][i])
                    for j in range(c) if self.matrix[j][i] > 0
                ))
                sigs.append((self.class_sizes[i], row_sig, col_sig))

            # Assign new labels by first-occurrence order so that label
            # assignment is independent of signature sort order (prevents
            # oscillation between equivalent relabelings).
            seen: dict[tuple, int] = {}
            new_labels: list[int] = []
            for sig in sigs:
                if sig not in seen:
                    seen[sig] = len(seen)
                new_labels.append(seen[sig])

            num_colors = len(seen)
            if num_colors == prev_num_colors:
                break  # partition did not refine further
            prev_num_colors = num_colors
            labels = new_labels

        # Group old classes by final label.
        groups: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            groups.setdefault(lab, []).append(i)

        new_c = len(groups)
        if new_c == c:
            return self  # nothing to merge

        group_list = sorted(groups.values(), key=lambda g: g[0])

        new_sizes = [
            sum(self.class_sizes[m] for m in members)
            for members in group_list
        ]

        # All old classes in a group have identical rows, so one
        # representative suffices.  New Q[I][J] = sum of old Q[rep][j]
        # for all j in group J.
        new_matrix: list[list[int]] = [[0] * new_c for _ in range(new_c)]
        for new_i, members_i in enumerate(group_list):
            rep = members_i[0]
            for new_j, members_j in enumerate(group_list):
                new_matrix[new_i][new_j] = sum(
                    self.matrix[rep][old_j] for old_j in members_j
                )

        return QuotientGraph(class_sizes=new_sizes, matrix=new_matrix)


# ------------------------------------------------------------------
#  Public API
# ------------------------------------------------------------------


def graham_sequence_wl1(
    adj: Sequence[Sequence[int]],
    k_max: int,
) -> list[int]:
    """Compute the Graham sequence via WL-1 quotient compression.

    The Graham sequence is ``gamma_k(G) = |V(L^k(G))|`` for
    ``k = 0, 1, ..., k_max``.  Instead of building each iterated line
    graph explicitly, this function tracks only the quotient matrix of
    the equitable partition, deriving L(G)'s quotient algebraically at
    each step.

    Parameters
    ----------
    adj : Sequence[Sequence[int]]
        Adjacency list of the input graph.
    k_max : int
        Maximum iteration depth.

    Returns
    -------
    list[int]
        ``[|V(G)|, |V(L(G))|, ..., |V(L^{k_max}(G))|]``.
        Stops early (with a trailing 0) if an iterate becomes edgeless.
    """
    q = QuotientGraph.from_adj(adj)
    seq = [q.num_vertices]

    for _ in range(k_max):
        if q.num_edges == 0:
            seq.append(0)
            break
        q = q.line_graph_quotient().compress()
        seq.append(q.num_vertices)

    return seq
