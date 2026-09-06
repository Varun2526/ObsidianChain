"""Union-find (disjoint-set) over a dense integer address space.

Each address is a node. Each co-spend edge asserts "these two addresses are
controlled by the same entity". Union-find maintains the transitive closure of
those assertions incrementally, so the connected components at the end are the
candidate entities.

Two optimisations keep it near-linear, and both are worth being able to state
out loud:

* **Path compression** - after ``find(x)`` walks from x up to its root, every
  node on that path is re-pointed directly at the root, so the next lookup on
  any of them is O(1). Trees stay flat instead of degenerating into chains.

* **Union by rank** - when merging two trees, the shorter one is hung under
  the taller one, never the reverse. This bounds tree height at O(log n)
  before compression even applies.

Together they give an amortised cost per operation of O(alpha(n)) - the
inverse Ackermann function, which is below 5 for any n that fits in memory.
Effectively constant.

NOTE: ``find`` and ``union`` are intentionally left unimplemented. Everything
else in this module is built on top of them and will work once they are.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

ParentDType = np.int32
RankDType = np.uint8  # rank is bounded by log2(n); 255 is far beyond enough


class UnionFind:
    """Disjoint-set forest with path compression and union by rank.

    Nodes are integers in ``range(n)``. Initially every node is its own root,
    i.e. n singleton components.

    >>> uf = UnionFind(5)
    >>> uf.n_nodes
    5
    """

    __slots__ = ("_parent", "_rank", "_n_nodes")

    def __init__(self, n: int) -> None:
        """Create ``n`` singleton components.

        Allocates two flat arrays and nothing else: ``parent`` as int32 and
        ``rank`` as uint8, so ~5 bytes per node (about 6.5 MB for Elliptic++'s
        ~1.3M addresses).

        Args:
            n: Number of nodes. Must be non-negative.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        self._n_nodes = int(n)
        # Every node starts as its own parent: n components of size 1.
        self._parent = np.arange(n, dtype=ParentDType)
        self._rank = np.zeros(n, dtype=RankDType)

    # ---- the two methods you are implementing -------------------------

    def find(self, x: int) -> int:
        """Return the representative (root) of the component containing ``x``.

        Two nodes are in the same component if and only if they have the same
        root, so this is the primitive every other operation is built on.

        Must apply **path compression**: after locating the root, every node
        visited on the way up should end up pointing straight at it. The
        method is therefore not read-only - it mutates ``self._parent`` - but
        it must never change which nodes are grouped together.

        Args:
            x: Node id in ``range(self.n_nodes)``.

        Returns:
            The root node id of ``x``'s component. ``find(root)`` is the root
            itself, so the operation is idempotent.
        """
        if x < 0 or x >= self._n_nodes:
            raise IndexError(f"node id out of range: {x}")

        root = x

        # Walk upward until we reach the root.
        while self._parent[root] != root:
            root = int(self._parent[root])

        # Path compression: point every node on the path directly
        # to the root.
        while self._parent[x] != x:
            parent = int(self._parent[x])
            self._parent[x] = root
            x = parent

        return root
        

    def union(self, a: int, b: int) -> bool:
        """Merge the components containing ``a`` and ``b``.

        Must apply **union by rank**: attach the lower-ranked root beneath the
        higher-ranked one. When both roots have equal rank, either may become
        the parent, but the surviving root's rank increases by one. Rank is a
        height bound, not a size - do not confuse it with component size.

        Args:
            a: Node id.
            b: Node id.

        Returns:
            True if two distinct components were merged, False if ``a`` and
            ``b`` were already in the same component. The return value lets
            callers count real merges without recomputing anything.
        """
        root_a = self.find(a)
        root_b = self.find(b)

        # Already in the same component.
        if root_a == root_b:
            return False

        # Union by rank: attach the shorter tree under the taller tree.
        if self._rank[root_a] < self._rank[root_b]:
            root_a, root_b = root_b, root_a

        self._parent[root_b] = root_a

        # If both trees had the same rank, the resulting tree is one level taller.
        if self._rank[root_a] == self._rank[root_b]:
            self._rank[root_a] += 1

        return True

    # ---- built on top of find/union -----------------------------------

    @property
    def n_nodes(self) -> int:
        """Total number of nodes, including singletons."""
        return self._n_nodes

    def connected(self, a: int, b: int) -> bool:
        """Whether ``a`` and ``b`` currently belong to the same component."""
        return self.find(a) == self.find(b)

    def union_star(self, members: Sequence[int]) -> int:
        """Union every member of ``members`` into one component, star-wise.

        Links ``members[0]`` to each subsequent member: k-1 unions for k
        members, rather than the k(k-1)/2 an all-pairs loop would perform.
        The resulting component is identical because union-find is transitive
        - that is the whole point of using it here.

        Args:
            members: Node ids known to belong together, e.g. all input
                addresses of one transaction.

        Returns:
            The number of unions that actually merged something.
        """
        if len(members) < 2:
            return 0
        centre = members[0]
        merged = 0
        for other in members[1:]:
            if self.union(centre, other):
                merged += 1
        return merged

    def add_edges(self, edges: np.ndarray | Iterable[tuple[int, int]]) -> int:
        """Apply an edge list, returning how many edges merged two components.

        Args:
            edges: An (m, 2) integer array, or any iterable of (a, b) pairs.

        Returns:
            Count of unions that merged distinct components. ``m`` minus this
            is the number of redundant edges - pairs already known to be
            connected via some other transaction.
        """
        if isinstance(edges, np.ndarray):
            # .tolist() yields native Python ints, which are markedly faster
            # to work with here than iterating numpy scalars.
            edges = edges.tolist()
        merged = 0
        for a, b in edges:
            if self.union(a, b):
                merged += 1
        return merged

    def roots(self) -> np.ndarray:
        """Return the root of every node, as an int32 array of length n.

        Calls :meth:`find` on each node, which also fully flattens the forest
        as a side effect.
        """
        out = np.empty(self._n_nodes, dtype=ParentDType)
        for i in range(self._n_nodes):
            out[i] = self.find(i)
        return out

    def component_sizes(self) -> np.ndarray:
        """Return the size of every component, sorted descending.

        Length equals the number of components; the sum equals ``n_nodes``.
        """
        if self._n_nodes == 0:
            return np.zeros(0, dtype=np.int64)
        counts = np.bincount(self.roots(), minlength=self._n_nodes)
        sizes = counts[counts > 0]
        return np.sort(sizes)[::-1]

    def n_components(self) -> int:
        """Number of disjoint components, singletons included."""
        return int(self.component_sizes().size)

    def largest_component_size(self) -> int:
        """Size of the largest component - the headline clustering number."""
        if self._n_nodes == 0:
            return 0
        return int(self.component_sizes()[0])
