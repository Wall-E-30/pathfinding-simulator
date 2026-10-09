"""Core directed graph representation for road networks and pathfinding."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterator, Tuple


def _validate_cost(cost: Any) -> float:
    """Validate that a travel cost is a finite, non-negative real number in seconds."""
    if isinstance(cost, bool) or not isinstance(cost, (int, float)):
        raise TypeError(f"Travel cost must be a numeric value, got {type(cost).__name__}")
    if math.isnan(cost) or math.isinf(cost):
        raise ValueError(f"Travel cost must be a finite, non-negative number, got {cost}")
    if cost < 0:
        raise ValueError(f"Travel cost must be a non-negative number, got {cost}")
    return float(cost)


class Graph:
    """Directed weighted graph storing one travel cost in seconds per (u, v) edge."""

    def __init__(self) -> None:
        self._adj: Dict[Any, Dict[Any, float]] = {}
        self._nodes: Dict[Any, Dict[str, Any]] = {}

    @property
    def node_count(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        """Return the total number of directed edges in the graph."""
        return sum(len(neighbors) for neighbors in self._adj.values())

    @property
    def nodes(self) -> Iterator[Any]:
        """Return an iterator over all node identifiers."""
        return iter(self._nodes.keys())

    def has_node(self, node: Any) -> bool:
        """Check if a node exists in the graph."""
        return node in self._nodes

    def add_node(self, node: Any, **attrs: Any) -> None:
        """Add a node with optional attributes (e.g. coordinates)."""
        if node not in self._nodes:
            self._nodes[node] = {}
            self._adj[node] = {}
        self._nodes[node].update(attrs)

    def get_node_data(self, node: Any) -> Dict[str, Any]:
        """Get attributes associated with a node."""
        if node not in self._nodes:
            raise KeyError(f"Node {node!r} does not exist.")
        return dict(self._nodes[node])

    def has_edge(self, u: Any, v: Any) -> bool:
        """Return True if a directed edge (u, v) exists."""
        return u in self._adj and v in self._adj[u]

    def add_edge(self, u: Any, v: Any, cost: float) -> None:
        """Add or overwrite a directed edge (u, v) with a travel cost in seconds.

        Ensures each directed (u, v) pair stores exactly one cost.
        """
        valid_cost = _validate_cost(cost)
        if u not in self._nodes:
            self.add_node(u)
        if v not in self._nodes:
            self.add_node(v)
        self._adj[u][v] = valid_cost

    def get_edge_cost(self, u: Any, v: Any) -> float:
        """Return the travel cost of directed edge (u, v) in seconds.

        Raises:
            KeyError: If directed edge (u, v) does not exist.
        """
        if not self.has_edge(u, v):
            raise KeyError(f"Edge ({u!r}, {v!r}) does not exist.")
        return self._adj[u][v]

    def get_neighbors(self, u: Any) -> Dict[Any, float]:
        """Return a mapping of neighbor nodes to their edge travel costs."""
        if u not in self._adj:
            return {}
        return dict(self._adj[u])

    def update_edge(self, u: Any, v: Any, new_cost: float) -> None:
        """Update the travel cost of an existing directed edge (u, v).

        Behavior for nonexistent edges:
            Raises KeyError if the directed edge (u, v) does not exist in the graph.
            This ensures callers do not silently introduce unintended edges without explicit creation.

        Args:
            u: Source node ID.
            v: Destination node ID.
            new_cost: Non-negative travel time in seconds.

        Raises:
            KeyError: If edge (u, v) does not exist.
            ValueError: If new_cost is negative, NaN, or infinite.
            TypeError: If new_cost is not numeric.
        """
        if not self.has_edge(u, v):
            raise KeyError(f"Edge ({u!r}, {v!r}) does not exist in graph.")
        valid_cost = _validate_cost(new_cost)
        self._adj[u][v] = valid_cost

    @classmethod
    def from_osmnx(cls, multi_di_graph: Any, weight_attr: str = "travel_time") -> Graph:
        """Create a Graph from an OSMnx / NetworkX MultiDiGraph.

        Parallel edges between the same (u, v) directed pair are collapsed
        by retaining the edge with the minimum travel time.
        Node attributes are preserved.
        """
        graph = cls()

        for node, data in multi_di_graph.nodes(data=True):
            graph.add_node(node, **data)

        # Group parallel edges by (u, v) and select minimum weight
        collapsed_edges: Dict[Tuple[Any, Any], float] = {}

        for u, v, data in multi_di_graph.edges(data=True):
            if weight_attr not in data:
                continue
            cost = _validate_cost(data[weight_attr])
            pair = (u, v)
            if pair not in collapsed_edges or cost < collapsed_edges[pair]:
                collapsed_edges[pair] = cost

        # Verify all edge pairs had at least one valid weight
        seen_pairs = {(u, v) for u, v in multi_di_graph.edges()}
        for pair in seen_pairs:
            if pair not in collapsed_edges:
                raise KeyError(f"No valid {weight_attr} found for edge {pair}")

        for (u, v), cost in collapsed_edges.items():
            graph.add_edge(u, v, cost)

        return graph
