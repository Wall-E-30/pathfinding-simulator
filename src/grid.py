"""Grid graph generator for testing pathfinding algorithms."""

from __future__ import annotations

from typing import Iterable, Set, Tuple
from src.graph import Graph

Coordinate = Tuple[int, int]


def _get_4_neighbors(x: int, y: int, width: int, height: int) -> list[Coordinate]:
    """Return valid 4-directional adjacent coordinates within grid bounds."""
    deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    neighbors = []
    for dx, dy in deltas:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height:
            neighbors.append((nx, ny))
    return neighbors


def create_grid_graph(
    width: int = 10,
    height: int = 10,
    walls: Iterable[Coordinate] | None = None,
    default_cost: float = 1.0,
) -> Graph:
    """Create a 2D grid graph with optional obstacle walls.

    Args:
        width: Number of columns.
        height: Number of rows.
        walls: Set or sequence of (x, y) coordinates representing blocked cells.
        default_cost: Edge traversal cost in seconds.

    Returns:
        Graph: Directed graph with node coordinates (x, y) and travel costs.
    """
    wall_set: Set[Coordinate] = set(walls) if walls else set()
    graph = Graph()

    # Add all nodes with spatial coordinates
    for x in range(width):
        for y in range(height):
            graph.add_node((x, y), x=float(x), y=float(y))

    # Add edges between non-wall nodes
    for x in range(width):
        for y in range(height):
            node = (x, y)
            if node in wall_set:
                continue

            for neighbor in _get_4_neighbors(x, y, width, height):
                if neighbor not in wall_set:
                    graph.add_edge(node, neighbor, default_cost)

    return graph
