"""Helper graphs and heuristics for testing pathfinding algorithms."""

from __future__ import annotations

import random
from typing import Set, Tuple
import networkx as nx

Coordinate = Tuple[int, int]


def make_grid(
    n: int = 10,
    walls: Set[Coordinate] | None = None,
    seed: int | None = None,
) -> nx.DiGraph:
    """Create a 4-connected grid DiGraph of size n x n with bidirectional edges.

    Each directed edge has attribute travel_time = 1.0 (seconds).
    Each node has attributes x (float) and y (float).
    Wall nodes and their incident edges are excluded.
    """
    wall_set = walls if walls is not None else set()
    graph = nx.DiGraph()

    for x in range(n):
        for y in range(n):
            if (x, y) not in wall_set:
                graph.add_node((x, y), x=float(x), y=float(y))

    deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for x, y in list(graph.nodes()):
        for dx, dy in deltas:
            nx_coord, ny_coord = x + dx, y + dy
            if 0 <= nx_coord < n and 0 <= ny_coord < n:
                if (nx_coord, ny_coord) not in wall_set:
                    graph.add_edge((x, y), (nx_coord, ny_coord), travel_time=1.0)

    return graph


def manhattan(n: Coordinate, goal: Coordinate) -> float:
    """Calculate Manhattan distance between two 2D grid coordinates.

    With unit travel time per grid step, this serves as an admissible
    heuristic lower bound for travel time in seconds.
    """
    return float(abs(n[0] - goal[0]) + abs(n[1] - goal[1]))


def random_grid_with_walls(
    seed: int | None = None,
    n: int = 10,
    wall_ratio: float = 0.2,
) -> nx.DiGraph:
    """Generate an n x n grid DiGraph with ~20% randomly placed wall obstacles."""
    rng = random.Random(seed)
    all_cells = [(x, y) for x in range(n) for y in range(n)]
    num_walls = int(len(all_cells) * wall_ratio)
    walls = set(rng.sample(all_cells, num_walls))
    return make_grid(n=n, walls=walls, seed=seed)

