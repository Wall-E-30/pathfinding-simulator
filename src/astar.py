"""A* and Dijkstra shortest path planning for road network DiGraphs."""

from __future__ import annotations

import heapq
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx


def plan(
    graph: nx.DiGraph,
    start: Any,
    goal: Any,
    heuristic: Optional[Callable[[Any, Any], float]] = None,
) -> Tuple[List[Any], float, Dict[str, Any]]:
    """Compute the shortest path from start to goal minimizing travel time.

    When heuristic is None, h(n) defaults to 0.0, behaving exactly as Dijkstra's algorithm.
    When heuristic is provided, operates as A* search with f(n) = g(n) + h(n).

    Args:
        graph: NetworkX DiGraph where edges store 'travel_time' (seconds).
        start: Origin node ID.
        goal: Destination node ID.
        heuristic: Optional admissible heuristic callable h(node, goal) -> float in seconds.

    Returns:
        A tuple of (path, cost, stats):
        - path: List of node IDs from start to goal, inclusive (empty list if unreachable).
        - cost: Total travel time in seconds (float("inf") if unreachable).
        - stats: Dictionary {"expansions": int, "time_ms": float}.

    Viva notes:
        1. f(n) = g(n) + h(n):
           g(n) is the exact travel time from start to n.
           h(n) is the estimated travel time from n to goal (admissible lower bound).
        2. Heap Tie-breaking:
           Python's heapq compares tuple elements sequentially. Pushing
           (f_score, counter, g_score, node) prevents comparing node objects
           when two entries share identical f_scores.
        3. Lazy Deletion:
           Instead of finding and updating nodes inside the binary heap (which takes O(N)),
           we push new (f, g) pairs and simply skip stale popped entries whose g-score
           exceeds the recorded best or whose node is already in closed_set.
    """
    t_start = time.perf_counter()

    # Edge Case: Start or Goal missing from graph
    if start not in graph or goal not in graph:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return ([], float("inf"), {"expansions": 0, "time_ms": round(elapsed_ms, 3)})

    # Edge Case: Start is Goal
    if start == goal:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return ([start], 0.0, {"expansions": 0, "time_ms": round(elapsed_ms, 3)})

    # Track best known travel times from start
    g_score: Dict[Any, float] = {start: 0.0}

    # Track predecessor pointers for path reconstruction
    came_from: Dict[Any, Any] = {}

    # Closed set of finalized nodes (never expand a node twice)
    closed_set: Set[Any] = set()

    # Priority Queue storing: (f_score, tie_breaker_count, g_score, node)
    open_heap: List[Tuple[float, int, float, Any]] = []
    counter: int = 0

    # Initial heuristic for start node
    h_start = float(heuristic(start, goal)) if heuristic is not None else 0.0
    heapq.heappush(open_heap, (h_start, counter, 0.0, start))

    expansions: int = 0

    while open_heap:
        f, _, g, current = heapq.heappop(open_heap)

        # Lazy Deletion Check: Skip if already expanded
        if current in closed_set:
            continue

        # Skip stale heap entries with inferior g-scores
        if g > g_score.get(current, float("inf")):
            continue

        # Goal Reached: Reconstruct shortest path by backtracking
        if current == goal:
            path: List[Any] = [goal]
            curr_backtrack = goal
            while curr_backtrack in came_from:
                curr_backtrack = came_from[curr_backtrack]
                path.append(curr_backtrack)
            path.reverse()

            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return (path, g, {"expansions": expansions, "time_ms": round(elapsed_ms, 3)})

        # Finalize current node
        closed_set.add(current)
        expansions += 1

        # Explore outgoing directed edges
        for neighbor, edge_data in graph[current].items():
            if neighbor in closed_set:
                continue

            travel_time = float(edge_data.get("travel_time", 1.0))
            tentative_g = g + travel_time

            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current

                # Compute heuristic (h=0 for Dijkstra)
                h_val = float(heuristic(neighbor, goal)) if heuristic is not None else 0.0
                f_val = tentative_g + h_val

                counter += 1
                heapq.heappush(open_heap, (f_val, counter, tentative_g, neighbor))

    # Goal unreachable after exhausting component
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    return ([], float("inf"), {"expansions": expansions, "time_ms": round(elapsed_ms, 3)})
