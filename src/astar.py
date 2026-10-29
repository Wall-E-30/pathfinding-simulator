"""A* and Dijkstra shortest path planning for road network DiGraphs.

Includes:
- plan: Static shortest-path planner adhering to the shared contract.
- run_astar_agent: Dynamic simulation agent with real-time replanning on network events.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

from src.events import Event, apply_events


def plan(
    graph: nx.DiGraph,
    start: Any,
    goal: Any,
    heuristic: Optional[Callable[[Any, Any], float]] = None,
) -> Tuple[List[Any], float, Dict[str, Any]]:
    """Compute the shortest path from start to goal minimizing travel time.

    When heuristic is None, h(n) defaults to 0.0, behaving exactly as Dijkstra's algorithm.
    When heuristic is provided, operates as A* search with f(n) = g(n) + h(n).
    Edges with travel_time = float("inf") are treated as blocked.

    Args:
        graph: NetworkX DiGraph where edges store 'travel_time' (seconds).
        start: Origin node ID.
        goal: Destination node ID.
        heuristic: Optional admissible heuristic callable h(node, goal) -> float in seconds.

    Returns:
        (path, cost, stats) where stats = {"expansions": int, "time_ms": float}.
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

    g_score: Dict[Any, float] = {start: 0.0}
    came_from: Dict[Any, Any] = {}
    closed_set: Set[Any] = set()

    # Priority Queue: (f_score, tie_breaker_count, g_score, node)
    open_heap: List[Tuple[float, int, float, Any]] = []
    counter: int = 0

    h_start = float(heuristic(start, goal)) if heuristic is not None else 0.0
    heapq.heappush(open_heap, (h_start, counter, 0.0, start))

    expansions: int = 0

    while open_heap:
        f, _, g, current = heapq.heappop(open_heap)

        if current in closed_set:
            continue
        if g > g_score.get(current, float("inf")):
            continue

        if current == goal:
            path: List[Any] = [goal]
            curr_backtrack = goal
            while curr_backtrack in came_from:
                curr_backtrack = came_from[curr_backtrack]
                path.append(curr_backtrack)
            path.reverse()

            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return (path, g, {"expansions": expansions, "time_ms": round(elapsed_ms, 3)})

        closed_set.add(current)
        expansions += 1

        for neighbor, edge_data in graph[current].items():
            if neighbor in closed_set:
                continue

            travel_time = float(edge_data.get("travel_time", 1.0))
            # Treat float("inf") or negative travel times as blocked edges
            if math.isinf(travel_time) or travel_time < 0:
                continue

            tentative_g = g + travel_time

            if tentative_g < g_score.get(neighbor, float("inf")):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current

                h_val = float(heuristic(neighbor, goal)) if heuristic is not None else 0.0
                f_val = tentative_g + h_val

                counter += 1
                heapq.heappush(open_heap, (f_val, counter, tentative_g, neighbor))

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    return ([], float("inf"), {"expansions": expansions, "time_ms": round(elapsed_ms, 3)})


def _verify_replan_with_dijkstra(
    g: nx.DiGraph,
    current: Any,
    goal: Any,
    planner_cost: float,
) -> None:
    """Assert planner cost equals fresh NetworkX Dijkstra on non-inf edges.

    Viva note:
        Runs independent Dijkstra from the current node on a filtered view excluding
        blocked (inf) edges. Proves correctness of the dynamic agent's replanned path.
    """
    valid_edges = [
        (u, v, d) for u, v, d in g.edges(data=True)
        if not math.isinf(float(d.get("travel_time", 1.0)))
    ]
    diag_g = nx.DiGraph()
    diag_g.add_nodes_from(g.nodes(data=True))
    diag_g.add_edges_from(valid_edges)

    try:
        nx_cost = nx.shortest_path_length(diag_g, current, goal, weight="travel_time")
        nx_reachable = True
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        nx_cost = float("inf")
        nx_reachable = False

    if nx_reachable:
        assert math.isclose(planner_cost, nx_cost, rel_tol=1e-5, abs_tol=1e-6), (
            f"Verification failed: planner cost {planner_cost} != Dijkstra cost {nx_cost}"
        )
    else:
        assert math.isinf(planner_cost), (
            f"Verification failed: Dijkstra found unreachable, but planner got {planner_cost}"
        )


def run_astar_agent(
    graph: nx.DiGraph,
    start: Any,
    goal: Any,
    events: Iterable[Event],
    heuristic: Optional[Callable[[Any, Any], float]] = None,
    verify: bool = False,
) -> Dict[str, Any]:
    """Execute dynamic emergency vehicle path traversal using reactive A* replanning.

    Viva note:
        1. Time model: t represents the count of moves made. Events with step == t
           are applied after the vehicle reaches its t-th node, BEFORE choosing its next move.
        2. Immutability: Operates on graph.copy() to guarantee the caller's graph is unmutated.
        3. Immediate Sensing: Re-plans whenever edge costs are modified or the forward edge is closed.
        4. Cycle Protection: max_steps = 5 * |V| prevents infinite livelocks in oscillating networks.

    Args:
        graph: NetworkX DiGraph representing street network.
        start: Starting node ID.
        goal: Destination node ID.
        events: Dynamic traffic events (closures and congestions).
        heuristic: Optional admissible heuristic h(n, goal) in seconds.
        verify: If True, asserts cost-to-goal matches fresh Dijkstra after every replan.

    Returns:
        Result dictionary containing status, path_taken, travel_time, steps, replans,
        expansions breakdown, and replan latencies.
    """
    # Work on a copy; never mutate caller's graph
    g = graph.copy()
    events_list = list(events)
    max_steps = 5 * g.number_of_nodes()

    # Initial plan at t = 0
    t0 = time.perf_counter()
    init_path, init_cost, init_stats = plan(g, start, goal, heuristic)
    init_lat_ms = (time.perf_counter() - t0) * 1000.0

    initial_expansions = init_stats["expansions"]
    replan_expansions = 0
    replans = 0
    replan_latencies_ms: List[float] = []

    current = start
    path_taken: List[Any] = [current]
    travel_time: float = 0.0
    t: int = 0

    if start not in g or goal not in g or init_cost == float("inf") or not init_path:
        return {
            "status": "stuck",
            "path_taken": path_taken,
            "travel_time": 0.0,
            "steps": 0,
            "replans": 0,
            "initial_expansions": initial_expansions,
            "replan_expansions": 0,
            "total_expansions": initial_expansions,
            "replan_latencies_ms": [],
            "total_replan_time_ms": 0.0,
        }

    if start == goal:
        return {
            "status": "reached",
            "path_taken": path_taken,
            "travel_time": 0.0,
            "steps": 0,
            "replans": 0,
            "initial_expansions": initial_expansions,
            "replan_expansions": 0,
            "total_expansions": initial_expansions,
            "replan_latencies_ms": [],
            "total_replan_time_ms": 0.0,
        }

    remaining_path = init_path[1:]

    while current != goal:
        if t >= max_steps:
            return {
                "status": "max_steps",
                "path_taken": path_taken,
                "travel_time": travel_time,
                "steps": t,
                "replans": replans,
                "initial_expansions": initial_expansions,
                "replan_expansions": replan_expansions,
                "total_expansions": initial_expansions + replan_expansions,
                "replan_latencies_ms": replan_latencies_ms,
                "total_replan_time_ms": sum(replan_latencies_ms),
            }

        # Advance along the planned path
        next_node = remaining_path[0]
        remaining_path = remaining_path[1:]

        edge_cost = float(g[current][next_node].get("travel_time", 1.0))
        travel_time += edge_cost

        current = next_node
        t += 1
        path_taken.append(current)

        if current == goal:
            break

        # Apply events with step == t before choosing next move
        changed_edges = apply_events(g, events_list, step=t)

        # Check if next edge is blocked or if network costs altered
        next_edge_blocked = (
            len(remaining_path) > 0
            and math.isinf(float(g[current][remaining_path[0]].get("travel_time", 1.0)))
        )

        if changed_edges or next_edge_blocked:
            t_replan = time.perf_counter()
            new_path, new_cost, new_stats = plan(g, current, goal, heuristic)
            replan_ms = (time.perf_counter() - t_replan) * 1000.0

            replans += 1
            replan_expansions += new_stats["expansions"]
            replan_latencies_ms.append(round(replan_ms, 3))

            if verify:
                _verify_replan_with_dijkstra(g, current, goal, new_cost)

            if new_cost == float("inf") or not new_path:
                return {
                    "status": "stuck",
                    "path_taken": path_taken,
                    "travel_time": travel_time,
                    "steps": t,
                    "replans": replans,
                    "initial_expansions": initial_expansions,
                    "replan_expansions": replan_expansions,
                    "total_expansions": initial_expansions + replan_expansions,
                    "replan_latencies_ms": replan_latencies_ms,
                    "total_replan_time_ms": sum(replan_latencies_ms),
                }

            remaining_path = new_path[1:]

    return {
        "status": "reached",
        "path_taken": path_taken,
        "travel_time": travel_time,
        "steps": t,
        "replans": replans,
        "initial_expansions": initial_expansions,
        "replan_expansions": replan_expansions,
        "total_expansions": initial_expansions + replan_expansions,
        "replan_latencies_ms": replan_latencies_ms,
        "total_replan_time_ms": sum(replan_latencies_ms),
    }
