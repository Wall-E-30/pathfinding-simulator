"""Baseline dynamic routing agent that recomputes paths from scratch using A*."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

import networkx as nx

from src.astar import plan
from src.events import Event, apply_events


def _verify_with_dijkstra(g: nx.DiGraph, cur: Any, goal: Any, planner_cost: float) -> None:
    """Verify that planner cost matches ground-truth Dijkstra on non-inf edges.

    Viva note:
        Filters out inf (closed) edges from the graph view and runs independent
        NetworkX Dijkstra from the current node. Asserts that the replanned cost
        is optimal or both agree the goal is unreachable.
    """
    valid_edges = [
        (u, v, d) for u, v, d in g.edges(data=True)
        if not math.isinf(float(d.get("travel_time", 1.0)))
    ]
    diag_g = nx.DiGraph()
    diag_g.add_nodes_from(g.nodes(data=True))
    diag_g.add_edges_from(valid_edges)

    try:
        nx_cost = nx.shortest_path_length(diag_g, cur, goal, weight="travel_time")
        nx_reachable = True
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        nx_cost = float("inf")
        nx_reachable = False

    if nx_reachable:
        assert math.isclose(planner_cost, nx_cost, rel_tol=1e-5, abs_tol=1e-6), (
            f"Dijkstra verification failed: planner cost {planner_cost} != Dijkstra cost {nx_cost}"
        )
    else:
        assert math.isinf(planner_cost), (
            f"Dijkstra verification failed: Dijkstra found unreachable, but planner got {planner_cost}"
        )


def run_baseline_agent(
    graph: nx.DiGraph,
    start: Any,
    goal: Any,
    events: Iterable[Event],
    heuristic: Optional[Callable[[Any, Any], float]] = None,
    verify: bool = False,
) -> Dict[str, Any]:
    """Execute dynamic path traversal by recomputing from scratch on every network change.

    Policy:
    1. Initial plan via astar.plan(G, start, goal, heuristic). Expansions go into initial_expansions.
    2. Loop until vehicle is at goal:
       - apply_events(G, events, t). If it returns changes, re-run astar.plan from the
         CURRENT node (full recompute), count one replan, record expansions and latency.
       - If no changes, keep following stored path.
       - If replan returns cost inf, set status "stuck" and stop.
       - Move to next node on path. Add travel_time (must be finite).
    3. With verify=True, run Dijkstra cross-check after every replan.

    Viva notes:
        1. Full Recomputation Baseline: Repeatedly solves the static A* problem from scratch
           upon detecting any network change. Serves as the primary performance benchmark against
           incremental repair algorithms (e.g., D* Lite).
        2. Immediate Sensing & Safety: Moves are only made after events at step t are applied,
           ensuring the vehicle never traverses a road that just closed.
        3. Immutability: Uses graph.copy() so the caller's graph is completely unaffected.

    Args:
        graph: NetworkX DiGraph representing road network.
        start: Starting node ID.
        goal: Destination node ID.
        events: Sequence of Event objects.
        heuristic: Optional admissible heuristic callable h(n, goal) in seconds.
        verify: If True, cross-checks every replan against Dijkstra.

    Returns:
        Result dictionary containing exact contract keys.
    """
    g = graph.copy()
    events_list = list(events)
    max_steps = 5 * g.number_of_nodes()

    # Initial planning at t = 0
    init_path, init_cost, init_stats = plan(g, start, goal, heuristic)
    initial_expansions = init_stats["expansions"]
    replan_expansions = 0
    replans = 0
    replan_latencies_ms: List[float] = []

    cur = start
    path_taken: List[Any] = [cur]
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

    stored_path = init_path[1:]

    while cur != goal:
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

        # Apply events at current step t before choosing next move
        changes = apply_events(g, events_list, step=t)

        if changes:
            t_replan = time.perf_counter()
            new_path, new_cost, new_stats = plan(g, cur, goal, heuristic)
            replan_ms = (time.perf_counter() - t_replan) * 1000.0

            replans += 1
            replan_expansions += new_stats["expansions"]
            replan_latencies_ms.append(round(replan_ms, 3))

            if verify:
                _verify_with_dijkstra(g, cur, goal, new_cost)

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

            stored_path = new_path[1:]

        if not stored_path:
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

        next_node = stored_path[0]
        stored_path = stored_path[1:]

        edge_cost = float(g[cur][next_node].get("travel_time", 1.0))
        assert not math.isinf(edge_cost), f"Attempted to traverse blocked edge ({cur}, {next_node})"
        travel_time += edge_cost

        cur = next_node
        t += 1
        path_taken.append(cur)

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

