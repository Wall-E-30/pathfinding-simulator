"""Dynamic routing agent powered by incremental D* Lite replanning."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, Iterable, List, Optional
import networkx as nx

from src.dstar_lite import DStarLite
from src.events import Event, apply_events


def _verify_with_dijkstra(g: nx.DiGraph, cur: Any, goal: Any, planner_cost: float) -> None:
    """Verify that planner cost matches ground-truth Dijkstra on non-inf edges.

    Filters out closed (inf) edges and runs independent NetworkX Dijkstra
    from current vehicle position.
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


def run_dstar_agent(
    graph: nx.DiGraph,
    start: Any,
    goal: Any,
    events: Iterable[Event],
    heuristic: Optional[Callable[[Any, Any], float]] = None,
    verify: bool = False,
) -> Dict[str, Any]:
    """Execute dynamic path traversal using D* Lite incremental replanning.

    Policy:
    1. G = graph.copy(); planner = DStarLite(G, start, goal, heuristic).
       planner.compute_shortest_path(). Expansions go in initial_expansions.
    2. Until the vehicle is at the goal:
       - changes = apply_events(G, events, t).
       - If there are any changes: time (perf_counter) planner.notify_changes(changes),
         count one replan, record expansions delta and latency in ms.
       - If verify=True, assert planner.cost_to_goal equals fresh Dijkstra (inf edges removed).
       - n = planner.next_step(). If None, set status "stuck" and stop.
       - Add G[cur][n]["travel_time"] to travel_time (must be finite).
       - Move to n, call planner.move_start(n), increment t.

    Viva notes:
        1. Incremental Repair: Unlike the baseline A* agent which re-solves the whole
           shortest path tree from scratch on every change, D* Lite only expands the
           local subgraph whose g/rhs consistency was disturbed by modified edges.
        2. Immutability: Operates on graph.copy() so the caller's graph is unaltered.
        3. Immediate Sensing: Events at step t trigger when vehicle reaches its t-th
           position, before it decides its next move.

    Args:
        graph: NetworkX DiGraph representing road network.
        start: Starting node ID.
        goal: Destination node ID.
        events: Sequence of Event tuples.
        heuristic: Optional admissible heuristic callable h(n, goal) in seconds.
        verify: If True, cross-checks cost-to-goal against Dijkstra after every replan.

    Returns:
        Result dictionary containing exact contract keys.
    """
    g = graph.copy()
    events_list = list(events)
    max_steps = 5 * g.number_of_nodes()

    # Initial planning at t = 0
    planner = DStarLite(g, start, goal, heuristic=heuristic)
    planner.initialize()
    planner.compute_shortest_path()

    initial_expansions = planner.expansions
    replan_expansions = 0
    replans = 0
    replan_latencies_ms: List[float] = []

    cur = start
    path_taken: List[Any] = [cur]
    travel_time: float = 0.0
    t: int = 0

    if start not in g or goal not in g or planner.cost_to_goal == float("inf"):
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
            delta_exp = planner.notify_changes(changes)
            replan_ms = (time.perf_counter() - t_replan) * 1000.0

            replans += 1
            replan_expansions += delta_exp
            replan_latencies_ms.append(round(replan_ms, 3))

            if verify:
                _verify_with_dijkstra(g, cur, goal, planner.cost_to_goal)

        # Select next step along optimal successor
        nxt = planner.next_step()

        if nxt is None:
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

        edge_cost = float(g[cur][nxt]["travel_time"])
        assert not math.isinf(edge_cost), f"Attempted to traverse blocked edge ({cur}, {nxt})"
        travel_time += edge_cost

        cur = nxt
        planner.move_start(cur)
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
