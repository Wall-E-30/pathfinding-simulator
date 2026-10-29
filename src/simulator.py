"""Dynamic traffic event generation and benchmark scenario creation.

Provides:
- generate_events: Generates reproducible traffic closures and congestions along a path.
- make_scenario: Finds candidate start/goal pairs meeting hop constraints and generates events.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, List, Optional, Set, Tuple

import networkx as nx

from src.astar import plan
from src.events import Event


def _is_path_feasible(graph: nx.DiGraph, path: List[Any], events: List[Event]) -> bool:
    """Check if all nodes along path can reach the goal given the events.

    Viva note:
        nx.ancestors(H, goal) returns all nodes with a valid directed path leading
        into goal. Unioning with {goal} ensures that if any closures cut off a path node
        completely, we detect it and can convert the closure to congestion.
    """
    h = graph.copy()
    for ev in events:
        if h.has_edge(ev.u, ev.v):
            h[ev.u][ev.v]["travel_time"] = ev.new_cost

    # Remove blocked (inf) edges
    blocked = [
        (u, v) for u, v, d in h.edges(data=True)
        if math.isinf(float(d.get("travel_time", 1.0)))
    ]
    h.remove_edges_from(blocked)

    goal = path[-1]
    try:
        can_reach_goal = nx.ancestors(h, goal) | {goal}
    except (nx.NetworkXError, KeyError):
        return False

    return all(node in can_reach_goal for node in path)


def generate_events(
    graph: nx.DiGraph,
    path: List[Any],
    seed: int,
    n_events: int = 6,
    closure_fraction: float = 0.4,
    decoy_fraction: float = 0.25,
) -> List[Event]:
    """Generate dynamic traffic events along a vehicle's path and across the network.

    Viva notes:
        1. Local PRNG: rng = random.Random(seed) ensures exact determinism across simulation
           runs without modifying or depending on the global random state.
        2. Edge lookahead (i >= s): Edge i connects path[i] -> path[i+1]. At step s,
           the vehicle is located at path[s]. Requiring i >= s guarantees the event
           occurs on an edge the vehicle has not yet traversed.
        3. Feasibility Repair: If a closure cuts off all alternate paths to the goal,
           we convert the latest closure into a congestion delay so the problem is always
           solvable for dynamic benchmark comparisons.

    Args:
        graph: NetworkX DiGraph (never mutated).
        path: Initial optimal path as a sequence of node IDs.
        seed: Random seed for deterministic generation.
        n_events: Total number of events to produce.
        closure_fraction: Ratio of events that close roads (new_cost = inf).
        decoy_fraction: Ratio of events placed on random graph edges outside the path.

    Returns:
        List of Event objects sorted stably by step.
    """
    rng = random.Random(seed)

    if n_events <= 0:
        return []

    n_decoys = int(round(n_events * decoy_fraction))
    n_path_events = n_events - n_decoys

    used_edges: Set[Tuple[Any, Any]] = set()
    selected_events_info: List[Tuple[int, Any, Any]] = []  # (step, u, v)

    # 1. Path events: choose step s in 1..len(path)-2 and edge index i >= s
    if len(path) >= 3:
        candidate_pairs: List[Tuple[int, int]] = [
            (s, i)
            for s in range(1, len(path) - 1)
            for i in range(s, len(path) - 1)
        ]
        rng.shuffle(candidate_pairs)

        for s, i in candidate_pairs:
            edge = (path[i], path[i + 1])
            if edge not in used_edges and graph.has_edge(*edge):
                used_edges.add(edge)
                selected_events_info.append((s, edge[0], edge[1]))
                if len(selected_events_info) == n_path_events:
                    break

    # 2. Decoy events: random edges anywhere in the graph
    remaining_events = n_events - len(selected_events_info)
    all_other_edges = [
        (u, v) for u, v in graph.edges()
        if (u, v) not in used_edges
    ]
    rng.shuffle(all_other_edges)

    max_step = max(1, len(path) - 2) if len(path) >= 3 else 1
    for u, v in all_other_edges:
        if len(selected_events_info) == n_events:
            break
        s = rng.randint(1, max_step)
        used_edges.add((u, v))
        selected_events_info.append((s, u, v))

    # 3. Assign closure vs congestion kinds
    n_closures = int(round(n_events * closure_fraction))
    n_congestions = n_events - n_closures
    kinds = ["closure"] * n_closures + ["congestion"] * n_congestions
    rng.shuffle(kinds)

    events: List[Event] = []
    for (s, u, v), kind in zip(selected_events_info, kinds):
        base_cost = float(graph[u][v].get("travel_time", 1.0))
        if kind == "closure":
            new_cost = float("inf")
        else:
            new_cost = base_cost * rng.uniform(1.5, 5.0)

        events.append(Event(step=s, u=u, v=v, new_cost=new_cost, kind=kind))

    # 4. Sort events stably by step
    events.sort(key=lambda e: e.step)

    # 5. Feasibility repair: if closures isolate any path node, convert last closure to congestion
    while not _is_path_feasible(graph, path, events):
        closure_indices = [idx for idx, e in enumerate(events) if e.kind == "closure"]
        if not closure_indices:
            break
        last_idx = closure_indices[-1]
        ev = events[last_idx]
        base_cost = float(graph[ev.u][ev.v].get("travel_time", 1.0))
        new_cost = base_cost * rng.uniform(1.5, 5.0)
        events[last_idx] = Event(step=ev.step, u=ev.u, v=ev.v, new_cost=new_cost, kind="congestion")

    return events


def make_scenario(
    graph: nx.DiGraph,
    heuristic: Optional[Callable[[Any, Any], float]] = None,
    seed: int = 42,
    n_events: int = 6,
    min_hops: int = 30,
) -> Tuple[Any, Any, List[Event]]:
    """Generate a valid origin-destination scenario with a minimum hop count and traffic events.

    Viva note:
        Guarantees non-trivial paths with length >= min_hops so that dynamic rerouting
        algorithms have sufficient runway to experience multiple disruptions and replan.

    Args:
        graph: NetworkX DiGraph.
        heuristic: Optional admissible heuristic callable for A*.
        seed: PRNG seed for reproducible pairing.
        n_events: Number of traffic events to synthesize.
        min_hops: Minimum node count required on the initial path.

    Returns:
        Tuple of (start_node, goal_node, events_list).

    Raises:
        ValueError: If no path with >= min_hops is found within 200 attempts.
    """
    rng = random.Random(seed)
    nodes = list(graph.nodes())

    if len(nodes) < 2:
        raise ValueError("Graph has fewer than 2 nodes to form a path.")

    MAX_ATTEMPTS = 200
    for _ in range(MAX_ATTEMPTS):
        start = rng.choice(nodes)
        goal = rng.choice(nodes)
        if start == goal:
            continue

        path, cost, _ = plan(graph, start, goal, heuristic=heuristic)
        if cost < float("inf") and len(path) >= min_hops:
            events = generate_events(graph, path, seed=seed, n_events=n_events)
            return (start, goal, events)

    raise ValueError(
        f"Could not find a path with at least {min_hops} hops after {MAX_ATTEMPTS} attempts."
    )
