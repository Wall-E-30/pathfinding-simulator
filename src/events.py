"""Dynamic street network events and cost mutations.

Provides:
- Event: NamedTuple representing a dynamic graph change (closure or congestion).
- apply_events: Mutates a DiGraph in place and returns only edges with altered costs.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, List, NamedTuple, Tuple

import networkx as nx


class Event(NamedTuple):
    """Dynamic network event representing road closure or traffic congestion.

    Attributes:
        step: The move step t (>= 1) at which the event triggers.
        u: Source node ID.
        v: Destination node ID.
        new_cost: New travel time in seconds (float("inf") for closure).
        kind: Event category ("closure" or "congestion").
    """
    step: int
    u: Any
    v: Any
    new_cost: float
    kind: str


def apply_events(
    graph: nx.DiGraph,
    events: Iterable[Event],
    step: int,
) -> List[Tuple[Any, Any, float, float]]:
    """Apply dynamic events matching the given step to graph in place.

    Viva note:
        Returns ONLY edges whose costs actually changed (old_cost != new_cost).
        This allows incremental replanners (such as D* Lite) or reactive replanners
        (such as repeated A*) to avoid redundant computation if an event targets
        an edge that already has that cost.

    Args:
        graph: NetworkX DiGraph whose edge travel times will be mutated in place.
        events: Sequence of Event tuples.
        step: Current vehicle move counter t.

    Returns:
        List of tuples: (u, v, old_cost, new_cost) for all mutated edges.
    """
    changed_edges: List[Tuple[Any, Any, float, float]] = []

    for event in events:
        if event.step != step:
            continue

        if not graph.has_edge(event.u, event.v):
            continue

        old_cost = float(graph[event.u][event.v].get("travel_time", 1.0))
        new_cost = float(event.new_cost)

        # In Python: float("inf") != float("inf") evaluates to False.
        # For finite floats, check exact or close inequality.
        has_changed = False
        if math.isinf(old_cost) and math.isinf(new_cost):
            has_changed = False
        elif math.isinf(old_cost) != math.isinf(new_cost):
            has_changed = True
        elif not math.isclose(old_cost, new_cost, rel_tol=1e-9, abs_tol=1e-9):
            has_changed = True

        if has_changed:
            graph[event.u][event.v]["travel_time"] = new_cost
            changed_edges.append((event.u, event.v, old_cost, new_cost))

    return changed_edges

