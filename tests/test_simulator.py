"""Unit tests for event generation and scenario builder in src/simulator.py."""

from __future__ import annotations

import math
import networkx as nx
import pytest

from src.astar import plan
from src.events import Event, apply_events
from src.simulator import generate_events, make_scenario
from tests.toy_graphs import make_grid, manhattan


@pytest.fixture
def grid_15x15() -> nx.DiGraph:
    """15x15 4-connected grid graph with unit travel times and coordinates."""
    return make_grid(n=15)


@pytest.fixture
def sample_path(grid_15x15: nx.DiGraph) -> list[tuple[int, int]]:
    """Sample path across the 15x15 grid from (0, 0) to (14, 14)."""
    path, cost, _ = plan(grid_15x15, (0, 0), (14, 14), heuristic=manhattan)
    assert len(path) == 29
    return path


class TestSimulatorEvents:
    """Tests for generate_events function."""

    def test_generate_events_determinism(self, grid_15x15, sample_path):
        """Same seed gives identical events; different seed gives different events."""
        events_seed42_a = generate_events(grid_15x15, sample_path, seed=42)
        events_seed42_b = generate_events(grid_15x15, sample_path, seed=42)
        events_seed99 = generate_events(grid_15x15, sample_path, seed=99)

        assert events_seed42_a == events_seed42_b
        assert events_seed42_a != events_seed99

    def test_generate_events_graph_immutability(self, grid_15x15, sample_path):
        """The input graph must be completely unchanged after calling generate_events."""
        # Record pre-call edge attributes
        pre_edges = {(u, v): dict(data) for u, v, data in grid_15x15.edges(data=True)}

        generate_events(grid_15x15, sample_path, seed=123, n_events=8)

        post_edges = {(u, v): dict(data) for u, v, data in grid_15x15.edges(data=True)}
        assert pre_edges == post_edges

    def test_events_properties_and_ranges(self, grid_15x15, sample_path):
        """Verify sorting, edge existence, uniqueness, steps, closures, and congestion bounds."""
        n_events = 6
        events = generate_events(
            grid_15x15, sample_path, seed=777, n_events=n_events, closure_fraction=0.4
        )

        assert len(events) == n_events

        # 1. Sorted by step (non-decreasing)
        for i in range(len(events) - 1):
            assert events[i].step <= events[i + 1].step

        # 2. At most one event per edge
        targeted_edges = [(e.u, e.v) for e in events]
        assert len(targeted_edges) == len(set(targeted_edges))

        # 3. Every edge exists in the graph
        for e in events:
            assert grid_15x15.has_edge(e.u, e.v)

        # 4. Closures have new_cost = inf; Congestion is within [1.5, 5.0] * base_cost
        for e in events:
            base_cost = grid_15x15[e.u][e.v]["travel_time"]
            if e.kind == "closure":
                assert math.isinf(e.new_cost)
            elif e.kind == "congestion":
                assert 1.5 * base_cost <= e.new_cost <= 5.0 * base_cost + 1e-6
            else:
                pytest.fail(f"Unexpected event kind: {e.kind}")

    def test_path_feasibility_after_events(self, grid_15x15, sample_path):
        """After applying all events, every node of path must still reach the goal."""
        events = generate_events(
            grid_15x15, sample_path, seed=55, n_events=8, closure_fraction=0.5
        )

        # Apply events to a copy of the graph
        h = grid_15x15.copy()
        for e in events:
            h[e.u][e.v]["travel_time"] = e.new_cost

        # Remove inf edges
        blocked = [(u, v) for u, v, d in h.edges(data=True) if math.isinf(d["travel_time"])]
        h.remove_edges_from(blocked)

        goal = sample_path[-1]
        reachable = nx.ancestors(h, goal) | {goal}

        for node in sample_path:
            assert node in reachable, f"Node {node} on path cannot reach goal {goal} after events"


class TestMakeScenario:
    """Tests for make_scenario function."""

    def test_make_scenario_determinism_and_hops(self):
        """make_scenario is deterministic and returns path with at least min_hops."""
        # 20x20 grid allows paths with >= 30 hops
        grid_20x20 = make_grid(n=20)

        start1, goal1, events1 = make_scenario(
            grid_20x20, heuristic=manhattan, seed=101, n_events=6, min_hops=30
        )
        start2, goal2, events2 = make_scenario(
            grid_20x20, heuristic=manhattan, seed=101, n_events=6, min_hops=30
        )

        assert (start1, goal1, events1) == (start2, goal2, events2)

        # Verify that path from start to goal has at least 30 hops
        path, cost, _ = plan(grid_20x20, start1, goal1, heuristic=manhattan)
        assert len(path) >= 30
        assert len(events1) == 6

    def test_make_scenario_raises_if_hops_unreachable(self):
        """make_scenario raises ValueError if min_hops cannot be satisfied within 200 tries."""
        small_grid = make_grid(n=3)  # Max path length in 3x3 grid is 5 nodes
        with pytest.raises(ValueError, match="Could not find a path with at least 50 hops"):
            make_scenario(small_grid, heuristic=manhattan, seed=42, min_hops=50)
