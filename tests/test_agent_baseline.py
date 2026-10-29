"""Unit tests for the baseline dynamic replanning agent in src/agent_baseline.py."""

from __future__ import annotations

import math
import random
import networkx as nx
import pytest

from src.agent_baseline import run_baseline_agent
from src.astar import plan
from src.events import Event
from src.simulator import generate_events, make_scenario
from tests.toy_graphs import make_grid, manhattan


@pytest.fixture
def detour_graph() -> nx.DiGraph:
    """Diamond graph with primary path 0 -> 1 -> 3 (cost 20) and detour 0 -> 2 -> 3 (cost 30).
    Includes backtracking edge 1 -> 0 so vehicle can divert if (1, 3) is closed.
    """
    g = nx.DiGraph()
    g.add_node(0, x=0.0, y=0.0)
    g.add_node(1, x=1.0, y=1.0)
    g.add_node(2, x=1.0, y=-1.0)
    g.add_node(3, x=2.0, y=0.0)

    g.add_edge(0, 1, travel_time=10.0)
    g.add_edge(1, 3, travel_time=10.0)
    g.add_edge(1, 0, travel_time=10.0)  # Backtracking link

    g.add_edge(0, 2, travel_time=15.0)
    g.add_edge(2, 3, travel_time=15.0)
    return g


class TestBaselineAgent:
    """Tests for run_baseline_agent."""

    def test_closure_forces_detour_without_traversing_inf(self, detour_graph):
        """1. Closure on planned route forces detour: reaches goal, no inf edge traversed, replans >= 1."""
        events = [Event(step=1, u=1, v=3, new_cost=float("inf"), kind="closure")]

        result = run_baseline_agent(
            detour_graph, start=0, goal=3, events=events, heuristic=None, verify=True
        )

        assert result["status"] == "reached"
        assert result["path_taken"] == [0, 1, 0, 2, 3]
        assert result["replans"] >= 1
        assert len(result["replan_latencies_ms"]) == result["replans"]

        # Verify no inf edge was traversed
        assert result["travel_time"] == 10.0 + 10.0 + 15.0 + 15.0  # 50.0
        assert not math.isinf(result["travel_time"])

    def test_no_events_zero_replans_and_optimal_cost(self, detour_graph):
        """2. No events: replans == 0 and travel_time equals initial optimal cost."""
        result = run_baseline_agent(
            detour_graph, start=0, goal=3, events=[], heuristic=None, verify=True
        )

        assert result["status"] == "reached"
        assert result["replans"] == 0
        assert result["replan_expansions"] == 0
        assert result["replan_latencies_ms"] == []
        assert result["total_replan_time_ms"] == 0.0
        assert result["path_taken"] == [0, 1, 3]
        assert result["travel_time"] == 20.0
        assert result["steps"] == 2

    def test_closure_cutting_off_goal_gives_stuck(self):
        """3. A closure that cuts off the goal gives status 'stuck'."""
        g = nx.DiGraph()
        g.add_node(0, x=0.0, y=0.0)
        g.add_node(1, x=1.0, y=0.0)
        g.add_node(2, x=2.0, y=0.0)
        g.add_edge(0, 1, travel_time=5.0)
        g.add_edge(1, 2, travel_time=5.0)

        events = [Event(step=1, u=1, v=2, new_cost=float("inf"), kind="closure")]

        result = run_baseline_agent(g, start=0, goal=2, events=events, heuristic=None, verify=True)

        assert result["status"] == "stuck"
        assert result["path_taken"] == [0, 1]
        assert result["replans"] == 1

    def test_path_taken_is_valid_walk_and_cost_matches(self, detour_graph):
        """4. path_taken is a valid walk of existing edges, and travel_time equals sum of edge costs."""
        events = [Event(step=1, u=1, v=3, new_cost=float("inf"), kind="closure")]

        result = run_baseline_agent(
            detour_graph, start=0, goal=3, events=events, heuristic=None, verify=False
        )

        path = result["path_taken"]
        assert len(path) >= 2
        assert result["steps"] == len(path) - 1

        for u, v in zip(path[:-1], path[1:]):
            assert detour_graph.has_edge(u, v)

    def test_caller_graph_unchanged_afterwards(self, detour_graph):
        """5. The caller's graph is unchanged afterwards."""
        pre_edges = {(u, v): dict(d) for u, v, d in detour_graph.edges(data=True)}
        events = [Event(step=1, u=1, v=3, new_cost=float("inf"), kind="closure")]

        run_baseline_agent(detour_graph, start=0, goal=3, events=events, heuristic=None)

        post_edges = {(u, v): dict(d) for u, v, d in detour_graph.edges(data=True)}
        assert pre_edges == post_edges

    def test_50_random_scenarios_on_20x20_grid_with_verify(self):
        """6. 50 random scenarios (seeded) on a 20x20 weighted grid, with verify=True."""
        grid = make_grid(n=20)
        # Add varied deterministic travel times across edges
        weight_rng = random.Random(42)
        for u, v in grid.edges():
            grid[u][v]["travel_time"] = round(weight_rng.uniform(1.0, 4.0), 2)

        for i in range(50):
            scenario_seed = 1000 + i
            start, goal, events = make_scenario(
                grid, heuristic=manhattan, seed=scenario_seed, n_events=4, min_hops=15
            )

            result = run_baseline_agent(
                grid, start=start, goal=goal, events=events, heuristic=manhattan, verify=True
            )

            assert result["status"] in {"reached", "stuck"}
            assert result["travel_time"] >= 0.0
            assert result["steps"] >= 0
            assert result["total_expansions"] == result["initial_expansions"] + result["replan_expansions"]
            assert len(result["replan_latencies_ms"]) == result["replans"]
            assert math.isclose(result["total_replan_time_ms"], sum(result["replan_latencies_ms"]), rel_tol=1e-5, abs_tol=1e-5)
