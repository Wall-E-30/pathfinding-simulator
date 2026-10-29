"""Unit tests for D* Lite algorithm (initial planning phase)."""

from __future__ import annotations

import math
import random
import pytest
import networkx as nx

try:
    from tests.toy_graphs import make_grid, manhattan, random_grid_with_walls
except ModuleNotFoundError:
    from toy_graphs import make_grid, manhattan, random_grid_with_walls

from src.dstar_lite import DStarLite, plan


class TestDStarLiteInitialPlanning:
    """Tests for initial pathfinding using D* Lite on grid DiGraphs."""

    def test_hand_made_5x5_grid(self):
        """Test on a hand-made 5x5 grid with known obstacles and shortest path."""
        # Create an obstacle barrier in column 2 from row 0 to 3, leaving row 4 open
        # Forces detour around the bottom of the wall
        walls = {(2, 0), (2, 1), (2, 2), (2, 3)}
        graph = make_grid(n=5, walls=walls)

        start = (0, 0)
        goal = (4, 0)
        # Expected path must navigate down to (x, 4) and back up to (4, 0):
        # (0,0) -> (1,0) -> ... down to (1,4) -> (2,4) -> (3,4) -> up to (4,0)
        expected_cost = nx.shortest_path_length(graph, start, goal, weight="travel_time")

        path, cost, stats = plan(graph, start, goal, heuristic=manhattan)

        assert math.isclose(cost, expected_cost), f"Expected cost {expected_cost}, got {cost}"
        assert path[0] == start
        assert path[-1] == goal
        assert isinstance(stats["expansions"], int)
        assert stats["expansions"] > 0
        assert stats["time_ms"] >= 0.0

    def test_100_random_grids_match_dijkstra(self):
        """Test that returned cost equals networkx.shortest_path_length across 100 random grids."""
        for seed in range(100):
            try:
                graph = random_grid_with_walls(seed=seed, n=10)
                rng = random.Random(seed)

                # Find components with at least 2 connected nodes to guarantee a path
                components = [
                    c for c in nx.connected_components(graph.to_undirected())
                    if len(c) >= 2
                ]
                if not components:
                    continue

                largest_component = sorted(max(components, key=len))
                start, goal = rng.sample(largest_component, 2)

                expected_cost = nx.shortest_path_length(graph, start, goal, weight="travel_time")
                path, cost, stats = plan(graph, start, goal, heuristic=manhattan)

                assert math.isclose(cost, expected_cost), (
                    f"Cost mismatch on seed {seed}: got {cost}, expected {expected_cost}"
                )
                assert len(path) >= 2
                assert path[0] == start
                assert path[-1] == goal
            except Exception:
                print(f"D* Lite failed on seed {seed}")
                raise

    def test_unreachable_goal_returns_empty_and_inf(self):
        """Test that a completely blocked off goal returns an empty path and float('inf')."""
        # Block all incoming/outgoing neighbors of (4, 4) in a 5x5 grid
        walls = {(3, 4), (4, 3)}
        graph = make_grid(n=5, walls=walls)

        start = (0, 0)
        goal = (4, 4)
        # Node (4, 4) was omitted from the graph, so let's add it isolated without edges
        graph.add_node(goal, x=4.0, y=4.0)

        path, cost, stats = plan(graph, start, goal, heuristic=manhattan)

        assert path == []
        assert cost == float("inf")
        assert isinstance(stats["expansions"], int)
        assert stats["time_ms"] >= 0.0

    def test_path_validity_and_cost_sum(self):
        """Test that returned path uses valid edges and summed edge costs equal returned cost."""
        graph = random_grid_with_walls(seed=42, n=10)
        components = [
            c for c in nx.connected_components(graph.to_undirected())
            if len(c) >= 5
        ]
        largest_comp = sorted(max(components, key=len))
        start, goal = largest_comp[0], largest_comp[-1]

        path, cost, stats = plan(graph, start, goal, heuristic=manhattan)

        assert len(path) >= 2
        assert path[0] == start
        assert path[-1] == goal

        # Verify edge continuity and accumulate weights
        summed_weight = 0.0
        for u, v in zip(path[:-1], path[1:]):
            assert graph.has_edge(u, v), f"Edge ({u}, {v}) does not exist in graph"
            summed_weight += graph[u][v]["travel_time"]

        assert math.isclose(summed_weight, cost), (
            f"Summed edge weight {summed_weight} does not equal returned cost {cost}"
        )

    def test_zero_heuristic_vs_manhattan_give_same_cost(self):
        """Run once with zero heuristic and once with Manhattan; both must produce the same cost."""
        graph = random_grid_with_walls(seed=7, n=10)
        components = [
            c for c in nx.connected_components(graph.to_undirected())
            if len(c) >= 5
        ]
        largest_comp = sorted(max(components, key=len))
        start, goal = largest_comp[0], largest_comp[-1]

        path_zero, cost_zero, stats_zero = plan(
            graph, start, goal, heuristic=lambda n, g: 0.0
        )
        path_manh, cost_manh, stats_manh = plan(
            graph, start, goal, heuristic=manhattan
        )

        assert math.isclose(cost_zero, cost_manh), (
            f"Cost mismatch: zero heuristic={cost_zero}, manhattan={cost_manh}"
        )
        assert stats_zero["expansions"] >= stats_manh["expansions"], (
            "Manhattan heuristic should not expand more nodes than zero heuristic"
        )

    def test_start_equals_goal(self):
        """When start == goal, return ([start], 0.0, stats) as required by interface contract."""
        graph = make_grid(n=5)
        start = (2, 2)
        goal = (2, 2)

        path, cost, stats = plan(graph, start, goal, heuristic=manhattan)

        assert path == [start]
        assert cost == 0.0
        assert isinstance(stats["expansions"], int)
        assert stats["time_ms"] >= 0.0

    def test_100_random_weighted_grids_match_dijkstra(self):
        """On 100 random grids with non-uniform weights (1 to 10), cost must match Dijkstra within 1e-6."""
        for seed in range(100):
            try:
                graph = random_grid_with_walls(seed=seed, n=10)
                rng = random.Random(seed)

                # Assign random non-uniform travel times between 1.0 and 10.0 seconds
                for u, v in graph.edges():
                    graph[u][v]["travel_time"] = rng.uniform(1.0, 10.0)

                components = [
                    c for c in nx.connected_components(graph.to_undirected())
                    if len(c) >= 2
                ]
                if not components:
                    continue

                largest_component = sorted(max(components, key=len))
                start, goal = rng.sample(largest_component, 2)

                expected_cost = nx.shortest_path_length(
                    graph, start, goal, weight="travel_time"
                )
                path, cost, stats = plan(graph, start, goal, heuristic=manhattan)

                assert abs(cost - expected_cost) <= 1e-6, (
                    f"Cost mismatch on weighted seed {seed}: got {cost}, expected {expected_cost}"
                )
                assert len(path) >= 2
                assert path[0] == start
                assert path[-1] == goal
            except Exception:
                print(f"Weighted D* Lite failed on seed {seed}")
                raise


