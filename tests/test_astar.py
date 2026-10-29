"""Unit tests for A* and Dijkstra pathfinding on road network DiGraphs."""

from __future__ import annotations

import math
import random
from typing import Any

import networkx as nx
import pytest

from src.astar import plan
from src.graph_utils import compute_v_max, haversine_m


@pytest.fixture(scope="module")
def grid_network() -> nx.DiGraph:
    """Fixture providing an 8x8 directed grid street network with coordinates and travel times.

    Viva note:
        A regular grid network with cross-streets provides multiple alternate paths
        between any two points, making it an ideal benchmark to evaluate search efficiency
        and compare A* node expansions against Dijkstra.
    """
    g = nx.DiGraph()
    rows, cols = 8, 8

    # Base coordinates: latitude ~23.0, longitude ~72.5, spaced ~0.002 deg (~220m)
    for r in range(rows):
        for c in range(cols):
            node_id = r * cols + c
            lat = 23.000 + (r * 0.002)
            lon = 72.500 + (c * 0.002)
            g.add_node(node_id, x=lon, y=lat)

    # Add horizontal and vertical directed edges
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            # Right neighbor
            if c + 1 < cols:
                v = r * cols + (c + 1)
                dist_m = haversine_m(g.nodes[u]["y"], g.nodes[u]["x"], g.nodes[v]["y"], g.nodes[v]["x"])
                # Travel speed ~40 km/h (11.11 m/s)
                travel_time = dist_m / (40.0 / 3.6)
                g.add_edge(u, v, travel_time=travel_time, length=dist_m, speed_kph=40.0)
                g.add_edge(v, u, travel_time=travel_time, length=dist_m, speed_kph=40.0)

            # Upward neighbor
            if r + 1 < rows:
                v = (r + 1) * cols + c
                dist_m = haversine_m(g.nodes[u]["y"], g.nodes[u]["x"], g.nodes[v]["y"], g.nodes[v]["x"])
                travel_time = dist_m / (40.0 / 3.6)
                g.add_edge(u, v, travel_time=travel_time, length=dist_m, speed_kph=40.0)
                g.add_edge(v, u, travel_time=travel_time, length=dist_m, speed_kph=40.0)

    return g


@pytest.fixture(scope="module")
def grid_heuristic(grid_network: nx.DiGraph):
    """Admissible heuristic function for grid_network."""
    v_max = compute_v_max(grid_network)
    coords = {n: (grid_network.nodes[n]["y"], grid_network.nodes[n]["x"]) for n in grid_network.nodes()}

    def h(n: Any, goal: Any) -> float:
        if n == goal:
            return 0.0
        lat_n, lon_n = coords[n]
        lat_g, lon_g = coords[goal]
        return haversine_m(lat_n, lon_n, lat_g, lon_g) / v_max

    return h


class TestAStarCorrectness:
    """Tests evaluating A* and Dijkstra against ground-truth NetworkX shortest paths."""

    def test_20_random_pairs_cost_equals_networkx(self, grid_network, grid_heuristic):
        """On 20 random (start, goal) pairs, cost must match networkx.shortest_path_length within 1e-6."""
        rng = random.Random(42)
        nodes = list(grid_network.nodes())
        pairs_tested = 0

        while pairs_tested < 20:
            s = rng.choice(nodes)
            g = rng.choice(nodes)
            if s == g:
                continue

            expected_cost = nx.shortest_path_length(grid_network, s, g, weight="travel_time")

            # Test A* with heuristic
            astar_path, astar_cost, astar_stats = plan(grid_network, s, g, heuristic=grid_heuristic)
            assert math.isclose(astar_cost, expected_cost, abs_tol=1e-6)

            # Test Dijkstra (heuristic=None)
            dijk_path, dijk_cost, dijk_stats = plan(grid_network, s, g, heuristic=None)
            assert math.isclose(dijk_cost, expected_cost, abs_tol=1e-6)

            pairs_tested += 1

    def test_path_validity_and_weight_summation(self, grid_network, grid_heuristic):
        """Every returned path must consist of valid consecutive edges and cost must match edge weights."""
        rng = random.Random(101)
        nodes = list(grid_network.nodes())

        for _ in range(10):
            s = rng.choice(nodes)
            g = rng.choice(nodes)
            if s == g:
                continue

            path, cost, stats = plan(grid_network, s, g, heuristic=grid_heuristic)

            # Path must start at s and end at g
            assert path[0] == s
            assert path[-1] == g

            # Path must be connected and edges must exist
            summed_cost = 0.0
            for u, v in zip(path[:-1], path[1:]):
                assert grid_network.has_edge(u, v)
                summed_cost += grid_network[u][v]["travel_time"]

            assert math.isclose(cost, summed_cost, rel_tol=1e-6)

    def test_astar_expands_fewer_nodes_than_dijkstra_on_average(self, grid_network, grid_heuristic):
        """A* must explore fewer or equal nodes than uninformed Dijkstra on average."""
        rng = random.Random(202)
        nodes = list(grid_network.nodes())
        astar_expansions_total = 0
        dijkstra_expansions_total = 0
        num_pairs = 20

        pairs_tested = 0
        while pairs_tested < num_pairs:
            s = rng.choice(nodes)
            g = rng.choice(nodes)
            if s == g:
                continue

            _, _, astar_stats = plan(grid_network, s, g, heuristic=grid_heuristic)
            _, _, dijk_stats = plan(grid_network, s, g, heuristic=None)

            astar_expansions_total += astar_stats["expansions"]
            dijkstra_expansions_total += dijk_stats["expansions"]
            pairs_tested += 1

        avg_astar = astar_expansions_total / num_pairs
        avg_dijkstra = dijkstra_expansions_total / num_pairs

        # A* directed search must prune exploration space compared to Dijkstra
        assert avg_astar < avg_dijkstra, f"Expected A* ({avg_astar}) < Dijkstra ({avg_dijkstra})"


class TestAStarEdgeCases:
    """Tests evaluating edge cases: start == goal, unreachable nodes, and nonexistent nodes."""

    def test_start_equals_goal(self, grid_network):
        """When start == goal, plan must return ([start], 0.0, stats) with 0 expansions."""
        node = 5
        path, cost, stats = plan(grid_network, node, node)

        assert path == [node]
        assert cost == 0.0
        assert stats["expansions"] == 0
        assert "time_ms" in stats and stats["time_ms"] >= 0.0

    def test_unreachable_goal(self, grid_network):
        """When goal is unreachable, plan must return ([], inf, stats)."""
        # Create an isolated disconnected node
        g = grid_network.copy()
        isolated_node = 999
        g.add_node(isolated_node, x=72.99, y=23.99)

        path, cost, stats = plan(g, 0, isolated_node)

        assert path == []
        assert cost == float("inf")
        assert stats["expansions"] > 0
        assert "time_ms" in stats

    def test_nonexistent_node(self, grid_network):
        """If start or goal does not exist in graph, return ([], inf, stats)."""
        path, cost, stats = plan(grid_network, 99999, 0)
        assert path == []
        assert cost == float("inf")
        assert stats["expansions"] == 0
