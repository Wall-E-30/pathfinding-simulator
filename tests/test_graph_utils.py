"""Unit tests for graph utility functions:
- load_graph
- prepare_graph
- haversine_m
- compute_v_max
- make_heuristic
- snap_to_node
"""

from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import pytest

from src.graph_utils import (
    compute_v_max,
    haversine_m,
    load_graph,
    make_heuristic,
    prepare_graph,
    snap_to_node,
)


@pytest.fixture(scope="module")
def sample_multidigraph() -> nx.MultiDiGraph:
    """Fixture providing a deterministic MultiDiGraph with parallel edges and coordinates."""
    mg = nx.MultiDiGraph(crs="epsg:4326")
    # Triangle cycle (1 -> 2 -> 3 -> 1) ensuring strong connectivity
    # Distance between 1 and 2 is ~1023.5 metres
    mg.add_node(1, x=72.5000, y=23.0000)
    mg.add_node(2, x=72.5100, y=23.0000)
    mg.add_node(3, x=72.5050, y=23.0100)
    # Disconnected node to test SCC filtering
    mg.add_node(99, x=72.6000, y=23.1000)

    # Parallel edges from 1 -> 2 (road length >= straight line distance = 1500m):
    # Edge 0: length 1500m, speed 30 km/h (8.33 m/s) -> travel_time = 180s
    mg.add_edge(1, 2, key=0, length=1500.0, speed_kph=30.0, travel_time=180.0, highway="residential")
    # Edge 1: length 1500m, speed 60 km/h (16.67 m/s) -> travel_time = 90s (faster)
    mg.add_edge(1, 2, key=1, length=1500.0, speed_kph=60.0, travel_time=90.0, highway="primary")

    # Edge 2 -> 3
    mg.add_edge(2, 3, key=0, length=1500.0, speed_kph=40.0, travel_time=135.0, highway="secondary")
    # Edge 3 -> 1
    mg.add_edge(3, 1, key=0, length=1500.0, speed_kph=50.0, travel_time=108.0, highway="secondary")

    # One-way edge to disconnected node (not strongly connected)
    mg.add_edge(2, 99, key=0, length=5000.0, speed_kph=30.0, travel_time=600.0, highway="residential")

    return mg


class TestHaversineDistance:
    """Tests for haversine_m great-circle distance computation in metres."""

    def test_haversine_identical_coordinates(self):
        """Distance between identical coordinates must be 0.0 metres."""
        assert haversine_m(23.0225, 72.5714, 23.0225, 72.5714) == 0.0

    def test_haversine_known_benchmark_london_to_paris(self):
        """London (51.5074, -0.1278) to Paris (48.8566, 2.3522) benchmark ~343,556 metres."""
        london = (51.5074, -0.1278)
        paris = (48.8566, 2.3522)
        dist_m = haversine_m(london[0], london[1], paris[0], paris[1])
        # Validate against standard geodetic distance within 0.5% tolerance
        assert math.isclose(dist_m, 343_556, rel_tol=0.005)

    def test_haversine_symmetry(self):
        """haversine_m(A, B) must equal haversine_m(B, A)."""
        d1 = haversine_m(23.0605, 72.5802, 23.0063, 72.6026)
        d2 = haversine_m(23.0063, 72.6026, 23.0605, 72.5802)
        assert math.isclose(d1, d2, rel_tol=1e-9)


class TestSnapToNode:
    """Tests for snapping coordinates to the nearest network node."""

    def test_snap_three_coordinates_within_100m(self):
        """Snap 3 query coordinates placed close to known nodes and verify distance < 100m."""
        g = nx.DiGraph()
        # Define 3 reference nodes in Ahmedabad
        g.add_node(101, x=72.5714, y=23.0225)
        g.add_node(102, x=72.5802, y=23.0605)
        g.add_node(103, x=72.6026, y=23.0063)

        # 3 query coordinates offset by ~30-50 metres
        queries = [
            (23.0227, 72.5716, 101),
            (23.0607, 72.5804, 102),
            (23.0061, 72.6024, 103),
        ]

        for q_lat, q_lon, expected_node in queries:
            snapped_node = snap_to_node(g, q_lat, q_lon)
            assert snapped_node == expected_node
            node_x = g.nodes[snapped_node]["x"]
            node_y = g.nodes[snapped_node]["y"]
            dist_m = haversine_m(q_lat, q_lon, node_y, node_x)
            assert dist_m <= 100.0, f"Snapped node {snapped_node} is {dist_m:.2f}m away (> 100m)"


class TestPrepareGraph:
    """Tests for preparing road network DiGraphs from MultiDiGraphs."""

    def test_prepare_graph_returns_strongly_connected_digraph(self, sample_multidigraph):
        """prepare_graph must return a nx.DiGraph that is strongly connected and parallel-edge free."""
        prepared = prepare_graph(sample_multidigraph)

        # Must be nx.DiGraph and strictly NOT MultiDiGraph
        assert isinstance(prepared, nx.DiGraph)
        assert not isinstance(prepared, nx.MultiDiGraph)

        # Exactly one edge per (u, v) pair
        for u, v in prepared.edges():
            assert prepared.number_of_edges(u, v) == 1

        # Travel time must be positive on every edge
        for u, v, data in prepared.edges(data=True):
            assert "travel_time" in data
            assert data["travel_time"] > 0.0

        # Parallel edge collapsing: 1 -> 2 had 180s and 90s; minimum (90s) must be retained
        assert math.isclose(prepared[1][2]["travel_time"], 90.0, rel_tol=1e-3)

        # Node coordinates x and y must be preserved as floats
        for node in prepared.nodes():
            assert "x" in prepared.nodes[node] and isinstance(prepared.nodes[node]["x"], float)
            assert "y" in prepared.nodes[node] and isinstance(prepared.nodes[node]["y"], float)

        # Disconnected node 99 must be pruned; only nodes {1, 2, 3} form the largest SCC
        assert 99 not in prepared
        assert set(prepared.nodes()) == {1, 2, 3}
        assert nx.is_strongly_connected(prepared)


class TestLoadGraph:
    """Tests for offline loading of GraphML files with numeric attribute conversion."""

    def test_load_graph_converts_attributes(self, tmp_path):
        """GraphML attributes stored as strings must be converted to float/int."""
        # Create a small valid GraphML file with string-typed coordinates
        raw_graph = nx.MultiDiGraph(crs="epsg:4326")
        raw_graph.add_node(101, x="72.50", y="23.00")
        raw_graph.add_node(102, x="72.51", y="23.01")
        raw_graph.add_edge(101, 102, key=0, length="500.5", maxspeed="40")

        test_file = tmp_path / "test_city.graphml"
        nx.write_graphml(raw_graph, test_file)

        loaded = load_graph(test_file)
        assert isinstance(loaded, (nx.MultiDiGraph, nx.DiGraph))
        assert isinstance(loaded.nodes[101]["x"], float)
        assert isinstance(loaded.nodes[101]["y"], float)
        assert math.isclose(loaded.nodes[101]["x"], 72.50)
        assert math.isclose(loaded.nodes[101]["y"], 23.00)


class TestHeuristic:
    """Tests for compute_v_max and make_heuristic."""

    def test_compute_v_max_positive(self, sample_multidigraph):
        """v_max should match the fastest edge speed in metres per second."""
        prepared = prepare_graph(sample_multidigraph)
        v_max = compute_v_max(prepared)
        # In sample_multidigraph, max speed was 60 km/h = 16.666... m/s
        assert math.isclose(v_max, 60.0 / 3.6, rel_tol=1e-3)

    def test_make_heuristic_evaluates_correctly(self, sample_multidigraph):
        """make_heuristic returns h(n, goal) in seconds that is zero at goal and admissible."""
        prepared = prepare_graph(sample_multidigraph)
        h = make_heuristic(prepared)

        # Distance to self is 0.0s
        assert h(1, 1) == 0.0

        # Distance between distinct nodes is positive
        h_1_to_2 = h(1, 2)
        assert h_1_to_2 > 0.0

        # Admissibility check: h(u, v) <= actual edge travel time
        # Straight-line distance is ~1023.5m, v_max = 16.67 m/s -> h ~ 61.4s <= 90.0s
        actual_time = prepared[1][2]["travel_time"]
        assert h_1_to_2 <= actual_time + 1e-6



class TestLazyImports:
    """Test that osmnx and geopandas are not eagerly imported at module level."""

    def test_graph_utils_and_src_importable_without_osmnx(self, monkeypatch):
        """Importing src.graph_utils and src should not require osmnx or geopandas."""
        import importlib
        import sys

        # Block osmnx and geopandas from being imported
        monkeypatch.setitem(sys.modules, "osmnx", None)
        monkeypatch.setitem(sys.modules, "geopandas", None)

        # Reloading graph_utils must succeed without osmnx
        import src.graph_utils as gu
        importlib.reload(gu)

        # Verify make_heuristic, haversine_m, compute_v_max work on tiny hand-built DiGraph
        g = nx.DiGraph()
        g.add_node("A", x=72.0, y=23.0)
        g.add_node("B", x=72.01, y=23.0)
        g.add_edge("A", "B", length=1000.0, travel_time=60.0)

        h = gu.make_heuristic(g)
        assert h("A", "A") == 0.0
        assert h("A", "B") > 0.0
        assert gu.haversine_m(23.0, 72.0, 23.0, 72.01) > 0.0
        assert gu.compute_v_max(g) > 0.0

        # Reloading src top-level package must also succeed without osmnx/geopandas
        import src
        importlib.reload(src)
        assert hasattr(src, "make_heuristic")
        assert hasattr(src, "plan")
