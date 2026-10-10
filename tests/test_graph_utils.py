"""Unit tests for graph utility functions: Haversine distance, node snapping,
speeds/travel times, edge collapsing, and v_max calculation.
"""

from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import osmnx as ox
import pytest

from src.graph import Graph
from src.graph_utils import (
    EARTH_RADIUS_M,
    add_edge_speeds,
    add_edge_speeds_and_travel_times,
    add_edge_travel_times,
    collapse_parallel_edges,
    compute_v_max,
    haversine,
    snap_to_node,
)


@pytest.fixture(scope="module")
def ahmedabad_graph() -> nx.MultiDiGraph:
    """Load the Ahmedabad city graph once for module tests if available."""
    graph_path = Path(__file__).resolve().parent.parent / "data" / "city.graphml"
    if not graph_path.exists() or graph_path.stat().st_size == 0:
        pytest.skip("data/city.graphml does not exist or is empty")
    return ox.load_graphml(graph_path)


class TestHaversineDistance:
    """Tests for great-circle distance calculation via the Haversine formula."""

    def test_identical_coordinates_return_zero_distance(self):
        """Distance between identical coordinates must be exactly 0.0 metres."""
        dist = haversine(23.0225, 72.5714, 23.0225, 72.5714)
        assert dist == 0.0

    def test_haversine_symmetry(self):
        """haversine(A, B) must equal haversine(B, A)."""
        lat1, lon1 = 23.0605, 72.5802
        lat2, lon2 = 23.0063, 72.6026
        d1 = haversine(lat1, lon1, lat2, lon2)
        d2 = haversine(lat2, lon2, lat1, lon1)
        assert math.isclose(d1, d2, rel_tol=1e-9)

    def test_known_equatorial_distance(self):
        """Test 1 degree of longitude along the equator against known arc length.

        Arc length for 1 deg at radius R = R * (pi / 180) ≈ 111,194.9266 metres.
        """
        expected_distance = EARTH_RADIUS_M * (math.pi / 180.0)
        computed_distance = haversine(0.0, 0.0, 0.0, 1.0)
        assert math.isclose(computed_distance, expected_distance, rel_tol=1e-5)
        # Approximately 111.19 km
        assert 111_100 < computed_distance < 111_300

    def test_known_intercity_distance_london_to_paris(self):
        """Test London to Paris great-circle distance against standard geodetic benchmark (~343.5 km)."""
        london = (51.5074, -0.1278)
        paris = (48.8566, 2.3522)
        dist_m = haversine(london[0], london[1], paris[0], paris[1])
        # Standard benchmark distance is ~343,556 metres; verify within 0.5%
        assert math.isclose(dist_m, 343_556, rel_tol=0.005)

    @pytest.mark.parametrize(
        "invalid_coords",
        [
            (95.0, 0.0, 0.0, 0.0),    # lat1 > 90
            (-95.0, 0.0, 0.0, 0.0),   # lat1 < -90
            (0.0, 185.0, 0.0, 0.0),   # lon1 > 180
            (0.0, -185.0, 0.0, 0.0),  # lon1 < -180
            (0.0, 0.0, 91.0, 0.0),    # lat2 > 90
            (0.0, 0.0, 0.0, 181.0),   # lon2 > 180
            (float("nan"), 0.0, 0.0, 0.0),
            (0.0, float("inf"), 0.0, 0.0),
        ],
    )
    def test_reject_out_of_bounds_or_non_finite_coordinates(self, invalid_coords):
        with pytest.raises(ValueError):
            haversine(*invalid_coords)

    @pytest.mark.parametrize(
        "invalid_types",
        [
            ("23.0", 72.0, 23.5, 72.5),
            (23.0, None, 23.5, 72.5),
            (True, 72.0, 23.5, 72.5),
        ],
    )
    def test_reject_non_numeric_coordinates(self, invalid_types):
        with pytest.raises(TypeError):
            haversine(*invalid_types)


class TestSnapToNode:
    """Tests for snapping geographic coordinates to nearest network nodes."""

    # Three well-known landmarks in Ahmedabad with coordinates
    AHMEDABAD_LANDMARKS = [
        ("Sabarmati Ashram", 23.0605, 72.5802, 300.0),
        ("Atal Pedestrian Bridge", 23.0232, 72.5714, 200.0),
        ("Kankaria Lake", 23.0063, 72.6026, 500.0),
    ]

    @pytest.mark.parametrize("name, lat, lon, max_distance_m", AHMEDABAD_LANDMARKS)
    def test_snap_ahmedabad_coordinates_to_sensible_nodes(
        self, ahmedabad_graph, name, lat, lon, max_distance_m
    ):
        """Verify three known Ahmedabad locations snap to valid, nearby nodes in the road network."""
        node_id = snap_to_node(ahmedabad_graph, lat, lon)

        # Node must exist in the graph
        assert node_id in ahmedabad_graph

        # Node must have valid geographic coordinates
        node_data = ahmedabad_graph.nodes[node_id]
        assert "x" in node_data and "y" in node_data
        node_lat = float(node_data["y"])
        node_lon = float(node_data["x"])

        # Distance from query point to the snapped node should be sensible (< max_distance_m)
        dist_m = haversine(lat, lon, node_lat, node_lon)
        assert dist_m < max_distance_m, (
            f"{name} ({lat}, {lon}) snapped to node {node_id} ({node_lat}, {node_lon}) "
            f"which is {dist_m:.1f}m away (expected < {max_distance_m}m)"
        )

    def test_snap_to_node_with_return_dist(self, ahmedabad_graph):
        """Verify snap_to_node returns (node, distance) when return_dist=True."""
        lat, lon = 23.0605, 72.5802
        node_id, dist = snap_to_node(ahmedabad_graph, lat, lon, return_dist=True)

        assert node_id in ahmedabad_graph
        assert isinstance(dist, (int, float))
        assert dist > 0.0

        # Verify returned distance matches calculated Haversine distance
        node_data = ahmedabad_graph.nodes[node_id]
        expected_dist = haversine(lat, lon, float(node_data["y"]), float(node_data["x"]))
        assert math.isclose(dist, expected_dist, rel_tol=0.01)

    def test_snap_to_node_on_synthetic_graph(self):
        """Test snapping on a small synthetic NetworkX graph."""
        g = nx.MultiDiGraph(crs="epsg:4326")
        g.add_node(101, x=72.50, y=23.00)
        g.add_node(102, x=72.52, y=23.02)
        g.add_edge(101, 102, key=0, length=100.0)

        # Point very close to node 101
        snapped = snap_to_node(g, lat=23.0001, lon=72.5001)
        assert snapped == 101

    def test_snap_to_node_on_custom_graph(self):
        """Test snapping on custom src.graph.Graph instance."""
        graph = Graph()
        graph.add_node("N1", x=72.50, y=23.00)
        graph.add_node("N2", x=72.55, y=23.05)
        graph.add_edge("N1", "N2", cost=60.0)

        # Query close to N2
        snapped = snap_to_node(graph, lat=23.049, lon=72.551)
        assert snapped == "N2"

    @pytest.mark.parametrize("invalid_lat, invalid_lon", [(100.0, 72.0), (-91.0, 72.0), (23.0, 200.0)])
    def test_snap_rejects_out_of_range_coordinates(self, ahmedabad_graph, invalid_lat, invalid_lon):
        with pytest.raises(ValueError):
            snap_to_node(ahmedabad_graph, invalid_lat, invalid_lon)


class TestSpeedsAndTravelTimes:
    """Tests for adding edge speeds and calculating travel times."""

    @pytest.fixture
    def sample_network(self) -> nx.MultiDiGraph:
        """Create a small MultiDiGraph with coordinates and length."""
        g = nx.MultiDiGraph(crs="epsg:4326")
        g.add_node(1, x=72.50, y=23.00)
        g.add_node(2, x=72.51, y=23.01)
        g.add_node(3, x=72.52, y=23.02)

        # Edge with explicit maxspeed
        g.add_edge(1, 2, key=0, length=1000.0, maxspeed="50", highway="primary")
        # Edge without explicit maxspeed
        g.add_edge(2, 3, key=0, length=500.0, highway="residential")
        return g

    def test_add_edge_speeds(self, sample_network):
        g = add_edge_speeds(sample_network, fallback=30.0)
        # All edges must have speed_kph
        for _, _, data in g.edges(data=True):
            assert "speed_kph" in data
            assert data["speed_kph"] > 0

    def test_add_edge_travel_times(self, sample_network):
        g = add_edge_speeds(sample_network, fallback=30.0)
        g = add_edge_travel_times(g)

        for _, _, data in g.edges(data=True):
            assert "travel_time" in data
            # travel_time (s) = length (m) / (speed_kph * 1000 / 3600)
            expected_time = data["length"] / (data["speed_kph"] * 1000.0 / 3600.0)
            assert math.isclose(data["travel_time"], expected_time, rel_tol=1e-5)

    def test_add_edge_speeds_and_travel_times_combined(self, sample_network):
        g = add_edge_speeds_and_travel_times(sample_network, fallback=40.0)
        for _, _, data in g.edges(data=True):
            assert "speed_kph" in data
            assert "travel_time" in data
            assert data["travel_time"] > 0


class TestCollapseParallelEdges:
    """Tests for collapsing parallel edges by retaining minimum travel time."""

    def test_collapse_parallel_edges_into_digraph(self):
        mg = nx.MultiDiGraph(crs="epsg:4326")
        mg.add_node(1, x=72.5, y=23.0)
        mg.add_node(2, x=72.6, y=23.1)

        # Three parallel edges between 1 and 2
        mg.add_edge(1, 2, key=0, travel_time=60.0, name="Slow Road")
        mg.add_edge(1, 2, key=1, travel_time=25.0, name="Expressway")
        mg.add_edge(1, 2, key=2, travel_time=45.0, name="Avenue")

        di_graph = collapse_parallel_edges(mg)
        assert isinstance(di_graph, nx.DiGraph)
        assert di_graph.number_of_nodes() == 2
        assert di_graph.number_of_edges() == 1
        assert di_graph.has_edge(1, 2)
        assert di_graph[1][2]["travel_time"] == 25.0
        assert di_graph[1][2]["name"] == "Expressway"

    def test_collapse_parallel_edges_into_custom_graph(self):
        mg = nx.MultiDiGraph(crs="epsg:4326")
        mg.add_node("A", x=72.5, y=23.0)
        mg.add_node("B", x=72.6, y=23.1)

        mg.add_edge("A", "B", key=0, travel_time=40.0)
        mg.add_edge("A", "B", key=1, travel_time=15.0)

        graph = collapse_parallel_edges(mg, as_custom_graph=True)
        assert isinstance(graph, Graph)
        assert graph.node_count == 2
        assert graph.edge_count == 1
        assert graph.get_edge_cost("A", "B") == 15.0


class TestComputeVMax:
    """Tests for calculating maximum traversal speed (v_max) in metres/second."""

    def test_compute_v_max_from_speed_kph(self):
        g = nx.DiGraph()
        g.add_edge(1, 2, speed_kph=50.0, length=500.0, travel_time=36.0)
        g.add_edge(2, 3, speed_kph=108.0, length=1000.0, travel_time=33.33)
        g.add_edge(3, 4, speed_kph=36.0, length=200.0, travel_time=20.0)

        # Max speed is 108.0 km/h = 108.0 / 3.6 = 30.0 m/s
        v_max = compute_v_max(g)
        assert math.isclose(v_max, 30.0, rel_tol=1e-5)

    def test_compute_v_max_from_length_and_travel_time(self):
        g = nx.DiGraph()
        # 100 metres in 5 seconds = 20 m/s
        g.add_edge(1, 2, length=100.0, travel_time=5.0)
        # 250 metres in 10 seconds = 25 m/s
        g.add_edge(2, 3, length=250.0, travel_time=10.0)

        v_max = compute_v_max(g)
        assert math.isclose(v_max, 25.0, rel_tol=1e-5)

    def test_compute_v_max_rejects_empty_or_speedless_graph(self):
        g = nx.DiGraph()
        with pytest.raises(ValueError, match="Could not determine valid positive edge speeds"):
            compute_v_max(g)

    def test_compute_v_max_on_ahmedabad_graph(self, ahmedabad_graph):
        g = add_edge_speeds_and_travel_times(ahmedabad_graph, fallback=30.0)
        v_max = compute_v_max(g)
        # In an urban road network, v_max is typically between 20 m/s (~72 km/h) and 35 m/s (~126 km/h)
        assert 15.0 <= v_max <= 40.0
