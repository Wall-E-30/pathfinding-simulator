"""Unit tests for the directed graph core representation and OSMnx import."""

import math
import pytest
import networkx as nx

from src.graph import Graph


class TestGraphRepresentation:
    """Tests for core directed graph storage and properties."""

    def test_empty_graph(self):
        graph = Graph()
        assert graph.node_count == 0
        assert graph.edge_count == 0
        assert list(graph.nodes) == []

    def test_add_node(self):
        graph = Graph()
        graph.add_node("A", x=10.0, y=20.0)
        assert graph.has_node("A")
        assert not graph.has_node("B")
        assert graph.get_node_data("A") == {"x": 10.0, "y": 20.0}

    def test_add_directed_edge_stores_travel_time_in_seconds(self):
        graph = Graph()
        graph.add_edge("A", "B", cost=15.5)
        assert graph.has_edge("A", "B")
        assert not graph.has_edge("B", "A")  # Directed: reverse does not exist
        assert graph.get_edge_cost("A", "B") == 15.5
        assert graph.edge_count == 1

    def test_exactly_one_cost_per_directed_pair(self):
        graph = Graph()
        graph.add_edge("A", "B", cost=10.0)
        graph.add_edge("A", "B", cost=20.0)
        assert graph.edge_count == 1
        assert graph.get_edge_cost("A", "B") == 20.0

    def test_get_neighbors(self):
        graph = Graph()
        graph.add_edge("A", "B", cost=5.0)
        graph.add_edge("A", "C", cost=8.0)
        neighbors = graph.get_neighbors("A")
        assert neighbors == {"B": 5.0, "C": 8.0}
        assert graph.get_neighbors("B") == {}


class TestCostValidation:
    """Tests for rejecting invalid or negative travel costs."""

    @pytest.mark.parametrize("invalid_cost", [-1.0, -0.001, -100])
    def test_reject_negative_costs_on_add(self, invalid_cost):
        graph = Graph()
        with pytest.raises(ValueError, match="Travel cost must be a non-negative number"):
            graph.add_edge("A", "B", cost=invalid_cost)

    @pytest.mark.parametrize("invalid_cost", ["fast", None, [10], {"cost": 5}])
    def test_reject_non_numeric_costs_on_add(self, invalid_cost):
        graph = Graph()
        with pytest.raises((TypeError, ValueError)):
            graph.add_edge("A", "B", cost=invalid_cost)

    @pytest.mark.parametrize("invalid_cost", [float("nan"), float("-inf"), float("inf")])
    def test_reject_nan_or_inf_costs_on_add(self, invalid_cost):
        graph = Graph()
        with pytest.raises(ValueError, match="Travel cost must be a finite, non-negative number"):
            graph.add_edge("A", "B", cost=invalid_cost)

    def test_allow_zero_cost(self):
        graph = Graph()
        graph.add_edge("A", "B", cost=0.0)
        assert graph.get_edge_cost("A", "B") == 0.0


class TestEdgeUpdate:
    """Tests for updating edge travel costs."""

    def test_update_existing_edge(self):
        graph = Graph()
        graph.add_edge("A", "B", cost=10.0)
        graph.update_edge("A", "B", 25.5)
        assert graph.get_edge_cost("A", "B") == 25.5
        assert graph.get_neighbors("A")["B"] == 25.5

    def test_update_edge_with_zero(self):
        graph = Graph()
        graph.add_edge("A", "B", cost=10.0)
        graph.update_edge("A", "B", 0.0)
        assert graph.get_edge_cost("A", "B") == 0.0

    @pytest.mark.parametrize("invalid_cost", [-5.0, -0.01, float("nan"), float("inf"), "invalid"])
    def test_reject_invalid_cost_on_update(self, invalid_cost):
        graph = Graph()
        graph.add_edge("A", "B", cost=10.0)
        with pytest.raises((ValueError, TypeError)):
            graph.update_edge("A", "B", invalid_cost)
        # Verify cost was not modified
        assert graph.get_edge_cost("A", "B") == 10.0

    def test_update_nonexistent_edge_raises_key_error(self):
        """Updating a nonexistent edge must raise KeyError to prevent unintended edge creation."""
        graph = Graph()
        graph.add_node("A")
        graph.add_node("B")
        with pytest.raises(KeyError, match=r"Edge \('A', 'B'\) does not exist"):
            graph.update_edge("A", "B", 10.0)
        assert not graph.has_edge("A", "B")

    def test_update_nonexistent_node_raises_key_error(self):
        graph = Graph()
        with pytest.raises(KeyError):
            graph.update_edge("X", "Y", 10.0)


class TestOSMnxMultiDiGraphImport:
    """Tests for converting MultiDiGraph and collapsing parallel edges."""

    def test_collapse_parallel_edges_retains_minimum_travel_time(self):
        # Create a NetworkX MultiDiGraph simulating an OSMnx road network
        multi_graph = nx.MultiDiGraph()
        multi_graph.add_node(1, x=-122.4, y=37.7)
        multi_graph.add_node(2, x=-122.3, y=37.8)

        # Three parallel edges between 1 and 2 with different travel times
        multi_graph.add_edge(1, 2, key=0, travel_time=45.0, name="Main St")
        multi_graph.add_edge(1, 2, key=1, travel_time=30.0, name="Expressway")
        multi_graph.add_edge(1, 2, key=2, travel_time=50.0, name="Side Alley")

        graph = Graph.from_osmnx(multi_graph)

        assert graph.node_count == 2
        assert graph.edge_count == 1
        assert graph.has_edge(1, 2)
        assert not graph.has_edge(2, 1)
        # Should keep the minimum: 30.0 seconds
        assert graph.get_edge_cost(1, 2) == 30.0

    def test_parallel_edges_in_opposite_directions_kept_separate(self):
        multi_graph = nx.MultiDiGraph()
        # 1 -> 2 parallel edges
        multi_graph.add_edge(1, 2, key=0, travel_time=12.0)
        multi_graph.add_edge(1, 2, key=1, travel_time=15.0)
        # 2 -> 1 edges
        multi_graph.add_edge(2, 1, key=0, travel_time=20.0)
        multi_graph.add_edge(2, 1, key=1, travel_time=18.0)

        graph = Graph.from_osmnx(multi_graph)

        assert graph.edge_count == 2
        assert graph.get_edge_cost(1, 2) == 12.0
        assert graph.get_edge_cost(2, 1) == 18.0

    def test_preserves_node_attributes(self):
        multi_graph = nx.MultiDiGraph()
        multi_graph.add_node("X", x=12.34, y=56.78, elevation=150)
        multi_graph.add_node("Y", x=12.35, y=56.79, elevation=152)
        multi_graph.add_edge("X", "Y", key=0, travel_time=10.0)

        graph = Graph.from_osmnx(multi_graph)
        assert graph.get_node_data("X") == {"x": 12.34, "y": 56.78, "elevation": 150}
        assert graph.get_node_data("Y") == {"x": 12.35, "y": 56.79, "elevation": 152}

    def test_custom_weight_attribute(self):
        multi_graph = nx.MultiDiGraph()
        multi_graph.add_edge("A", "B", key=0, cost_seconds=40.0)
        multi_graph.add_edge("A", "B", key=1, cost_seconds=25.0)

        graph = Graph.from_osmnx(multi_graph, weight_attr="cost_seconds")
        assert graph.get_edge_cost("A", "B") == 25.0

    def test_raises_error_if_weight_missing_on_all_edges(self):
        multi_graph = nx.MultiDiGraph()
        multi_graph.add_edge("A", "B", key=0, length=100.0)

        with pytest.raises(KeyError, match="No valid travel_time found for edge"):
            Graph.from_osmnx(multi_graph, weight_attr="travel_time")
