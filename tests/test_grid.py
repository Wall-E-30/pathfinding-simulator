"""Unit tests for the toy grid graph builder used for pathfinding tests."""

import pytest
from src.grid import create_grid_graph


class TestGridGraphBuilder:
    """Tests for creating toy grid graphs with walls."""

    def test_create_default_10x10_grid(self):
        graph = create_grid_graph(width=10, height=10)
        assert graph.node_count == 100
        # Check coordinates of origin and top-right corner
        assert graph.get_node_data((0, 0)) == {"x": 0.0, "y": 0.0}
        assert graph.get_node_data((9, 9)) == {"x": 9.0, "y": 9.0}

    def test_grid_4_connected_neighbors(self):
        graph = create_grid_graph(width=10, height=10, default_cost=1.0)
        # Corner node has 2 neighbors
        corner_neighbors = graph.get_neighbors((0, 0))
        assert set(corner_neighbors.keys()) == {(1, 0), (0, 1)}
        assert all(cost == 1.0 for cost in corner_neighbors.values())

        # Interior node has 4 neighbors
        center_neighbors = graph.get_neighbors((5, 5))
        assert set(center_neighbors.keys()) == {(4, 5), (6, 5), (5, 4), (5, 6)}
        assert all(cost == 1.0 for cost in center_neighbors.values())

    def test_grid_with_walls(self):
        walls = {(1, 0), (1, 1), (1, 2)}
        graph = create_grid_graph(width=10, height=10, walls=walls)

        # Wall nodes must not have any outgoing edges
        for wall in walls:
            assert graph.get_neighbors(wall) == {}

        # Non-wall neighbors should not have edges pointing to walls
        neighbors_of_0_0 = graph.get_neighbors((0, 0))
        assert (1, 0) not in neighbors_of_0_0
        assert (0, 1) in neighbors_of_0_0

    def test_edge_cost_in_seconds(self):
        graph = create_grid_graph(width=5, height=5, default_cost=2.5)
        assert graph.get_edge_cost((0, 0), (1, 0)) == 2.5

    def test_supports_dynamic_edge_update(self):
        graph = create_grid_graph(width=10, height=10, default_cost=1.0)
        assert graph.get_edge_cost((0, 0), (1, 0)) == 1.0
        graph.update_edge((0, 0), (1, 0), 10.0)
        assert graph.get_edge_cost((0, 0), (1, 0)) == 10.0
