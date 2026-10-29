"""Unit tests for dynamic events and graph mutation in src/events.py."""

from __future__ import annotations

import math
import networkx as nx
import pytest

from src.events import Event, apply_events


@pytest.fixture
def sample_graph() -> nx.DiGraph:
    """Fixture providing a simple 3-node graph with travel_time edge attributes."""
    g = nx.DiGraph()
    g.add_node("A", x=0.0, y=0.0)
    g.add_node("B", x=1.0, y=0.0)
    g.add_node("C", x=2.0, y=0.0)

    g.add_edge("A", "B", travel_time=10.0)
    g.add_edge("B", "C", travel_time=15.0)
    return g


class TestEvents:
    """Tests for Event namedtuple and apply_events mutation function."""

    def test_event_namedtuple_fields(self):
        """Event must store step, u, v, new_cost, and kind."""
        ev = Event(step=1, u="A", v="B", new_cost=25.0, kind="congestion")
        assert ev.step == 1
        assert ev.u == "A"
        assert ev.v == "B"
        assert ev.new_cost == 25.0
        assert ev.kind == "congestion"

    def test_apply_events_matching_step_mutates_graph(self, sample_graph):
        """apply_events mutates graph in place and returns list of (u, v, old_cost, new_cost)."""
        events = [
            Event(step=1, u="A", v="B", new_cost=20.0, kind="congestion"),
            Event(step=2, u="B", v="C", new_cost=30.0, kind="congestion"),
        ]

        changed = apply_events(sample_graph, events, step=1)

        # Step 1 event should have mutated edge ("A", "B")
        assert sample_graph["A"]["B"]["travel_time"] == 20.0
        # Edge ("B", "C") should remain untouched
        assert sample_graph["B"]["C"]["travel_time"] == 15.0

        # Return value must report the changed edge
        assert changed == [("A", "B", 10.0, 20.0)]

    def test_apply_events_returns_only_changed_edges(self, sample_graph):
        """Edges whose new_cost equals old_cost must NOT be included in the return list."""
        events = [
            Event(step=1, u="A", v="B", new_cost=10.0, kind="congestion"),  # Same cost!
            Event(step=1, u="B", v="C", new_cost=50.0, kind="congestion"),  # Changed
        ]

        changed = apply_events(sample_graph, events, step=1)

        assert changed == [("B", "C", 15.0, 50.0)]
        assert sample_graph["A"]["B"]["travel_time"] == 10.0
        assert sample_graph["B"]["C"]["travel_time"] == 50.0

    def test_apply_events_road_closure(self, sample_graph):
        """Closure sets travel_time = float('inf') and reports it."""
        events = [Event(step=1, u="A", v="B", new_cost=float("inf"), kind="closure")]

        changed = apply_events(sample_graph, events, step=1)

        assert math.isinf(sample_graph["A"]["B"]["travel_time"])
        assert len(changed) == 1
        u, v, old_c, new_c = changed[0]
        assert (u, v, old_c) == ("A", "B", 10.0)
        assert math.isinf(new_c)

    def test_apply_events_non_matching_step_leaves_graph_untouched(self, sample_graph):
        """When step does not match, return is empty and graph is untouched."""
        events = [Event(step=3, u="A", v="B", new_cost=float("inf"), kind="closure")]

        changed = apply_events(sample_graph, events, step=1)

        assert changed == []
        assert sample_graph["A"]["B"]["travel_time"] == 10.0
        assert sample_graph["B"]["C"]["travel_time"] == 15.0

