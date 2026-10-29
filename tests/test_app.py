"""Tests for Task C / Monday Task A: Streamlit deployment and UI helpers.

Ensures strict compliance with:
1. app/requirements.txt includes only the 5 required cloud dependencies and NO osmnx/geopandas/matplotlib.
2. app/ui_helpers.py pure functions for coordinates, event edges, comparison formatting,
   expansion sentences, and bounds.
"""

from pathlib import Path
import pytest
import networkx as nx

from src.events import Event

APP_DIR = Path(__file__).resolve().parent.parent / "app"
ROOT_DIR = Path(__file__).resolve().parent.parent


class TestAppRequirements:
    """Validate Streamlit cloud requirements."""

    def test_app_requirements_strict_packages(self):
        """app/requirements.txt must contain only the 5 allowed packages, and strictly NO osmnx/geopandas/matplotlib."""
        req_file = APP_DIR / "requirements.txt"
        assert req_file.exists(), f"Missing {req_file}"

        content = req_file.read_text(encoding="utf-8")
        lines = [
            line.strip().lower()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        pkg_names = {
            line.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].strip()
            for line in lines
        }

        expected_pkgs = {"streamlit", "folium", "streamlit-folium", "networkx", "numpy"}
        assert pkg_names == expected_pkgs, f"Expected {expected_pkgs}, got {pkg_names}"

        assert "osmnx" not in pkg_names
        assert "geopandas" not in pkg_names
        assert "matplotlib" not in pkg_names


class TestUIHelpers:
    """Validate pure helper functions for map rendering and metrics formatting."""

    @pytest.fixture
    def sample_graph(self):
        """Create a small directed graph with lat/lon node coordinates."""
        g = nx.DiGraph()
        g.add_node(1, y=23.02, x=72.57)
        g.add_node(2, y=23.03, x=72.58)
        g.add_node(3, y=23.04, x=72.59)
        g.add_edge(1, 2, travel_time=10.0)
        g.add_edge(2, 3, travel_time=15.0)
        return g

    def test_build_polyline_coords(self, sample_graph):
        from app.ui_helpers import build_polyline_coords

        coords = build_polyline_coords(sample_graph, [1, 2, 3])
        assert coords == [[23.02, 72.57], [23.03, 72.58], [23.04, 72.59]]

        empty_coords = build_polyline_coords(sample_graph, [])
        assert empty_coords == []

    def test_build_event_edges_data(self, sample_graph):
        from app.ui_helpers import build_event_edges_data

        events = [
            Event(step=1, u=1, v=2, new_cost=float("inf"), kind="closure"),
            Event(step=2, u=2, v=3, new_cost=45.0, kind="congestion"),
        ]
        edge_data = build_event_edges_data(sample_graph, events)
        assert len(edge_data) == 2

        # Closure edge
        assert edge_data[0]["color"] == "red"
        assert edge_data[0]["kind"] == "closure"
        assert "Closure" in edge_data[0]["tooltip"]
        assert edge_data[0]["coords"] == [[23.02, 72.57], [23.03, 72.58]]

        # Congestion edge
        assert edge_data[1]["color"] in ["orange", "#FF8C00"]
        assert edge_data[1]["kind"] == "congestion"
        assert "Congestion" in edge_data[1]["tooltip"]

    def test_format_comparison_table(self):
        from app.ui_helpers import format_comparison_table

        base_res = {
            "status": "reached",
            "steps": 12,
            "replans": 3,
            "replan_expansions": 450,
            "total_replan_time_ms": 15.2,
            "replan_latencies_ms": [5.0, 5.2, 5.0],
            "travel_time": 120.5,
        }
        dstar_res = {
            "status": "reached",
            "steps": 12,
            "replans": 3,
            "replan_expansions": 45,
            "total_replan_time_ms": 2.1,
            "replan_latencies_ms": [0.7, 0.7, 0.7],
            "travel_time": 120.5,
        }
        df_data = format_comparison_table(base_res, dstar_res)
        assert "Metric" in df_data
        assert "Baseline (A*)" in df_data
        assert "D* Lite" in df_data
        assert len(df_data["Metric"]) == 7

    def test_compute_expansion_sentence(self):
        from app.ui_helpers import compute_expansion_sentence

        # 10x fewer expansions
        msg = compute_expansion_sentence(baseline_exp=500, dstar_exp=50)
        assert "10.0x fewer expansions" in msg

        # Zero replans / expansions
        msg_zero = compute_expansion_sentence(baseline_exp=0, dstar_exp=0)
        assert "0 expansions" in msg_zero.lower() or "same" in msg_zero.lower()

    def test_compute_travel_time_warning(self):
        from app.ui_helpers import compute_travel_time_warning

        # Same travel time
        assert compute_travel_time_warning(120.0, 120.0) is None

        # Discrepancy > 1e-6
        warn = compute_travel_time_warning(120.0, 125.5)
        assert warn is not None
        assert "120.00" in warn and "125.50" in warn

    def test_build_map_bounds(self):
        from app.ui_helpers import build_map_bounds

        coords = [[23.01, 72.50], [23.10, 72.60]]
        bounds = build_map_bounds(coords)
        assert bounds == [[23.01, 72.50], [23.10, 72.60]]
        assert build_map_bounds([]) is None
