"""Tests for Task C: Streamlit deployment skeleton and configuration.

Ensures strict compliance with deployment requirements:
1. app/requirements.txt includes only the 5 required cloud dependencies and NO osmnx/geopandas.
2. app/app.py defines a valid Streamlit app with hard-coded lat/lon and Folium map.
3. README.md contains instructions to run `streamlit run app/app.py`.
"""

import ast
from pathlib import Path
import pytest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
ROOT_DIR = Path(__file__).resolve().parent.parent


class TestAppSkeleton:
    """Validate Streamlit deployment skeleton and cloud requirements."""

    def test_app_requirements_file_exists_and_strict_packages(self):
        """app/requirements.txt must contain only the 5 allowed packages, and strictly NO osmnx/geopandas."""
        req_file = APP_DIR / "requirements.txt"
        assert req_file.exists(), f"Missing {req_file}"

        content = req_file.read_text(encoding="utf-8")
        lines = [
            line.strip().lower()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        # Extract base package names ignoring version specifiers
        pkg_names = {
            line.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].strip()
            for line in lines
        }

        expected_pkgs = {"streamlit", "folium", "streamlit-folium", "networkx", "numpy"}
        assert pkg_names == expected_pkgs, f"Expected {expected_pkgs}, got {pkg_names}"

        # Explicitly ensure heavy C-extension GIS packages are omitted
        assert "osmnx" not in pkg_names
        assert "geopandas" not in pkg_names
        assert "gdal" not in pkg_names
        assert "pyproj" not in pkg_names

    def test_app_py_exists_and_syntax_valid(self):
        """app/app.py must exist and be valid Python code without syntax errors."""
        app_file = APP_DIR / "app.py"
        assert app_file.exists(), f"Missing {app_file}"

        code = app_file.read_text(encoding="utf-8")
        # ast.parse will raise SyntaxError if python code is invalid
        tree = ast.parse(code)
        assert tree is not None

    def test_app_py_creates_folium_map_at_hardcoded_coords(self):
        """app/app.py must define hard-coded center coordinates and build a Folium map."""
        app_file = APP_DIR / "app.py"
        assert app_file.exists(), f"Missing {app_file}"

        import app.app as app_module

        # Must have hardcoded lat/lon coords
        assert hasattr(app_module, "DEFAULT_CENTER")
        center = app_module.DEFAULT_CENTER
        assert isinstance(center, (list, tuple))
        assert len(center) == 2
        lat, lon = center
        assert 20.0 <= lat <= 30.0  # Around Ahmedabad [23.0225, 72.5714]
        assert 70.0 <= lon <= 80.0

        # Function to build map exists and returns a folium.Map
        assert hasattr(app_module, "create_map")
        m = app_module.create_map()
        import folium
        assert isinstance(m, folium.Map)

    def test_readme_contains_streamlit_run_instruction(self):
        """README.md must instruct users on how to run the app."""
        readme_file = ROOT_DIR / "README.md"
        assert readme_file.exists()
        content = readme_file.read_text(encoding="utf-8")
        assert "streamlit run app/app.py" in content



class TestUIHelpers:
    """Validate pure UI helper logic for coordinate extraction, maps, and tabular formatting."""

    @pytest.fixture
    def sample_graph(self):
        """Create a tiny DiGraph with coordinates for UI helper tests."""
        import networkx as nx
        g = nx.DiGraph()
        g.add_node(1, x=72.50, y=23.00)
        g.add_node(2, x=72.51, y=23.01)
        g.add_node(3, x=72.52, y=23.02)
        g.add_edge(1, 2, travel_time=60.0)
        g.add_edge(2, 3, travel_time=80.0)
        return g

    def test_node_path_to_coords(self, sample_graph):
        """node_path_to_coords returns [[lat, lon], ...] in traversal order."""
        from app.ui_helpers import node_path_to_coords

        coords = node_path_to_coords(sample_graph, [1, 2, 3])
        assert len(coords) == 3
        assert coords[0] == [23.00, 72.50]
        assert coords[1] == [23.01, 72.51]
        assert coords[2] == [23.02, 72.52]

        # Missing node is safely skipped
        partial_coords = node_path_to_coords(sample_graph, [1, 999, 2])
        assert len(partial_coords) == 2

    def test_event_to_coords_and_tooltip(self, sample_graph):
        """event_to_coords extracts endpoints and format_event_tooltip formats labels."""
        from src.events import Event
        from app.ui_helpers import event_to_coords, format_event_tooltip

        ev_closure = Event(step=2, u=1, v=2, new_cost=float("inf"), kind="closure")
        ev_coords = event_to_coords(sample_graph, ev_closure)
        assert ev_coords == [[23.00, 72.50], [23.01, 72.51]]

        tt_closure = format_event_tooltip(ev_closure, base_cost=60.0)
        assert "Closure" in tt_closure
        assert "Blocked (inf)" in tt_closure

        ev_cong = Event(step=3, u=2, v=3, new_cost=160.0, kind="congestion")
        tt_cong = format_event_tooltip(ev_cong, base_cost=80.0)
        assert "Congestion" in tt_cong
        assert "160.0s" in tt_cong
        assert "2.0x" in tt_cong

    def test_compute_route_bounds(self):
        """compute_route_bounds correctly identifies enclosing bounding box."""
        from app.ui_helpers import compute_route_bounds

        pts1 = [[23.00, 72.50], [23.05, 72.55]]
        pts2 = [[22.95, 72.60]]
        bounds = compute_route_bounds([pts1, pts2])
        assert bounds == [[22.95, 72.50], [23.05, 72.60]]

        assert compute_route_bounds([]) is None

    def test_build_comparison_rows(self):
        """build_comparison_rows builds structured records for baseline and D* Lite."""
        from app.ui_helpers import build_comparison_rows

        b_res = {
            "status": "reached",
            "steps": 25,
            "replans": 4,
            "replan_expansions": 500,
            "total_replan_time_ms": 40.0,
            "replan_latencies_ms": [10.0, 10.0, 10.0, 10.0],
            "travel_time": 350.5,
        }
        d_res = {
            "status": "reached",
            "steps": 25,
            "replans": 4,
            "replan_expansions": 125,
            "total_replan_time_ms": 20.0,
            "replan_latencies_ms": [5.0, 5.0, 5.0, 5.0],
            "travel_time": 350.5,
        }

        rows = build_comparison_rows(b_res, d_res)
        assert len(rows) == 2
        assert rows[0]["Algorithm"].startswith("Baseline")
        assert rows[0]["Replan Expansions"] == 500
        assert rows[0]["Mean Replan Latency (ms)"] == 10.0

        assert rows[1]["Algorithm"].startswith("D* Lite")
        assert rows[1]["Replan Expansions"] == 125
        assert rows[1]["Mean Replan Latency (ms)"] == 5.0

    def test_compute_summary_sentence_and_discrepancy(self):
        """Summary sentence correctly highlights expansion speedup and detects discrepancies."""
        from app.ui_helpers import check_travel_time_discrepancy, compute_summary_sentence

        b_res = {"replan_expansions": 400, "travel_time": 200.0}
        d_res = {"replan_expansions": 100, "travel_time": 200.0}

        sentence = compute_summary_sentence(b_res, d_res)
        assert "4.00x fewer expansions" in sentence

        has_diff, diff_val = check_travel_time_discrepancy(b_res, d_res)
        assert not has_diff
        assert diff_val == 0.0

        # Discrepancy case
        d_res_diff = {"replan_expansions": 100, "travel_time": 205.5}
        has_diff, diff_val = check_travel_time_discrepancy(b_res, d_res_diff)
        assert has_diff
        assert abs(diff_val - 5.5) < 1e-9

    def test_build_simulation_map(self, sample_graph):
        """build_simulation_map builds a folium.Map without errors."""
        import folium
        from src.events import Event
        from app.ui_helpers import build_simulation_map

        sim_data = {
            "start": 1,
            "goal": 3,
            "init_path": [1, 2, 3],
            "events": [Event(step=1, u=1, v=2, new_cost=float("inf"), kind="closure")],
            "baseline": {"path_taken": [1, 2, 3]},
            "dstar": {"path_taken": [1, 2, 3]},
        }

        m = build_simulation_map(sample_graph, sim_data)
        assert isinstance(m, folium.Map)
