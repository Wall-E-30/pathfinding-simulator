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
