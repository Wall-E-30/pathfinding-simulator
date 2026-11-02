"""Streamlit Web Application: Dynamic Emergency Vehicle Rerouting on Live Street Networks.

Visualizes dynamic replanning (Baseline A* vs. D* Lite) under real-time
road disruptions (closures and congestion) on an interactive Leaflet/Folium map.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Sequence

# Ensure repository root is on sys.path for Streamlit Cloud deployment
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import folium
import streamlit as st
from streamlit_folium import st_folium

from app.ui_helpers import (
    build_comparison_rows,
    build_simulation_map,
    check_travel_time_discrepancy,
    compute_summary_sentence,
)
from src.agent_baseline import run_baseline_agent
from src.agent_dstar import run_dstar_agent
from src.astar import plan
from src.graph_utils import make_heuristic
from src.simulator import make_scenario

DEFAULT_CENTER: list[float] = [23.0225, 72.5714]
DEFAULT_ZOOM: int = 12


def create_map(
    center: Sequence[float] = DEFAULT_CENTER,
    zoom: int = DEFAULT_ZOOM,
) -> folium.Map:
    """Create default base Folium map centered on the network."""
    m = folium.Map(location=[center[0], center[1]], zoom_start=zoom, tiles="OpenStreetMap")
    folium.Marker(
        location=[center[0], center[1]],
        popup="Emergency Dispatch HQ (Ahmedabad)",
        tooltip="Dispatch Center",
        icon=folium.Icon(color="red", icon="plus", prefix="fa"),
    ).add_to(m)
    return m


@st.cache_resource
def load_app_graph() -> tuple[Any, Any]:
    """Load cropped road network graph and build admissible travel-time heuristic."""
    pkl_path = REPO_ROOT / "data" / "app_graph.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing {pkl_path}. Run scripts/export_app_graph.py first.")
    with open(pkl_path, "rb") as f:
        graph = pickle.load(f)
    heuristic = make_heuristic(graph)
    return graph, heuristic


def main() -> None:
    st.set_page_config(page_title="Emergency Vehicle Rerouting", page_icon="🚑", layout="wide")
    st.title("🚑 Dynamic Emergency Vehicle Rerouting on Live Street Networks")

    try:
        graph, heuristic = load_app_graph()
    except Exception as e:
        st.error(f"Failed to load road network: {e}")
        return

    # Sidebar controls
    with st.sidebar:
        st.header("Simulation Settings")
        seed = int(st.number_input("Scenario Seed", min_value=0, value=0, step=1))
        n_events = int(st.slider("Dynamic Traffic Events", min_value=3, max_value=10, value=6, step=1))
        run_btn = st.button("Run simulation", type="primary", use_container_width=True)

    if "sim_data" not in st.session_state:
        st.session_state["sim_data"] = None

    if run_btn:
        try:
            with st.spinner("Generating dynamic scenario and executing agents..."):
                start, goal, events = make_scenario(graph, heuristic=heuristic, seed=seed, n_events=n_events)
                init_path, _, _ = plan(graph, start, goal, heuristic=heuristic)
                b_res = run_baseline_agent(graph, start, goal, events, heuristic=heuristic, verify=False)
                d_res = run_dstar_agent(graph, start, goal, events, heuristic=heuristic, verify=False)
                st.session_state["sim_data"] = {
                    "start": start,
                    "goal": goal,
                    "events": events,
                    "init_path": init_path,
                    "baseline": b_res,
                    "dstar": d_res,
                }
        except Exception as exc:
            st.error(f"Simulation Error: {exc}")

    # Map visualization
    st.subheader("Route & Dynamic Disruption Map")
    if st.session_state["sim_data"] is not None:
        folium_map = build_simulation_map(graph, st.session_state["sim_data"], default_zoom=DEFAULT_ZOOM)
    else:
        folium_map = create_map()

    st_folium(folium_map, returned_objects=[], width=None, height=520, use_container_width=True)

    # Comparative evaluation table and metrics
    if st.session_state["sim_data"] is not None:
        sim = st.session_state["sim_data"]
        st.subheader("Agent Performance Evaluation")
        table_rows = build_comparison_rows(sim["baseline"], sim["dstar"])
        st.dataframe(table_rows, use_container_width=True)

        st.info(f"💡 {compute_summary_sentence(sim['baseline'], sim['dstar'])}")

        has_diff, diff_val = check_travel_time_discrepancy(sim["baseline"], sim["dstar"])
        if has_diff:
            st.warning(f"⚠️ Warning: Travel times differ by {diff_val:.6f} seconds (> 1e-6)!")


if __name__ == "__main__":
    main()
