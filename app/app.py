"""Streamlit web application for Dynamic Emergency Vehicle Rerouting.

Visualizes live road networks, dynamic traffic incidents (closures and congestion),
and compares baseline repeated A* vs incremental D* Lite.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Callable

import folium
import networkx as nx
import streamlit as st
from streamlit_folium import st_folium

# Add repository root to sys.path so 'src' and 'app' packages resolve on Streamlit Cloud
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ui_helpers import (
    build_event_edges_data,
    build_map_bounds,
    build_polyline_coords,
    compute_expansion_sentence,
    compute_travel_time_warning,
    format_comparison_table,
    get_legend_html,
)
from src.agent_baseline import run_baseline_agent
from src.agent_dstar import run_dstar_agent
from src.astar import plan
from src.graph_utils import make_heuristic
from src.simulator import make_scenario


@st.cache_resource
def load_app_graph() -> tuple[nx.DiGraph, Callable[[Any, Any], float]]:
    """Unpickle the pre-processed road network and build the admissible heuristic."""
    pkl_path = REPO_ROOT / "data" / "app_graph.pkl"
    with open(pkl_path, "rb") as f:
        graph: nx.DiGraph = pickle.load(f)
    heuristic = make_heuristic(graph)
    return graph, heuristic


def render_folium_map(
    graph: nx.DiGraph,
    start: Any,
    goal: Any,
    initial_path: list[Any],
    dstar_path: list[Any],
    baseline_path: list[Any],
    events: list[Any],
) -> folium.Map:
    """Build an interactive Folium map illustrating routes, events, and incident markers."""
    start_lat, start_lon = float(graph.nodes[start]["y"]), float(graph.nodes[start]["x"])
    goal_lat, goal_lon = float(graph.nodes[goal]["y"]), float(graph.nodes[goal]["x"])

    m = folium.Map(location=[start_lat, start_lon], zoom_start=13, tiles="OpenStreetMap")

    # Start and Goal incident markers
    folium.Marker(
        [start_lat, start_lon],
        popup=f"Dispatch HQ ({start})",
        tooltip="Dispatch Start",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)

    folium.Marker(
        [goal_lat, goal_lon],
        popup=f"Emergency Incident ({goal})",
        tooltip="Incident Goal",
        icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa"),
    ).add_to(m)

    # Initial planned path (blue dashed polyline)
    init_coords = build_polyline_coords(graph, initial_path)
    if init_coords:
        folium.PolyLine(
            init_coords, color="blue", weight=3, dash_array="6, 8",
            tooltip="Initial Planned Route (Static A*)"
        ).add_to(m)

    # Baseline path (orange) rendered only if different from D* Lite
    dstar_coords = build_polyline_coords(graph, dstar_path)
    base_coords = build_polyline_coords(graph, baseline_path)
    if dstar_path != baseline_path and base_coords:
        folium.PolyLine(
            base_coords, color="#FF8C00", weight=4, opacity=0.8,
            dash_array="4, 6", tooltip="Baseline Actual Path Taken"
        ).add_to(m)

    # D* Lite path (solid green polyline)
    if dstar_coords:
        folium.PolyLine(
            dstar_coords, color="#28a745", weight=5, opacity=0.9,
            tooltip="D* Lite Actual Path Taken"
        ).add_to(m)

    # Dynamic event edges (closures in red, congestion in orange)
    event_edges = build_event_edges_data(graph, events)
    for edge in event_edges:
        folium.PolyLine(
            edge["coords"], color=edge["color"], weight=5, opacity=0.9,
            tooltip=edge["tooltip"]
        ).add_to(m)

    # Fit camera bounds to encompass the complete route and event area
    all_coords = init_coords + dstar_coords + base_coords + [c for e in event_edges for c in e["coords"]]
    bounds = build_map_bounds(all_coords)
    if bounds:
        m.fit_bounds(bounds, padding=[20, 20])

    m.get_root().html.add_child(folium.Element(get_legend_html()))
    return m


def main() -> None:
    st.set_page_config(page_title="Emergency Vehicle Rerouting", page_icon="🚑", layout="wide")
    st.title("🚑 Dynamic Emergency Vehicle Rerouting on Live Street Networks")
    st.caption("Real-time comparative evaluation of D* Lite vs Repeated A* under dynamic traffic events.")

    try:
        graph, heuristic = load_app_graph()
    except Exception as err:
        st.error(f"Failed to load road network: {err}")
        return

    # Sidebar simulation parameters
    st.sidebar.header("Simulation Settings")
    seed = st.sidebar.number_input("Random Seed", min_value=0, value=0, step=1)
    n_events = st.sidebar.slider("Number of Events", min_value=3, max_value=10, value=6)
    run_btn = st.sidebar.button("Run simulation", type="primary")

    if run_btn:
        with st.spinner("Generating scenario and simulating emergency routing..."):
            try:
                start, goal, events = make_scenario(graph, heuristic, seed=int(seed), n_events=int(n_events))
                init_path, init_cost, _ = plan(graph, start, goal, heuristic)
                res_baseline = run_baseline_agent(graph, start, goal, events, heuristic, verify=False)
                res_dstar = run_dstar_agent(graph, start, goal, events, heuristic, verify=False)

                st.session_state["sim_data"] = {
                    "start": start, "goal": goal, "events": events, "init_path": init_path,
                    "baseline": res_baseline, "dstar": res_dstar,
                }
            except Exception as e:
                st.error(f"Simulation failed: {e}")

    # Render results from session state if present
    sim_data = st.session_state.get("sim_data")
    if sim_data:
        folium_map = render_folium_map(
            graph, sim_data["start"], sim_data["goal"], sim_data["init_path"],
            sim_data["dstar"]["path_taken"], sim_data["baseline"]["path_taken"], sim_data["events"]
        )
        st_folium(folium_map, returned_objects=[], use_container_width=True, height=520)

        # Metrics comparison table
        st.subheader("Performance Comparison")
        table_dict = format_comparison_table(sim_data["baseline"], sim_data["dstar"])
        st.dataframe(table_dict, hide_index=True, use_container_width=True)

        # Summary sentence & travel time discrepancy check
        summary_sentence = compute_expansion_sentence(
            sim_data["baseline"]["replan_expansions"], sim_data["dstar"]["replan_expansions"]
        )
        st.info(summary_sentence)

        warning_msg = compute_travel_time_warning(
            sim_data["baseline"]["travel_time"], sim_data["dstar"]["travel_time"]
        )
        if warning_msg:
            st.warning(warning_msg)
    else:
        st.info("👈 Set the simulation parameters in the sidebar and click **Run simulation**.")


if __name__ == "__main__":
    main()
