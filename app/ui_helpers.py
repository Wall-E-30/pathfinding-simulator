"""Pure UI helper functions for the Streamlit Emergency Vehicle Rerouting app.

Contains coordinate extraction, event styling, bounding box computation,
Folium simulation map rendering, and benchmark comparison table generation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import folium
import networkx as nx


def node_path_to_coords(
    graph: nx.DiGraph,
    path: Sequence[Any],
) -> List[List[float]]:
    """Convert an ordered sequence of node IDs into Leaflet [lat, lon] coordinates."""
    coords: List[List[float]] = []
    for node in path:
        if node in graph:
            data = graph.nodes[node]
            coords.append([float(data["y"]), float(data["x"])])
    return coords


def event_to_coords(
    graph: nx.DiGraph,
    event: Any,
) -> Optional[List[List[float]]]:
    """Extract [[lat_u, lon_u], [lat_v, lon_v]] coordinates for an event edge."""
    if event.u not in graph or event.v not in graph:
        return None
    u_data = graph.nodes[event.u]
    v_data = graph.nodes[event.v]
    return [
        [float(u_data["y"]), float(u_data["x"])],
        [float(v_data["y"]), float(v_data["x"])],
    ]


def format_event_tooltip(event: Any, base_cost: float) -> str:
    """Format an informative tooltip for an event edge showing step, kind, and costs."""
    cost_str = "Blocked (inf)" if math.isinf(event.new_cost) else f"{event.new_cost:.1f}s"
    multiplier_str = ""
    if not math.isinf(event.new_cost) and base_cost > 0:
        mult = event.new_cost / base_cost
        multiplier_str = f" ({mult:.1f}x)"
    return (
        f"Event @ Step {event.step}: {event.kind.capitalize()} | "
        f"Base: {base_cost:.1f}s -> New: {cost_str}{multiplier_str}"
    )


def compute_route_bounds(
    coord_lists: Sequence[Sequence[Sequence[float]]],
) -> Optional[List[List[float]]]:
    """Compute bounding box [[min_lat, min_lon], [max_lat, max_lon]] enclosing all routes."""
    all_lats: List[float] = []
    all_lons: List[float] = []

    for coords in coord_lists:
        for pt in coords:
            all_lats.append(pt[0])
            all_lons.append(pt[1])

    if not all_lats or not all_lons:
        return None

    return [
        [min(all_lats), min(all_lons)],
        [max(all_lats), max(all_lons)],
    ]


def build_comparison_rows(
    baseline_result: Dict[str, Any],
    dstar_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Format comparative agent performance records for tabular display."""
    def _mean_lat(res: Dict[str, Any]) -> float:
        replans = int(res.get("replans", 0))
        total_time = float(res.get("total_replan_time_ms", 0.0))
        latencies = res.get("replan_latencies_ms", [])
        if replans > 0 and latencies:
            return float(sum(latencies) / len(latencies))
        if replans > 0 and total_time > 0:
            return float(total_time / replans)
        return 0.0

    return [
        {
            "Algorithm": "Baseline (Full A* Recompute)",
            "Status": str(baseline_result.get("status", "unknown")),
            "Steps": int(baseline_result.get("steps", 0)),
            "Replans": int(baseline_result.get("replans", 0)),
            "Replan Expansions": int(baseline_result.get("replan_expansions", 0)),
            "Total Replan Time (ms)": round(float(baseline_result.get("total_replan_time_ms", 0.0)), 2),
            "Mean Replan Latency (ms)": round(_mean_lat(baseline_result), 2),
            "Travel Time (s)": round(float(baseline_result.get("travel_time", 0.0)), 2),
        },
        {
            "Algorithm": "D* Lite (Incremental Replanning)",
            "Status": str(dstar_result.get("status", "unknown")),
            "Steps": int(dstar_result.get("steps", 0)),
            "Replans": int(dstar_result.get("replans", 0)),
            "Replan Expansions": int(dstar_result.get("replan_expansions", 0)),
            "Total Replan Time (ms)": round(float(dstar_result.get("total_replan_time_ms", 0.0)), 2),
            "Mean Replan Latency (ms)": round(_mean_lat(dstar_result), 2),
            "Travel Time (s)": round(float(dstar_result.get("travel_time", 0.0)), 2),
        },
    ]


def compute_summary_sentence(
    baseline_result: Dict[str, Any],
    dstar_result: Dict[str, Any],
) -> str:
    """Generate a high-level summary statement comparing expansions."""
    b_exp = int(baseline_result.get("replan_expansions", 0))
    d_exp = int(dstar_result.get("replan_expansions", 0))

    if d_exp == 0 and b_exp == 0:
        return "Neither algorithm required replan expansions (no dynamic obstacles impacted the path)."
    if d_exp == 0 and b_exp > 0:
        return f"D* Lite repaired the route without node expansions, while Baseline A* required {b_exp} expansions."
    if b_exp > d_exp:
        ratio = b_exp / d_exp
        return f"D* Lite needed {ratio:.2f}x fewer expansions to replan compared to Baseline A*."
    if b_exp < d_exp:
        ratio = d_exp / b_exp
        return f"Baseline A* expanded {ratio:.2f}x fewer nodes than D* Lite on this specific scenario."
    return f"Both algorithms required an identical number of replan expansions ({b_exp})."


def check_travel_time_discrepancy(
    baseline_result: Dict[str, Any],
    dstar_result: Dict[str, Any],
    tol: float = 1e-6,
) -> Tuple[bool, float]:
    """Check if agent travel times differ by more than tolerance."""
    b_tt = float(baseline_result.get("travel_time", 0.0))
    d_tt = float(dstar_result.get("travel_time", 0.0))

    if math.isinf(b_tt) or math.isinf(d_tt):
        return (b_tt != d_tt, float("inf"))

    diff = abs(b_tt - d_tt)
    return (diff > tol, diff)


def build_simulation_map(
    graph: nx.DiGraph,
    sim: Dict[str, Any],
    default_zoom: int = 12,
) -> folium.Map:
    """Construct a Folium map depicting start/goal, routes, dynamic disruptions, and legend."""
    start, goal = sim["start"], sim["goal"]
    start_loc = [float(graph.nodes[start]["y"]), float(graph.nodes[start]["x"])]
    goal_loc = [float(graph.nodes[goal]["y"]), float(graph.nodes[goal]["x"])]

    m = folium.Map(location=start_loc, zoom_start=default_zoom, tiles="OpenStreetMap")

    folium.Marker(
        location=start_loc,
        popup=f"Dispatch Start: {start}",
        tooltip="Dispatch Origin",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)
    folium.Marker(
        location=goal_loc,
        popup=f"Incident Goal: {goal}",
        tooltip="Incident Goal",
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
    ).add_to(m)

    init_coords = node_path_to_coords(graph, sim.get("init_path", []))
    if init_coords:
        folium.PolyLine(
            locations=init_coords,
            color="blue",
            weight=4,
            opacity=0.8,
            dash_array="6, 8",
            tooltip="Initial Plan (A*)",
        ).add_to(m)

    dstar_coords = node_path_to_coords(graph, sim.get("dstar", {}).get("path_taken", []))
    if dstar_coords:
        folium.PolyLine(
            locations=dstar_coords,
            color="green",
            weight=4,
            opacity=0.8,
            tooltip="D* Lite Traversed Path",
        ).add_to(m)

    base_path = sim.get("baseline", {}).get("path_taken", [])
    dstar_path = sim.get("dstar", {}).get("path_taken", [])
    base_coords = node_path_to_coords(graph, base_path)
    if base_path != dstar_path and base_coords:
        folium.PolyLine(
            locations=base_coords,
            color="orange",
            weight=3,
            opacity=0.8,
            dash_array="4, 6",
            tooltip="Baseline A* Traversed Path",
        ).add_to(m)

    for ev in sim.get("events", []):
        edge_pts = event_to_coords(graph, ev)
        if not edge_pts:
            continue
        base_cost = float(graph[ev.u][ev.v].get("travel_time", 1.0)) if graph.has_edge(ev.u, ev.v) else 1.0
        color = "red" if ev.kind == "closure" else "orange"
        folium.PolyLine(
            locations=edge_pts,
            color=color,
            weight=6,
            opacity=0.9,
            tooltip=format_event_tooltip(ev, base_cost),
        ).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom: 25px; left: 25px; z-index:9999;
                background-color: white; border: 2px solid #888; border-radius: 5px;
                padding: 8px 12px; font-size: 12px; box-shadow: 2px 2px 6px rgba(0,0,0,0.25);">
      <b>Legend</b><br>
      <span style="color: blue;">▬ ▬</span> Initial Plan (A*)<br>
      <span style="color: green;">▬▬</span> D* Lite Traversed<br>
      <span style="color: orange;">▬ ▬</span> Baseline Traversed (if differs)<br>
      <span style="color: red;">▬▬</span> Road Closure<br>
      <span style="color: orange;">▬▬</span> Congestion (1.5x - 5x)<br>
      <span style="color: green;">●</span> Origin &nbsp; <span style="color: red;">●</span> Incident Goal
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    bounds = compute_route_bounds([init_coords, dstar_coords, base_coords])
    if bounds:
        m.fit_bounds(bounds, padding=[25, 25])

    return m
