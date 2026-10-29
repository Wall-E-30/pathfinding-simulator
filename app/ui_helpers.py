"""Pure helper functions for Streamlit map visualization and metrics comparison.

Kept in a standalone module free from Streamlit dependencies so they can be
cleanly unit tested with pytest.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional
import networkx as nx

from src.events import Event


def build_polyline_coords(graph: nx.DiGraph, path: List[Any]) -> List[List[float]]:
    """Convert a sequence of node IDs into Leaflet [lat, lon] coordinate pairs.

    Viva note:
        Node attributes store geographic coordinates as 'y' (latitude) and 'x' (longitude).
        Folium/Leaflet polylines expect coordinates in [lat, lon] order.
    """
    coords: List[List[float]] = []
    for node in path:
        if node in graph:
            data = graph.nodes[node]
            if "y" in data and "x" in data:
                coords.append([float(data["y"]), float(data["x"])])
    return coords


def build_event_edges_data(
    graph: nx.DiGraph,
    events: List[Event],
) -> List[dict[str, Any]]:
    """Extract coordinate pairs, styling, and informative tooltips for traffic events.

    Viva note:
        Closures represent impassable links (infinite cost) styled in solid red.
        Congestion events represent speed drops (higher finite cost) styled in orange.
    """
    event_data: List[dict[str, Any]] = []

    for ev in events:
        if ev.u in graph and ev.v in graph:
            u_data = graph.nodes[ev.u]
            v_data = graph.nodes[ev.v]
            if "y" in u_data and "x" in u_data and "y" in v_data and "x" in v_data:
                coords = [
                    [float(u_data["y"]), float(u_data["x"])],
                    [float(v_data["y"]), float(v_data["x"])],
                ]

                # Base cost in original graph if present
                base_cost = (
                    graph[ev.u][ev.v].get("travel_time", 1.0)
                    if graph.has_edge(ev.u, ev.v)
                    else 1.0
                )

                is_closure = ev.kind == "closure" or math.isinf(ev.new_cost)
                color = "red" if is_closure else "#FF8C00"
                kind_label = "Closure" if is_closure else "Congestion"

                cost_str = (
                    "blocked (∞)"
                    if math.isinf(ev.new_cost)
                    else f"{ev.new_cost:.1f}s"
                )
                tooltip = (
                    f"Step {ev.step}: {kind_label} on ({ev.u} → {ev.v}) | "
                    f"New: {cost_str} vs Base: {base_cost:.1f}s"
                )

                event_data.append(
                    {
                        "coords": coords,
                        "color": color,
                        "kind": ev.kind,
                        "tooltip": tooltip,
                        "step": ev.step,
                    }
                )

    return event_data


def build_map_bounds(coords: List[List[float]]) -> Optional[List[List[float]]]:
    """Compute min/max [[south, west], [north, east]] bounds to frame a route."""
    if not coords:
        return None

    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


def format_comparison_table(
    baseline_res: dict[str, Any],
    dstar_res: dict[str, Any],
) -> dict[str, List[Any]]:
    """Format simulation results into a comparison dataframe dictionary.

    Metrics include: status, steps, replans, replan_expansions, total_replan_time_ms,
    mean replan latency, and final travel time.
    """
    base_replans = max(1, baseline_res.get("replans", 0))
    dstar_replans = max(1, dstar_res.get("replans", 0))

    base_mean_lat = (
        sum(baseline_res.get("replan_latencies_ms", [])) / len(baseline_res["replan_latencies_ms"])
        if baseline_res.get("replan_latencies_ms")
        else baseline_res.get("total_replan_time_ms", 0.0) / base_replans
    )
    dstar_mean_lat = (
        sum(dstar_res.get("replan_latencies_ms", [])) / len(dstar_res["replan_latencies_ms"])
        if dstar_res.get("replan_latencies_ms")
        else dstar_res.get("total_replan_time_ms", 0.0) / dstar_replans
    )

    metrics = [
        "Status",
        "Steps Taken",
        "Replans Triggered",
        "Replan Expansions",
        "Total Replan Time (ms)",
        "Mean Replan Latency (ms)",
        "Total Travel Time (s)",
    ]

    base_vals = [
        str(baseline_res.get("status", "unknown")),
        int(baseline_res.get("steps", 0)),
        int(baseline_res.get("replans", 0)),
        int(baseline_res.get("replan_expansions", 0)),
        f"{baseline_res.get('total_replan_time_ms', 0.0):.2f}",
        f"{base_mean_lat:.2f}",
        f"{baseline_res.get('travel_time', 0.0):.2f}",
    ]

    dstar_vals = [
        str(dstar_res.get("status", "unknown")),
        int(dstar_res.get("steps", 0)),
        int(dstar_res.get("replans", 0)),
        int(dstar_res.get("replan_expansions", 0)),
        f"{dstar_res.get('total_replan_time_ms', 0.0):.2f}",
        f"{dstar_mean_lat:.2f}",
        f"{dstar_res.get('travel_time', 0.0):.2f}",
    ]

    return {
        "Metric": metrics,
        "Baseline (A*)": base_vals,
        "D* Lite": dstar_vals,
    }


def compute_expansion_sentence(baseline_exp: int, dstar_exp: int) -> str:
    """Generate a readable viva summary comparing replanning expansions."""
    if baseline_exp == 0 and dstar_exp == 0:
        return "D* Lite and Baseline required 0 expansions to replan (no events affected the path)."

    if dstar_exp == 0 and baseline_exp > 0:
        return f"D* Lite needed 0 expansions to replan, whereas Baseline expanded {baseline_exp} nodes."

    if baseline_exp >= dstar_exp:
        ratio = baseline_exp / max(1, dstar_exp)
        return f"D* Lite needed {ratio:.1f}x fewer expansions to replan than baseline A*."

    ratio = dstar_exp / max(1, baseline_exp)
    return f"Baseline A* needed {ratio:.1f}x fewer expansions to replan than D* Lite."


def compute_travel_time_warning(
    baseline_tt: float,
    dstar_tt: float,
    tol: float = 1e-6,
) -> Optional[str]:
    """Return a warning string if travel times differ by more than tolerance."""
    diff = abs(baseline_tt - dstar_tt)
    if diff > tol:
        return (
            f"Notice: Travel times differ by {diff:.4f}s (> {tol}s). "
            f"Baseline: {baseline_tt:.2f}s, D* Lite: {dstar_tt:.2f}s."
        )
    return None


def get_legend_html() -> str:
    """HTML widget overlay providing a map legend for route polylines and incident pins."""
    return """
    <div style="
        position: fixed;
        bottom: 25px;
        right: 25px;
        z-index: 1000;
        background-color: white;
        padding: 10px 14px;
        border: 2px solid #ccc;
        border-radius: 6px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
        font-family: sans-serif;
        font-size: 12px;
        line-height: 18px;
    ">
        <b>Simulation Legend</b><br>
        <span style="color: green; font-size: 16px;">●</span> Dispatch Start (Green)<br>
        <span style="color: red; font-size: 16px;">●</span> Incident Goal (Red)<br>
        <span style="color: blue; font-weight: bold;">---</span> Initial Planned Path<br>
        <span style="color: green; font-weight: bold;">━━━</span> D* Lite Actual Path<br>
        <span style="color: orange; font-weight: bold;">━━━</span> Baseline Path (if different)<br>
        <span style="color: red; font-weight: bold;">━━━</span> Road Closure (∞)<br>
        <span style="color: #FF8C00; font-weight: bold;">━━━</span> Road Congestion (Traffic)
    </div>
    """
