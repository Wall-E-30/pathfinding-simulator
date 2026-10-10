"""Graph utility functions for road network processing and pathfinding.

Provides utilities for:
- Computing Haversine great-circle distances in metres.
- Imputing edge speeds and calculating travel times using OSMnx.
- Collapsing parallel directed edges by retaining the minimum travel time.
- Computing maximum traversal speed (v_max) in metres/second for admissible heuristics.
- Snapping geographical coordinates to the nearest road network node.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union

import networkx as nx
import osmnx as ox

from src.graph import Graph, _validate_cost

# Mean Earth radius in metres (WGS 84 / IUGG standard)
EARTH_RADIUS_M: float = 6_371_000.0


def haversine(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    radius: float = EARTH_RADIUS_M,
) -> float:
    """Compute the great-circle distance between two geographic coordinates in metres.

    Uses the Haversine formula:
        a = sin²(Δφ/2) + cos(φ1) * cos(φ2) * sin²(Δλ/2)
        c = 2 * atan2(√a, √(1−a))
        d = R * c

    Args:
        lat1: Latitude of the first point in decimal degrees [-90.0, 90.0].
        lon1: Longitude of the first point in decimal degrees [-180.0, 180.0].
        lat2: Latitude of the second point in decimal degrees [-90.0, 90.0].
        lon2: Longitude of the second point in decimal degrees [-180.0, 180.0].
        radius: Earth radius in metres (defaults to 6,371,000.0 m).

    Returns:
        Great-circle distance between the two points in metres.

    Raises:
        TypeError: If coordinates or radius are not numeric.
        ValueError: If coordinates are outside valid geographic ranges or non-finite.
    """
    for name, val in [("lat1", lat1), ("lon1", lon1), ("lat2", lat2), ("lon2", lon2), ("radius", radius)]:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be numeric, got {type(val).__name__}")
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"{name} must be a finite number, got {val}")

    if not (-90.0 <= lat1 <= 90.0):
        raise ValueError(f"lat1 must be in [-90.0, 90.0], got {lat1}")
    if not (-90.0 <= lat2 <= 90.0):
        raise ValueError(f"lat2 must be in [-90.0, 90.0], got {lat2}")
    if not (-180.0 <= lon1 <= 180.0):
        raise ValueError(f"lon1 must be in [-180.0, 180.0], got {lon1}")
    if not (-180.0 <= lon2 <= 180.0):
        raise ValueError(f"lon2 must be in [-180.0, 180.0], got {lon2}")
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    # Identical points have 0 distance
    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2)
    )
    # Clamp to [0.0, 1.0] to guard against floating-point inaccuracies
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return float(radius * c)


def add_edge_speeds(
    G: nx.MultiDiGraph,
    hwy_speeds: Dict[str, float] | None = None,
    fallback: float | None = 30.0,
    **kwargs: Any,
) -> nx.MultiDiGraph:
    """Impute edge speeds (km/h) across graph edges using ox.add_edge_speeds().

    Args:
        G: Input NetworkX MultiDiGraph (road network).
        hwy_speeds: Lookup mapping of highway types to speeds in km/h.
        fallback: Default fallback speed in km/h when maxspeed is missing (defaults to 30.0).
        **kwargs: Additional parameters forwarded to ox.add_edge_speeds().

    Returns:
        The modified graph G with 'speed_kph' edge attributes.
    """
    if "crs" not in G.graph:
        G.graph["crs"] = "epsg:4326"
    return ox.add_edge_speeds(G, hwy_speeds=hwy_speeds, fallback=fallback, **kwargs)


def add_edge_travel_times(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Calculate travel times (seconds) for all edges using ox.add_edge_travel_times().

    Requires each edge to have 'length' (metres) and 'speed_kph' attributes.
    Calculates: travel_time = length / (speed_kph * 1000 / 3600).

    Args:
        G: Input NetworkX MultiDiGraph with 'speed_kph' and 'length' attributes.

    Returns:
        The modified graph G with 'travel_time' edge attributes in seconds.
    """
    return ox.add_edge_travel_times(G)


def add_edge_speeds_and_travel_times(
    G: nx.MultiDiGraph,
    hwy_speeds: Dict[str, float] | None = None,
    fallback: float | None = 30.0,
) -> nx.MultiDiGraph:
    """Add edge speeds and calculate edge travel times in sequence.

    Uses ox.add_edge_speeds() followed by ox.add_edge_travel_times().

    Args:
        G: Input NetworkX MultiDiGraph.
        hwy_speeds: Lookup mapping of highway types to speed in km/h.
        fallback: Fallback speed in km/h when highway speed is unmapped.

    Returns:
        The graph G populated with both 'speed_kph' and 'travel_time' edge attributes.
    """
    G = add_edge_speeds(G, hwy_speeds=hwy_speeds, fallback=fallback)
    G = add_edge_travel_times(G)
    return G


def collapse_parallel_edges(
    G: nx.MultiDiGraph,
    weight_attr: str = "travel_time",
    as_custom_graph: bool = False,
) -> Union[nx.DiGraph, Graph]:
    """Collapse parallel directed edges in a MultiDiGraph using minimum travel time (seconds).

    For every pair of nodes (u, v) with one or more directed edges, retains the edge
    with the minimum value of weight_attr (travel time).

    Args:
        G: Input NetworkX MultiDiGraph.
        weight_attr: Edge attribute representing travel cost (default 'travel_time').
        as_custom_graph: If True, returns a src.graph.Graph instance;
                         if False, returns a NetworkX DiGraph preserving all edge metadata.

    Returns:
        nx.DiGraph or Graph containing exactly one directed edge per (u, v) pair.
    """
    if as_custom_graph:
        return Graph.from_osmnx(G, weight_attr=weight_attr)

    di_graph = nx.DiGraph()
    di_graph.graph.update(G.graph)

    # Preserve all node attributes
    for node, data in G.nodes(data=True):
        di_graph.add_node(node, **data)

    # Group parallel edges and keep the one with minimum travel time
    for u, v, data in G.edges(data=True):
        if weight_attr not in data:
            continue
        cost = _validate_cost(data[weight_attr])

        if di_graph.has_edge(u, v):
            if cost < di_graph[u][v][weight_attr]:
                # Overwrite with faster edge attributes
                edge_data = dict(data)
                edge_data[weight_attr] = cost
                di_graph[u][v].update(edge_data)
        else:
            edge_data = dict(data)
            edge_data[weight_attr] = cost
            di_graph.add_edge(u, v, **edge_data)

    return di_graph


def compute_v_max(G: Any) -> float:
    """Compute the maximum travel speed (v_max) across all edges in metres per second.

    Used by heuristic pathfinding algorithms (such as A*) to calculate admissible
    travel-time heuristics:
        h(u, goal) = haversine(u, goal) / v_max

    Speed is computed from:
    1. Edge 'speed_kph' converted to m/s: (speed_kph * 1000 / 3600 = speed_kph / 3.6).
    2. Edge ratio: length (metres) / travel_time (seconds) if length and travel_time exist.

    Args:
        G: NetworkX MultiDiGraph, DiGraph, or custom Graph.

    Returns:
        Maximum traversal speed in metres per second (float).

    Raises:
        ValueError: If the graph has no edges or no valid positive speed attributes.
    """
    max_mps: float = 0.0

    if isinstance(G, (nx.MultiDiGraph, nx.DiGraph, nx.Graph)):
        for _, _, data in G.edges(data=True):
            edge_speed_mps: float | None = None

            # Check explicit speed_kph
            if "speed_kph" in data:
                raw_speed = data["speed_kph"]
                if isinstance(raw_speed, (int, float)) and raw_speed > 0:
                    edge_speed_mps = float(raw_speed) / 3.6
                elif isinstance(raw_speed, Iterable) and not isinstance(raw_speed, (str, bytes)):
                    valid_speeds = [float(s) for s in raw_speed if isinstance(s, (int, float)) and s > 0]
                    if valid_speeds:
                        edge_speed_mps = max(valid_speeds) / 3.6

            # Check length and travel_time ratio
            if edge_speed_mps is None and "length" in data and "travel_time" in data:
                length = float(data["length"])
                travel_time = float(data["travel_time"])
                if travel_time > 0 and length > 0:
                    edge_speed_mps = length / travel_time

            if edge_speed_mps is not None and edge_speed_mps > max_mps:
                max_mps = edge_speed_mps

    elif isinstance(G, Graph):
        # Inspect edge costs and node coordinates if available
        for u in G.nodes:
            u_data = G.get_node_data(u)
            if "x" not in u_data or "y" not in u_data:
                continue
            u_lat, u_lon = float(u_data["y"]), float(u_data["x"])

            for v, cost in G.get_neighbors(u).items():
                if cost <= 0:
                    continue
                v_data = G.get_node_data(v)
                if "x" not in v_data or "y" not in v_data:
                    continue
                v_lat, v_lon = float(v_data["y"]), float(v_data["x"])

                dist_m = haversine(u_lat, u_lon, v_lat, v_lon)
                speed_mps = dist_m / cost
                if speed_mps > max_mps:
                    max_mps = speed_mps

    if max_mps <= 0.0:
        raise ValueError("Could not determine valid positive edge speeds from the provided graph.")

    return max_mps


def snap_to_node(
    G: Any,
    lat: float,
    lon: float,
    return_dist: bool = False,
) -> Any:
    """Snap a geographic coordinate (lat, lon) to the nearest road network node.

    Uses OSMnx nearest-node functionality (ox.nearest_nodes).

    Args:
        G: NetworkX MultiDiGraph, DiGraph, or custom Graph containing node coordinates.
        lat: Target latitude in decimal degrees [-90.0, 90.0].
        lon: Target longitude in decimal degrees [-180.0, 180.0].
        return_dist: If True, returns a tuple of (node_id, distance_in_metres).
                     Defaults to False (returns node_id only).

    Returns:
        Nearest node identifier, or (node_id, dist_m) if return_dist=True.

    Raises:
        TypeError: If lat or lon are not numeric.
        ValueError: If lat or lon are outside geographic bounds or non-finite.
    """
    for name, val in [("lat", lat), ("lon", lon)]:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be numeric, got {type(val).__name__}")
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"{name} must be a finite number, got {val}")

    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"lat must be in [-90.0, 90.0], got {lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"lon must be in [-180.0, 180.0], got {lon}")

    # NetworkX / OSMnx graphs
    if isinstance(G, (nx.MultiDiGraph, nx.DiGraph, nx.Graph)):
        # OSMnx nearest_nodes expects X=lon, Y=lat
        result = ox.nearest_nodes(G, X=float(lon), Y=float(lat), return_dist=return_dist)
        return result

    # Custom Graph class fallback
    if isinstance(G, Graph):
        best_node = None
        best_dist = float("inf")

        for node in G.nodes:
            nd = G.get_node_data(node)
            if "x" in nd and "y" in nd:
                d = haversine(lat, lon, float(nd["y"]), float(nd["x"]))
                if d < best_dist:
                    best_dist = d
                    best_node = node

        if best_node is None:
            raise ValueError("Graph contains no nodes with 'x' and 'y' coordinates.")

        if return_dist:
            return best_node, best_dist
        return best_node

    raise TypeError(f"Unsupported graph type: {type(G).__name__}")
