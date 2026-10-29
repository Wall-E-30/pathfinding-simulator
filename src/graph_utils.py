"""Graph utility functions for road network processing and pathfinding.

Provides:
- haversine_m: Great-circle distance between coordinates in metres.
- load_graph: Loads a GraphML file offline and converts string attributes to numeric.
- prepare_graph: Converts MultiDiGraph to a strongly connected DiGraph with travel times.
- compute_v_max: Calculates maximum network speed (m/s) for admissible heuristics.
- make_heuristic: Generates admissible travel-time heuristic function h(n, goal).
- snap_to_node: Snaps geographic coordinates to the nearest network node.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, Union

import networkx as nx
import osmnx as ox

# Mean Earth radius in metres (WGS 84 / IUGG standard)
EARTH_RADIUS_M: float = 6_371_000.0


def haversine_m(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    radius: float = EARTH_RADIUS_M,
) -> float:
    """Compute the great-circle distance between two geographic coordinates in metres.

    Viva note:
        Uses the Haversine formula on spherical coordinates:
        a = sin²(Δφ/2) + cos(φ1) * cos(φ2) * sin²(Δλ/2)
        c = 2 * atan2(√a, √(1−a))
        d = R * c
        Clamping 'a' to [0.0, 1.0] guards against micro floating-point rounding errors
        that would otherwise produce ValueError when calculating sqrt(1 - a).
    """
    for name, val in [("lat1", lat1), ("lon1", lon1), ("lat2", lat2), ("lon2", lon2)]:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be numeric, got {type(val).__name__}")
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"{name} must be finite, got {val}")

    if not (-90.0 <= lat1 <= 90.0) or not (-90.0 <= lat2 <= 90.0):
        raise ValueError("Latitude must be in [-90.0, 90.0]")
    if not (-180.0 <= lon1 <= 180.0) or not (-180.0 <= lon2 <= 180.0):
        raise ValueError("Longitude must be in [-180.0, 180.0]")

    # Identical coordinates have 0.0 distance
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
    # Clamp to [0.0, 1.0] for floating point domain safety
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return float(radius * c)


# Backward-compatible alias
haversine = haversine_m


def load_graph(path: Union[str, Path]) -> nx.MultiDiGraph:
    """Load an OSMnx street network from a GraphML file offline.

    Viva note:
        GraphML format stores all node and edge attributes as strings.
        We convert coordinates ('x', 'y') and numerical edge properties ('length')
        to proper Python floats so geometric calculations do not fail with string errors.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Graph file not found: {file_path}")

    # Load offline GraphML structure (fallback to nx.read_graphml for non-integer node IDs)
    try:
        G = ox.load_graphml(file_path)
    except (ValueError, TypeError):
        G = nx.read_graphml(file_path)

    # Cast node coordinates to floats
    for _, data in G.nodes(data=True):
        if "x" in data:
            data["x"] = float(data["x"])
        if "y" in data:
            data["y"] = float(data["y"])

    # Cast numeric edge attributes
    for _, _, data in G.edges(data=True):
        if "length" in data:
            try:
                data["length"] = float(data["length"])
            except (ValueError, TypeError):
                pass
        if "speed_kph" in data:
            try:
                data["speed_kph"] = float(data["speed_kph"])
            except (ValueError, TypeError):
                pass

    return G


def prepare_graph(G: nx.MultiDiGraph) -> nx.DiGraph:
    """Prepare a road network for routing algorithms.

    Steps:
    1. Impute edge speeds (km/h) and travel times (seconds).
    2. Collapse parallel directed edges by retaining the minimum travel_time per (u, v).
    3. Retain node 'x' and 'y' attributes.
    4. Extract the largest strongly connected component (SCC) so every node pair is reachable.

    Viva note:
        Urban street networks have one-way streets and dead ends. Filtering to the
        largest Strongly Connected Component (SCC) ensures any node in the graph
        can reach any other node, preventing unreachable goal failures in evaluation.
    """
    # 1. Ensure edge speeds and travel times exist
    # Only impute speeds if edges lack speed_kph
    has_speeds = any("speed_kph" in data for _, _, data in G.edges(data=True))
    if not has_speeds:
        G = ox.add_edge_speeds(G, fallback=30.0)

    # Only impute travel times if edges lack travel_time
    has_tt = any("travel_time" in data for _, _, data in G.edges(data=True))
    if not has_tt:
        G = ox.add_edge_travel_times(G)

    # 2. Collapse parallel edges into a DiGraph (keep min travel_time)
    di_g = nx.DiGraph()
    di_g.graph.update(G.graph)

    # Copy nodes with float coordinates
    for node, data in G.nodes(data=True):
        node_attrs = dict(data)
        if "x" in node_attrs:
            node_attrs["x"] = float(node_attrs["x"])
        if "y" in node_attrs:
            node_attrs["y"] = float(node_attrs["y"])
        di_g.add_node(node, **node_attrs)

    # Collapse parallel edges: group by (u, v) and pick minimum travel_time
    for u, v, data in G.edges(data=True):
        if "travel_time" in data:
            tt = float(data["travel_time"])
        elif "length" in data and "speed_kph" in data:
            # travel_time = length (m) / speed (m/s)
            speed_mps = float(data["speed_kph"]) / 3.6
            tt = float(data["length"]) / speed_mps if speed_mps > 0 else 1.0
        else:
            tt = 1.0

        if tt <= 0:
            tt = 1.0  # Guard against non-positive travel times

        if di_g.has_edge(u, v):
            if tt < di_g[u][v]["travel_time"]:
                edge_data = dict(data)
                edge_data["travel_time"] = tt
                di_g[u][v].update(edge_data)
        else:
            edge_data = dict(data)
            edge_data["travel_time"] = tt
            di_g.add_edge(u, v, **edge_data)

    # 3. Extract the largest Strongly Connected Component (SCC)
    if len(di_g) > 0 and not nx.is_strongly_connected(di_g):
        largest_scc_nodes = max(nx.strongly_connected_components(di_g), key=len)
        di_g = di_g.subgraph(largest_scc_nodes).copy()

    return di_g


def compute_v_max(G: Union[nx.DiGraph, nx.MultiDiGraph]) -> float:
    """Compute the maximum traversal speed (v_max) across all edges in metres per second.

    Viva note:
        For A* heuristic admissibility, h(n, goal) = distance(n, goal) / v_max.
        By dividing the Euclidean/Haversine distance by the highest possible edge speed
        anywhere in the network, h(n, goal) is guaranteed to never overestimate the actual
        travel time, satisfying the admissibility condition h(n) <= h*(n).
    """
    max_mps: float = 0.0

    for _, _, data in G.edges(data=True):
        edge_speed_mps: float | None = None

        # Check explicit speed_kph
        if "speed_kph" in data:
            raw_speed = data["speed_kph"]
            if isinstance(raw_speed, (int, float)) and raw_speed > 0:
                edge_speed_mps = float(raw_speed) / 3.6

        # Ratio check: length / travel_time
        if edge_speed_mps is None and "length" in data and "travel_time" in data:
            try:
                length = float(data["length"])
                travel_time = float(data["travel_time"])
                if travel_time > 0 and length > 0:
                    edge_speed_mps = length / travel_time
            except (ValueError, TypeError):
                pass

        if edge_speed_mps is not None and edge_speed_mps > max_mps:
            max_mps = edge_speed_mps

    if max_mps <= 0.0:
        # Fallback to standard 60 km/h (16.67 m/s) if no speed could be inferred
        return 60.0 / 3.6

    return max_mps


def make_heuristic(G: nx.DiGraph) -> Callable[[Any, Any], float]:
    """Return an admissible travel-time heuristic function h(n, goal) in seconds.

    Viva note:
        Pre-caching node coordinates into a dictionary gives O(1) coordinate lookups
        during priority queue expansions, avoiding overhead in A*.
        h(n, goal) = haversine_m(lat_n, lon_n, lat_g, lon_g) / v_max.
    """
    v_max = compute_v_max(G)

    # Pre-cache coordinates: node_id -> (lat, lon)
    coords: Dict[Any, tuple[float, float]] = {}
    for node, data in G.nodes(data=True):
        coords[node] = (float(data["y"]), float(data["x"]))

    def heuristic(n: Any, goal: Any) -> float:
        if n == goal:
            return 0.0
        lat_n, lon_n = coords[n]
        lat_g, lon_g = coords[goal]
        dist_m = haversine_m(lat_n, lon_n, lat_g, lon_g)
        return dist_m / v_max

    return heuristic


def snap_to_node(
    G: Union[nx.DiGraph, nx.MultiDiGraph],
    lat: float,
    lon: float,
) -> Any:
    """Snap geographic coordinates (lat, lon) to the nearest road network node ID.

    Viva note:
        Uses Haversine great-circle distance to find the closest node in G.
        Works across both MultiDiGraph and DiGraph.
    """
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        raise ValueError("Coordinates out of geographic range")

    best_node = None
    min_dist = float("inf")

    for node, data in G.nodes(data=True):
        if "x" in data and "y" in data:
            dist = haversine_m(lat, lon, float(data["y"]), float(data["x"]))
            if dist < min_dist:
                min_dist = dist
                best_node = node

    if best_node is None:
        raise ValueError("Graph contains no nodes with coordinates.")

    return best_node
