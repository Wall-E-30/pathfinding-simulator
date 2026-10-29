"""Export a cropped, lightweight road network graph for the Streamlit app and benchmarks.

Loads data/city.graphml, prepares the DiGraph with travel times, crops to a 2500m
radius around central Ahmedabad (23.0225, 72.5714), extracts the largest strongly
connected component (SCC), strips unused attributes, and serializes to
data/app_graph.pkl with pickle protocol 4.
"""

from __future__ import annotations

import math
import pickle
import random
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import networkx as nx

from src.astar import plan
from src.graph_utils import haversine_m, load_graph, make_heuristic, prepare_graph

# Cropping parameters
RADIUS_M: float = 2500.0
CENTER_LAT: float = 23.0225
CENTER_LON: float = 72.5714


def main() -> None:
    raw_path = project_root / "data" / "city.graphml"
    out_path = project_root / "data" / "app_graph.pkl"

    if not raw_path.exists():
        print(f"Error: Raw graph file not found at {raw_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading street network from {raw_path}...")
    raw_graph = load_graph(raw_path)

    print("Preparing road network DiGraph (speeds, travel times, SCC)...")
    full_graph = prepare_graph(raw_graph)
    print(f"Full network: {full_graph.number_of_nodes()} nodes, {full_graph.number_of_edges()} edges.")

    # 1. Filter nodes within RADIUS_M of center coordinates using haversine_m
    print(f"\nCropping network to radius of {RADIUS_M}m around ({CENTER_LAT}, {CENTER_LON})...")
    nodes_within_radius = [
        node for node, data in full_graph.nodes(data=True)
        if haversine_m(CENTER_LAT, CENTER_LON, float(data["y"]), float(data["x"])) <= RADIUS_M
    ]

    # Create induced subgraph
    sub_graph = full_graph.subgraph(nodes_within_radius).copy()

    # 2. Extract largest strongly connected component (SCC) to guarantee mutual reachability
    if len(sub_graph) > 0 and not nx.is_strongly_connected(sub_graph):
        largest_scc = max(nx.strongly_connected_components(sub_graph), key=len)
        sub_graph = sub_graph.subgraph(largest_scc).copy()

    node_count = sub_graph.number_of_nodes()
    edge_count = sub_graph.number_of_edges()
    print(f"Cropped SCC network: {node_count} nodes, {edge_count} edges.")

    # Check target node count range (8,000 - 12,000 nodes)
    target_min, target_max = 8000, 12000
    if not (target_min <= node_count <= target_max):
        print(
            f"NOTE: Resulting node count ({node_count}) is outside the target range [{target_min}, {target_max}].\n"
            f"To adjust node count, change RADIUS_M at the top of scripts/export_app_graph.py "
            f"(e.g., increase if < {target_min}, decrease if > {target_max})."
        )
    else:
        print(f"Node count ({node_count}) is within the target range [{target_min}, {target_max}].")

    # 3. Keep only the attributes the app needs: node x, y; edge travel_time (floats)
    # Dropping geometry, names, highway types, etc., to minimize serialized size
    print("Stripping extra attributes and building clean DiGraph...")
    clean_graph = nx.DiGraph()
    clean_graph.graph["name"] = "Ahmedabad Cropped Street Network"
    clean_graph.graph["center_lat"] = CENTER_LAT
    clean_graph.graph["center_lon"] = CENTER_LON
    clean_graph.graph["radius_m"] = RADIUS_M

    for node, data in sub_graph.nodes(data=True):
        clean_graph.add_node(
            node,
            x=float(data["x"]),
            y=float(data["y"]),
        )

    for u, v, data in sub_graph.edges(data=True):
        clean_graph.add_edge(
            u,
            v,
            travel_time=float(data["travel_time"]),
        )

    # 4. Save with pickle protocol 4 for cross-version Python compatibility
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving graph to {out_path} with pickle protocol 4...")
    with open(out_path, "wb") as f:
        pickle.dump(clean_graph, f, protocol=4)

    size_bytes = out_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"Exported app graph file size: {size_mb:.2f} MB ({size_bytes} bytes).")

    if size_mb > 25.0:
        print(f"WARNING: app_graph.pkl is {size_mb:.2f} MB, which exceeds the 25 MB warning threshold!")
    else:
        print("File size is comfortably under the 25 MB threshold.")

    # 5. Sanity check: reload pickle and run astar.plan on 5 random pairs (seed 0)
    print("\nRunning sanity check: reloading pickle and verifying 5 random pairs against Dijkstra...")
    with open(out_path, "rb") as f:
        reloaded_graph = pickle.load(f)

    assert reloaded_graph.number_of_nodes() == clean_graph.number_of_nodes()
    assert reloaded_graph.number_of_edges() == clean_graph.number_of_edges()

    rng = random.Random(0)
    all_nodes = list(reloaded_graph.nodes())
    heuristic = make_heuristic(reloaded_graph)

    for i in range(5):
        s = rng.choice(all_nodes)
        t = rng.choice(all_nodes)
        path, cost, stats = plan(reloaded_graph, s, t, heuristic=heuristic)
        nx_cost = nx.shortest_path_length(reloaded_graph, s, t, weight="travel_time")
        assert math.isclose(cost, nx_cost, rel_tol=1e-5, abs_tol=1e-6), (
            f"Cost mismatch on pair {s} -> {t}: astar={cost}, dijkstra={nx_cost}"
        )
        assert len(path) >= 1
        print(f"  Check {i+1}: Node {s} -> {t} | cost = {cost:.2f}s | expansions = {stats['expansions']} [OK]")

    print("\nAll sanity checks passed successfully!")


if __name__ == "__main__":
    main()

