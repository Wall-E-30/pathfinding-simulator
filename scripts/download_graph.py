"""Download a drivable street network around a geographic coordinate using OSMnx."""

from pathlib import Path
import osmnx as ox

# ============================================================================
# Geographic Coordinates and Download Parameters
# (Fill in your desired latitude and longitude here)
# ============================================================================
LAT: float = 23.0225   # Latitude in decimal degrees
LON: float = 72.5714   # Longitude in decimal degrees
DIST: int = 1500       # Search radius in metres around (LAT, LON)
NETWORK_TYPE: str = "drive"


def main() -> None:
    # Resolve paths relative to project root (parent directory of scripts/)
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "city.graphml"

    print(f"Downloading drivable network around ({LAT}, {LON}) with radius {DIST}m...")
    graph = ox.graph_from_point(
        (LAT, LON),
        dist=DIST,
        network_type=NETWORK_TYPE,
    )

    # Save to data/city.graphml
    ox.save_graphml(graph, filepath=output_path)

    node_count = len(graph.nodes)
    edge_count = len(graph.edges)

    print("\nGraph download complete!")
    print(f"Saved to: {output_path}")
    print(f"Node count: {node_count}")
    print(f"Edge count: {edge_count}")

    # Check against target size (1,000 - 2,000 nodes)
    TARGET_MIN = 1000
    TARGET_MAX = 2000

    if node_count > TARGET_MAX:
        print(
            f"\n[Notice] The graph has {node_count} nodes, which is larger than the target "
            f"range of {TARGET_MIN}-{TARGET_MAX} nodes.\n"
            f"To reduce the size, decrease the 'DIST' constant at the top of scripts/download_graph.py\n"
            f"(e.g., reduce DIST from {DIST} to 1000 or 800) and re-run."
        )
    elif node_count < TARGET_MIN:
        print(
            f"\n[Notice] The graph has {node_count} nodes, which is smaller than the target "
            f"range of {TARGET_MIN}-{TARGET_MAX} nodes.\n"
            f"To increase the size, increase the 'DIST' constant at the top of scripts/download_graph.py\n"
            f"(e.g., increase DIST from {DIST} to 2000) and re-run."
        )
    else:
        print(f"\n[Success] Graph size ({node_count} nodes) is within the target range ({TARGET_MIN}-{TARGET_MAX}).")


if __name__ == "__main__":
    main()

