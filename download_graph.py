from pathlib import Path
import osmnx as ox

# Project root is the folder containing this script
project_root = Path(__file__).resolve().parent

# Save the graph directly into the data folder
data_dir = project_root / "data"
data_dir.mkdir(parents=True, exist_ok=True)

graph_path = data_dir / "city.graphml"

# Configure OSMnx settings for reliability
# The default overpass-api.de has intermittent connection timeouts on some IP routes;
# lz4.overpass-api.de is an official high-speed mirror.
ox.settings.overpass_url = "https://lz4.overpass-api.de/api"
ox.settings.use_cache = True
ox.settings.log_console = True

# Download Ahmedabad City driving road network
# Note: "Ahmedabad, Gujarat, India" as a place query resolves in OSM to the entire
# 8,707 km² rural administrative district, triggering Overpass timeouts (504).
# A 10 km radius around Ahmedabad city center covers the full urban road network.
AHMEDABAD_CENTER = (23.0225, 72.5714)
RADIUS_METERS = 10000

print(f"Downloading road network for Ahmedabad (radius={RADIUS_METERS}m)...")
G = ox.graph.graph_from_point(
    AHMEDABAD_CENTER,
    dist=RADIUS_METERS,
    network_type="drive"
)

# Save the graph
print(f"Saving graph to {graph_path}...")
ox.save_graphml(G, filepath=graph_path)

print("Graph downloaded and saved successfully!")
print("Location:", graph_path)
print("Nodes:", len(G.nodes))
print("Edges:", len(G.edges))
