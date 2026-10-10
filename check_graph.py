
from pathlib import Path
import osmnx as ox

project_root = Path(__file__).resolve().parent
graph_path = project_root / "data" / "city.graphml"

G = ox.load_graphml(graph_path)

print("Graph loaded successfully!")
print("Nodes:", len(G.nodes))
print("Edges:", len(G.edges))
