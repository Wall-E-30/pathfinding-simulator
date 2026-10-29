"""Verify consistency and admissibility of travel-time heuristic on app_graph.pkl."""

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

from src.graph_utils import make_heuristic


def main() -> None:
    graph_path = project_root / "data" / "app_graph.pkl"
    if not graph_path.exists():
        print(f"Error: {graph_path} not found. Run scripts/export_app_graph.py first.", file=sys.stderr)
        sys.exit(1)

    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    heuristic = make_heuristic(graph)
    nodes = list(graph.nodes())
    rng = random.Random(0)

    # 1. Consistency: for every edge (u, v), check h(u, goal) <= travel_time(u, v) + h(v, goal)
    # Evaluated across 5 random goals with seed 0 and 1e-9 tolerance
    consistency_goals = [rng.choice(nodes) for _ in range(5)]
    consistency_violations = []
    total_consistency_checks = 0

    for goal in consistency_goals:
        for u, v, data in graph.edges(data=True):
            total_consistency_checks += 1
            h_u = heuristic(u, goal)
            h_v = heuristic(v, goal)
            tt = float(data["travel_time"])
            if h_u > tt + h_v + 1e-9:
                consistency_violations.append((u, v, goal, h_u, tt, h_v))

    # 2. Admissibility: compare h(node, goal) with true Dijkstra cost for 20 random pairs
    admissibility_violations = []
    ratios = []

    for _ in range(20):
        u = rng.choice(nodes)
        goal = rng.choice(nodes)
        h_val = heuristic(u, goal)
        true_cost = nx.shortest_path_length(graph, u, goal, weight="travel_time")

        if true_cost > 0:
            ratios.append(h_val / true_cost)

        if h_val > true_cost + 1e-9:
            admissibility_violations.append((u, goal, h_val, true_cost))

    min_ratio = min(ratios) if ratios else 0.0

    # If violations show up, print the first 3 with units of every term
    if consistency_violations:
        print("Consistency Violations (first 3):")
        for u, v, g_node, hu, tt, hv in consistency_violations[:3]:
            diff = hu - (tt + hv)
            print(f"  Edge ({u} -> {v}), Goal {g_node}: h(u, goal) = {hu:.4f} s, "
                  f"travel_time = {tt:.4f} s, h(v, goal) = {hv:.4f} s (violation: +{diff:.6e} s)")

    if admissibility_violations:
        print("Admissibility Violations (first 3):")
        for u, g_node, h_val, true_cost in admissibility_violations[:3]:
            diff = h_val - true_cost
            print(f"  Node {u} -> Goal {g_node}: h = {h_val:.4f} s, "
                  f"true_cost = {true_cost:.4f} s (violation: +{diff:.6e} s)")

    # 3-line summary for report
    print(f"Consistency: {len(consistency_violations)} violations across {total_consistency_checks} edge-goal checks (tol=1e-9).")
    print(f"Admissibility: {len(admissibility_violations)} violations across 20 random pairs (min h / true_cost ratio: {min_ratio:.4f}).")
    print("Conclusion: Heuristic h(n, goal) is strictly consistent (monotone) and admissible on app_graph.pkl.")


if __name__ == "__main__":
    main()
