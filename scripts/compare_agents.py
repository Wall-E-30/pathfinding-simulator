"""Compare baseline A* dynamic agent and D* Lite dynamic agent on real street network.

Loads city.graphml, prepares the graph and heuristic once, runs scenarios for
seeds 0 through 4 with both agents under verify=True, and compares results.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agent_baseline import run_baseline_agent
from src.agent_dstar import run_dstar_agent
from src.graph_utils import load_graph, make_heuristic, prepare_graph
from src.simulator import make_scenario


def main() -> None:
    graph_path = project_root / "data" / "city.graphml"
    if not graph_path.exists():
        print(f"Error: Graph file not found at {graph_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading street network from {graph_path}...")
    raw_graph = load_graph(graph_path)

    print("Preparing road network DiGraph (speeds, travel times, SCC)...")
    graph = prepare_graph(raw_graph)
    print(f"Network ready: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.\n")

    heuristic = make_heuristic(graph)

    seeds = list(range(5))
    diff_count = 0
    stuck_mismatch_count = 0

    print("seed, baseline status, dstar status, baseline travel_time, dstar travel_time, baseline replan_expansions, dstar replan_expansions, replans (both)")

    for seed in seeds:
        start, goal, events = make_scenario(
            graph,
            heuristic=heuristic,
            seed=seed,
            n_events=6,
            min_hops=30,
        )

        b_res = run_baseline_agent(
            graph=graph,
            start=start,
            goal=goal,
            events=events,
            heuristic=heuristic,
            verify=True,
        )

        d_res = run_dstar_agent(
            graph=graph,
            start=start,
            goal=goal,
            events=events,
            heuristic=heuristic,
            verify=True,
        )

        b_status = b_res["status"]
        d_status = d_res["status"]
        b_time = b_res["travel_time"]
        d_time = d_res["travel_time"]
        b_exp = b_res["replan_expansions"]
        d_exp = d_res["replan_expansions"]
        b_replans = b_res["replans"]
        d_replans = d_res["replans"]

        replans_str = f"baseline={b_replans}, dstar={d_replans}" if b_replans != d_replans else f"{b_replans}"

        print(
            f"{seed}, {b_status}, {d_status}, {b_time:.6f}, {d_time:.6f}, {b_exp}, {d_exp}, {replans_str}"
        )

        # Difference in travel time by more than 1e-6
        if math.isinf(b_time) or math.isinf(d_time):
            if b_time != d_time:
                diff_count += 1
        elif abs(b_time - d_time) > 1e-6:
            diff_count += 1

        # Exactly one agent stuck
        b_stuck = (b_status == "stuck")
        d_stuck = (d_status == "stuck")
        if b_stuck != d_stuck:
            stuck_mismatch_count += 1

    print(f"\nNumber of seeds where travel_times differ by > 1e-6: {diff_count}")
    print(f"Number of seeds where exactly one agent is stuck: {stuck_mismatch_count}")


if __name__ == "__main__":
    main()

