"""Command-line script to run dynamic benchmark scenarios on real street networks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for direct invocation
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agent_baseline import run_baseline_agent
from src.graph_utils import load_graph, make_heuristic, prepare_graph
from src.simulator import make_scenario


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run dynamic emergency vehicle routing simulation on real street network."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for scenario generation (default: 0)",
    )
    parser.add_argument(
        "--events",
        type=int,
        default=6,
        help="Number of dynamic traffic events to synthesize (default: 6)",
    )
    args = parser.parse_args()

    # 1. Resolve path to data/city.graphml
    graph_path = project_root / "data" / "city.graphml"
    if not graph_path.exists():
        print(f"Error: Graph file not found at {graph_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading street network from {graph_path}...")
    raw_graph = load_graph(graph_path)

    print("Preparing road network DiGraph (speeds, travel times, SCC)...")
    graph = prepare_graph(raw_graph)
    print(f"Network ready: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

    # 2. Build admissible travel-time heuristic h(n, goal) = haversine / v_max
    heuristic = make_heuristic(graph)

    # 3. Generate benchmark scenario with seed and event count
    print(f"Generating benchmark scenario (seed={args.seed}, n_events={args.events})...")
    start, goal, events = make_scenario(
        graph,
        heuristic=heuristic,
        seed=args.seed,
        n_events=args.events,
        min_hops=30,
    )
    print(f"Scenario generated: Start={start}, Goal={goal}, Events={len(events)}")

    # 4. Run baseline A* dynamic agent with Dijkstra cross-verification
    print("Running baseline A* dynamic agent (verify=True)...")
    result = run_baseline_agent(
        graph=graph,
        start=start,
        goal=goal,
        events=events,
        heuristic=heuristic,
        verify=True,
    )

    # 5. Output required metrics
    print("\n--- Simulation Results ---")
    print(f"status: {result['status']}")
    print(f"steps: {result['steps']}")
    print(f"replans: {result['replans']}")
    print(f"initial_expansions: {result['initial_expansions']}")
    print(f"replan_expansions: {result['replan_expansions']}")
    print(f"travel_time: {result['travel_time']:.2f}")
    print(f"total_replan_time_ms: {result['total_replan_time_ms']:.2f}")


if __name__ == "__main__":
    main()
