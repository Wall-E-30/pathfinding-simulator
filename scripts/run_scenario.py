"""Command-line script to run dynamic benchmark scenarios on real street networks."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# Set UTF-8 encoding output for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path for direct invocation
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agent_baseline import run_baseline_agent
from src.astar import plan
from src.events import apply_events
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
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-event details (step, edge, closure/congestion, replan expansions) and decision sequences.",
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

    # 4. If verbose, print scheduled events table
    if args.verbose:
        print("\n--- Scheduled Dynamic Traffic Events ---")
        for i, ev in enumerate(events, 1):
            base_tt = float(graph[ev.u][ev.v].get("travel_time", 1.0)) if graph.has_edge(ev.u, ev.v) else 1.0
            cost_str = "inf (Blocked)" if math.isinf(ev.new_cost) else f"{ev.new_cost:.2f}s"
            mult_str = "" if math.isinf(ev.new_cost) else f" ({ev.new_cost / base_tt:.1f}x)"
            print(
                f"  Event {i}: Step {ev.step:2d} | Edge ({ev.u}, {ev.v}) | "
                f"Kind: {ev.kind:10s} | Base: {base_tt:6.2f}s -> New: {cost_str}{mult_str}"
            )

    # 5. Run baseline A* dynamic agent with Dijkstra cross-verification
    print("\nRunning baseline A* dynamic agent (verify=True)...")
    result = run_baseline_agent(
        graph=graph,
        start=start,
        goal=goal,
        events=events,
        heuristic=heuristic,
        verify=True,
    )

    # 6. If verbose, print step-by-step decision sequence
    if args.verbose:
        print("\n--- Step-by-Step Decision Sequence ---")
        g_trace = graph.copy()
        replan_count = 0
        events_by_step = {}
        for ev in events:
            events_by_step.setdefault(ev.step, []).append(ev)

        path_taken = result["path_taken"]
        total_steps = result["steps"]

        for t in range(total_steps):
            cur_node = path_taken[t]
            next_node = path_taken[t + 1] if t + 1 < len(path_taken) else None

            # Check if any events trigger at step t
            step_events = events_by_step.get(t, [])
            if step_events:
                changes = apply_events(g_trace, events, step=t)
                if changes:
                    replan_count += 1
                    t0 = time.perf_counter()
                    new_path, new_cost, new_stats = plan(g_trace, cur_node, goal, heuristic)
                    ms = (time.perf_counter() - t0) * 1000.0

                    print(f"\n[Step {t:2d}] Vehicle at Node {cur_node}")
                    for u, v, old_c, new_c in changes:
                        ev_item = next(e for e in step_events if e.u == u and e.v == v)
                        c_str = "inf (Blocked Road)" if math.isinf(new_c) else f"{new_c:.2f}s ({new_c/old_c:.1f}x)"
                        print(f"  [DISRUPTION] {ev_item.kind.upper()} on Edge ({u} -> {v})")
                        print(f"               Base travel time: {old_c:.2f}s -> New travel time: {c_str}")

                    exp = new_stats["expansions"]
                    print(
                        f"  [REPLAN #{replan_count}] (A* Full Recompute): "
                        f"expanded {exp} nodes in {ms:.2f}ms | "
                        f"New route length: {len(new_path)} hops | "
                        f"Remaining travel time: {new_cost:.2f}s"
                    )
                    print(f"  [DECISION] Moving from {cur_node} -> {next_node}")

        print(f"\nFinal destination reached at step {total_steps}: Node {result['path_taken'][-1]} == Goal {goal}")

    # 7. Output required summary metrics
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
