"""Benchmark script comparing Baseline A* dynamic agent against D* Lite on real network."""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt

from src.agent_baseline import run_baseline_agent
from src.agent_dstar import run_dstar_agent
from src.graph_utils import make_heuristic
from src.simulator import make_scenario

CSV_COLUMNS = [
    "seed",
    "algorithm",
    "status",
    "steps",
    "replans",
    "initial_expansions",
    "replan_expansions",
    "total_expansions",
    "travel_time",
    "total_replan_time_ms",
    "mean_replan_latency_ms",
]


def build_csv_row(seed: int, algorithm: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an agent result dictionary into a flat row matching CSV schema.

    Calculates mean_replan_latency_ms as total_replan_time_ms / replans (or 0.0 if no replans).
    """
    replans = int(result.get("replans", 0))
    latencies = result.get("replan_latencies_ms", [])
    total_time = float(result.get("total_replan_time_ms", 0.0))

    if replans > 0 and latencies:
        mean_latency = float(sum(latencies) / len(latencies))
    elif replans > 0 and total_time > 0.0:
        mean_latency = float(total_time / replans)
    else:
        mean_latency = 0.0

    return {
        "seed": seed,
        "algorithm": algorithm,
        "status": str(result.get("status", "unknown")),
        "steps": int(result.get("steps", 0)),
        "replans": replans,
        "initial_expansions": int(result.get("initial_expansions", 0)),
        "replan_expansions": int(result.get("replan_expansions", 0)),
        "total_expansions": int(result.get("total_expansions", 0)),
        "travel_time": float(result.get("travel_time", 0.0)),
        "total_replan_time_ms": total_time,
        "mean_replan_latency_ms": mean_latency,
    }


def compute_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute benchmark summary statistics across baseline and D* Lite runs."""
    base_rows = [r for r in rows if r["algorithm"] == "baseline"]
    dstar_rows = [r for r in rows if r["algorithm"] == "dstar"]

    base_exp = [r["replan_expansions"] for r in base_rows]
    dstar_exp = [r["replan_expansions"] for r in dstar_rows]

    base_lat = [r["total_replan_time_ms"] for r in base_rows]
    dstar_lat = [r["total_replan_time_ms"] for r in dstar_rows]

    # Calculate means and medians
    b_mean_exp = statistics.mean(base_exp) if base_exp else 0.0
    b_med_exp = statistics.median(base_exp) if base_exp else 0.0
    d_mean_exp = statistics.mean(dstar_exp) if dstar_exp else 0.0
    d_med_exp = statistics.median(dstar_exp) if dstar_exp else 0.0

    b_mean_lat = statistics.mean(base_lat) if base_lat else 0.0
    b_med_lat = statistics.median(base_lat) if base_lat else 0.0
    d_mean_lat = statistics.mean(dstar_lat) if dstar_lat else 0.0
    d_med_lat = statistics.median(dstar_lat) if dstar_lat else 0.0

    # Speed-up ratios (baseline / dstar)
    exp_speedup = (b_mean_exp / d_mean_exp) if d_mean_exp > 0 else float("inf")
    lat_speedup = (b_mean_lat / d_mean_lat) if d_mean_lat > 0 else float("inf")

    # Seeds with status != "reached" (grouped by seed)
    seeds = sorted(list({r["seed"] for r in rows}))
    unreached_count = 0
    travel_time_diff_count = 0

    for s in seeds:
        b_r = next((r for r in base_rows if r["seed"] == s), None)
        d_r = next((r for r in dstar_rows if r["seed"] == s), None)
        if not b_r or not d_r:
            continue

        if b_r["status"] != "reached" or d_r["status"] != "reached":
            unreached_count += 1

        b_tt = b_r["travel_time"]
        d_tt = d_r["travel_time"]
        if math.isinf(b_tt) or math.isinf(d_tt):
            if b_tt != d_tt:
                travel_time_diff_count += 1
        elif abs(b_tt - d_tt) > 1e-6:
            travel_time_diff_count += 1

    return {
        "baseline_mean_expansions": b_mean_exp,
        "baseline_median_expansions": b_med_exp,
        "dstar_mean_expansions": d_mean_exp,
        "dstar_median_expansions": d_med_exp,
        "baseline_mean_latency_ms": b_mean_lat,
        "baseline_median_latency_ms": b_med_lat,
        "dstar_mean_latency_ms": d_mean_lat,
        "dstar_median_latency_ms": d_med_lat,
        "expansions_speedup": exp_speedup,
        "latency_speedup": lat_speedup,
        "unreached_count": unreached_count,
        "travel_time_diff_count": travel_time_diff_count,
    }


def plot_benchmark(rows: List[Dict[str, Any]], output_png: Path) -> None:
    """Generate side-by-side bar charts comparing expansions and latency with error bars."""
    base_rows = [r for r in rows if r["algorithm"] == "baseline"]
    dstar_rows = [r for r in rows if r["algorithm"] == "dstar"]

    b_exp = [r["replan_expansions"] for r in base_rows]
    d_exp = [r["replan_expansions"] for r in dstar_rows]
    b_lat = [r["total_replan_time_ms"] for r in base_rows]
    d_lat = [r["total_replan_time_ms"] for r in dstar_rows]

    b_mean_exp = statistics.mean(b_exp) if b_exp else 0.0
    d_mean_exp = statistics.mean(d_exp) if d_exp else 0.0
    b_sem_exp = (statistics.stdev(b_exp) / math.sqrt(len(b_exp))) if len(b_exp) > 1 else 0.0
    d_sem_exp = (statistics.stdev(d_exp) / math.sqrt(len(d_exp))) if len(d_exp) > 1 else 0.0

    b_mean_lat = statistics.mean(b_lat) if b_lat else 0.0
    d_mean_lat = statistics.mean(d_lat) if d_lat else 0.0
    b_sem_lat = (statistics.stdev(b_lat) / math.sqrt(len(b_lat))) if len(b_lat) > 1 else 0.0
    d_sem_lat = (statistics.stdev(d_lat) / math.sqrt(len(d_lat))) if len(d_lat) > 1 else 0.0

    algorithms = ["Baseline (A*)", "D* Lite"]
    exp_means = [b_mean_exp, d_mean_exp]
    exp_sems = [b_sem_exp, d_sem_exp]
    lat_means = [b_mean_lat, d_mean_lat]
    lat_sems = [b_sem_lat, d_sem_lat]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Subplot 1: Replan Expansions
    bars1 = ax1.bar(algorithms, exp_means, yerr=exp_sems, capsize=6, color=["#1f77b4", "#ff7f0e"], alpha=0.85)
    ax1.set_ylabel("Mean Replan Expansions", fontsize=11)
    ax1.set_title("Replan Node Expansions (Mean ± SE)", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f"{height:.1f}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10)

    # Subplot 2: Replan Latency (ms)
    bars2 = ax2.bar(algorithms, lat_means, yerr=lat_sems, capsize=6, color=["#1f77b4", "#ff7f0e"], alpha=0.85)
    ax2.set_ylabel("Mean Replan Latency (ms)", fontsize=11)
    ax2.set_title("Replan Latency (Mean ± SE)", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f"{height:.2f} ms",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run comprehensive benchmark comparing Baseline A* vs D* Lite.")
    parser.add_argument("--seeds", type=int, default=30, help="Number of seeds to evaluate (default: 30)")
    parser.add_argument("--events", type=int, default=6, help="Number of dynamic events per scenario (default: 6)")
    parser.add_argument("--out", type=str, default="results", help="Output directory for CSV and PNG (default: results)")
    args = parser.parse_args()

    out_dir = project_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / "benchmark.csv"
    png_file = out_dir / "benchmark.png"

    # 1. Load data/app_graph.pkl
    graph_path = project_root / "data" / "app_graph.pkl"
    if not graph_path.exists():
        print(f"Error: Prepared graph {graph_path} not found. Run scripts/export_app_graph.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading prepared app graph from {graph_path}...")
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    heuristic = make_heuristic(graph)
    print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.\n")

    # 2. Warm-up run so first measurement is not skewed by compilation/initialization
    print("Performing throwaway warm-up run...")
    try:
        w_start, w_goal, w_events = make_scenario(graph, heuristic, seed=9999, n_events=args.events, min_hops=15)
        run_baseline_agent(graph, w_start, w_goal, w_events, heuristic, verify=False)
        run_dstar_agent(graph, w_start, w_goal, w_events, heuristic, verify=False)
    except Exception:
        pass
    print("Warm-up complete.\n")

    # 3. Main benchmark loop
    print(f"Running benchmark across {args.seeds} seeds (n_events={args.events}, verify=False)...")
    rows: List[Dict[str, Any]] = []
    scenarios_run: List[tuple[int, Any, Any, list]] = []

    for seed in range(args.seeds):
        try:
            start, goal, events = make_scenario(
                graph,
                heuristic=heuristic,
                seed=seed,
                n_events=args.events,
                min_hops=20,
            )
        except ValueError as e:
            print(f"Skipping seed {seed}: {e}")
            continue

        scenarios_run.append((seed, start, goal, events))

        # Run Baseline Agent (timed, verify=False)
        b_res = run_baseline_agent(
            graph=graph,
            start=start,
            goal=goal,
            events=events,
            heuristic=heuristic,
            verify=False,
        )
        rows.append(build_csv_row(seed=seed, algorithm="baseline", result=b_res))

        # Run D* Lite Agent (timed, verify=False)
        d_res = run_dstar_agent(
            graph=graph,
            start=start,
            goal=goal,
            events=events,
            heuristic=heuristic,
            verify=False,
        )
        rows.append(build_csv_row(seed=seed, algorithm="dstar", result=d_res))

        print(f"  Seed {seed:2d}: baseline_exp={b_res['replan_expansions']:5d}, "
              f"dstar_exp={d_res['replan_expansions']:4d}, "
              f"base_time={b_res['total_replan_time_ms']:5.2f}ms, "
              f"dstar_time={d_res['total_replan_time_ms']:5.2f}ms")

    # 4. Separately re-run first 5 seeds with verify=True
    print("\nRunning separate verification check on first 5 seeds (verify=True)...")
    verification_passed = True
    verified_seeds = 0

    for seed, start, goal, events in scenarios_run[:5]:
        try:
            run_baseline_agent(graph, start, goal, events, heuristic, verify=True)
            run_dstar_agent(graph, start, goal, events, heuristic, verify=True)
            verified_seeds += 1
        except Exception as e:
            print(f"Verification FAILED on seed {seed}: {e}")
            verification_passed = False
            break

    if verification_passed and verified_seeds > 0:
        print(f"Dijkstra cross-check on first {verified_seeds} seeds: verified OK\n")

    # 5. Write results/benchmark.csv
    print(f"Writing results to {csv_file}...")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # 6. Generate results/benchmark.png
    print(f"Generating plot at {png_file}...")
    plot_benchmark(rows, png_file)

    # 7. Print summary
    summary = compute_summary(rows)
    print("\n================== BENCHMARK SUMMARY ==================")
    print(f"Total Scenarios Evaluated: {len(scenarios_run)}")
    print(f"Baseline Mean Replan Expansions: {summary['baseline_mean_expansions']:.2f}")
    print(f"Baseline Median Replan Expansions: {summary['baseline_median_expansions']:.2f}")
    print(f"D* Lite  Mean Replan Expansions: {summary['dstar_mean_expansions']:.2f}")
    print(f"D* Lite  Median Replan Expansions: {summary['dstar_median_expansions']:.2f}")
    print(f"Expansion Speed-up (Baseline / D*): {summary['expansions_speedup']:.2f}x\n")

    print(f"Baseline Mean Replan Latency: {summary['baseline_mean_latency_ms']:.2f} ms")
    print(f"Baseline Median Replan Latency: {summary['baseline_median_latency_ms']:.2f} ms")
    print(f"D* Lite  Mean Replan Latency: {summary['dstar_mean_latency_ms']:.2f} ms")
    print(f"D* Lite  Median Replan Latency: {summary['dstar_median_latency_ms']:.2f} ms")
    print(f"Latency Speed-up (Baseline / D*): {summary['latency_speedup']:.2f}x\n")

    print(f"Scenarios with status != 'reached': {summary['unreached_count']}")
    print(f"Scenarios with travel_time diff > 1e-6: {summary['travel_time_diff_count']}")
    print("=======================================================")


if __name__ == "__main__":
    main()

