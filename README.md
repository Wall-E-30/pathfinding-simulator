# Dynamic Emergency Vehicle Rerouting on Live Street Networks

A pathfinding simulation framework comparing static and dynamic graph routing algorithms for emergency vehicles navigating live urban road networks. The system benchmarks full A* recomputation against incremental D* Lite replanning under dynamic real-time traffic disruptions. Built with Python 3.11+, NetworkX, OSMnx, Matplotlib, and Streamlit.

---

## Live Web Application

- **Live URL**: [https://pathfinding-simulator.streamlit.app/](https://pathfinding-simulator.streamlit.app/)
- Hosted on Streamlit Community Cloud with interactive Leaflet/Folium map rendering for route visualization and dynamic incident inspection.

---

## Setup & Installation

### Requirements
- **Python**: 3.11 or newer (compatible with Python 3.11–3.14).
- Clone the repository and install the dependencies:

```bash
git clone https://github.com/Wall-E-30/pathfinding-simulator.git
cd pathfinding-simulator
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to Run

### 1. Run Unit Tests
To execute the complete unit test suite across all modules:
```bash
pytest -v
```

### 2. Regenerate the App Graph
To crop and export the lightweight strongly connected component graph from raw data:
```bash
python scripts/export_app_graph.py
```
This loads `data/city.graphml`, crops within a 2500m radius of central Ahmedabad (`23.0225, 72.5714`), extracts the largest SCC, strips unused attributes, and serializes to `data/app_graph.pkl` (protocol 4, < 25 MB).

### 3. Run the CLI Simulation Demo
To run a dynamic emergency routing scenario on the full city network:
```bash
python scripts/run_scenario.py --seed 0 --events 6
```

### 4. Run the Comparative Benchmark
To execute the automated benchmark comparing Baseline A* vs. D* Lite across multiple random seeds:
```bash
python scripts/benchmark.py --seeds 30 --events 6 --out results
```
Generates tabular data in `results/benchmark.csv` and side-by-side comparison charts in `results/benchmark.png`.

### 5. Run the Streamlit Web Application
To run the interactive dashboard locally:
```bash
streamlit run app/app.py
```

---

## Shared Interface Contract

All routing algorithms adhere strictly to the shared contract:

### 1. Static Planning Interface
```python
def plan(
    graph: nx.DiGraph,
    start: Any,
    goal: Any,
    heuristic: Callable[[Any, Any], float] | None = None
) -> tuple[list[Any], float, dict[str, Any]]:
```
- **Returns**: `(path, cost, stats)` where `stats` contains `{"expansions": int, "time_ms": float}`.
- Unreachable goals return `([], float("inf"), stats)`.
- Identical start and goal (`start == goal`) returns `([start], 0.0, stats)`.

### 2. Dynamic Agent Interface
```python
def run_baseline_agent(graph, start, goal, events, heuristic, verify=False) -> dict: ...
def run_dstar_agent(graph, start, goal, events, heuristic, verify=False) -> dict: ...
```
- Works on `graph.copy()`, leaving caller's graph unmutated.
- When `verify=True`, asserts after every replan that the planner's cost-to-goal matches an independent NetworkX Dijkstra search on non-`inf` edges.
- **Return Dictionary Keys**:
  ```python
  {
      "status": "reached" | "stuck" | "max_steps",
      "path_taken": list[Any],
      "travel_time": float,          # Sum of edge costs at moment traversed
      "steps": int,
      "replans": int,
      "initial_expansions": int,
      "replan_expansions": int,
      "total_expansions": int,
      "replan_latencies_ms": list[float],
      "total_replan_time_ms": float,
  }
  ```

---

## Time Model and Cost Semantics

1. **Edge Cost Semantics**:
   - **Road Closure**: `travel_time = float("inf")`. All algorithms treat `inf` edges as impassable/blocked.
   - **Congestion**: `new_cost = base_cost * multiplier` where `multiplier ∈ [1.5, 5.0]` and `base_cost` is the edge cost in the original graph.
   - **Admissibility Condition**: Edge costs never drop below `base_cost`, ensuring the Euclidean/Haversine heuristic remains strictly admissible.
2. **Discrete Time Model**:
   - Counter $t \ge 1$ tracks moves made.
   - Events with `step == t` are applied when the vehicle reaches its $t$-th node, **before** selecting its next move.
   - **Omniscient Sensing**: All network disruptions are sensed immediately by the agent upon occurrence.

---

## Benchmark Results

Comprehensive 30-seed benchmark metrics comparing Baseline full A* recomputation against incremental D* Lite:
- See [`results/benchmark.csv`](results/benchmark.csv) for raw per-seed run data.
- See [`results/benchmark.png`](results/benchmark.png) for graphical comparison of mean replan expansions and latencies.

---

## Limitations and Future Work

1. **No ALT or Bidirectional A\***: A* currently uses forward search with a Haversine heuristic; landmark-based (ALT) or bidirectional dynamic search is not yet implemented.
2. **Single-Vehicle Focus**: Current routing algorithms model a single emergency vehicle; multi-vehicle fleet coordination and congestion feedback loops are omitted.
3. **Synthetic Incidents**: Road closures and congestion events are synthetically generated using seeded pseudo-random patterns rather than live municipal traffic feeds.
4. **Full-Information Instant Sensing**: Changes are assumed to be sensed immediately across the network without communication horizons, sensor latency, or radio range limits.
5. **Memory Profiling**: Performance evaluation benchmarks node expansions and execution latency; memory heap footprint profiling of D* Lite priority queues has not been quantified.
