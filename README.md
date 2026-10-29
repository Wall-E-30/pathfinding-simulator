# Dynamic Emergency Vehicle Rerouting on Live Street Networks

A pathfinding simulation project evaluating static and dynamic routing algorithms for emergency vehicles navigating live urban road networks. Built with Python 3.11, NetworkX, OSMnx, and pytest.

---

## Shared Interface Contract

All routing algorithms adhere strictly to the following specification:

### 1. Graph Representation
- **Type**: `networkx.DiGraph` (directed graph).
- **Edges**: Exactly **one** directed edge per `(u, v)` pair.
- **Edge Attribute**: `"travel_time"` (travel time in seconds, non-negative float).
- **Node Attributes**: `"x"` (longitude in degrees) and `"y"` (latitude in degrees).

### 2. Algorithm Interface
Every routing algorithm exposes a standard `plan` function:
```python
def plan(
    graph: nx.DiGraph,
    start: Any,
    goal: Any,
    heuristic: Callable[[Any, Any], float] | None = None
) -> tuple[list[Any], float, dict[str, Any]]:
    ...
```

#### Return Value
- `path`: List of node IDs from `start` to `goal`, inclusive (e.g., `[start, ..., goal]`).
- `cost`: Total travel time along the path in seconds (`float`).
- `stats`: Execution statistics dictionary:
  ```python
  {
      "expansions": int,   # Total number of nodes popped and expanded from priority queue
      "time_ms": float     # Total wall-clock execution time in milliseconds
  }
  ```

#### Edge Cases
- **Unreachable Goal**: Returns `([], float("inf"), stats)`.
- **Identical Start and Goal (`start == goal`)**: Returns `([start], 0.0, stats)`.

### 3. Heuristic Function
For informed search algorithms (such as A*), the admissible travel-time heuristic is defined as:
```python
h(n, goal) = haversine_m(n, goal) / v_max
```
- `haversine_m`: Great-circle distance in metres between node `n` and `goal`.
- `v_max`: Maximum permissible traversal speed (m/s) across all edges in the graph.
- Units: Seconds.
- Guarantee: Admissible and consistent because straight-line distance divided by the network's maximum theoretical speed never overestimates the true travel time.
