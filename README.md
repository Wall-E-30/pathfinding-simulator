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


---

## Interactive Streamlit Web Application

An interactive web interface is provided in `app/app.py` for map visualization and simulation inspection.

### Running Locally
To run the Streamlit application:
```bash
streamlit run app/app.py
```

### Deployment on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app** and select the repository: `Wall-E-30/pathfinding-simulator`.
3. Set **Main file path** to:
   ```text
   app/app.py
   ```
4. Under **Advanced settings**, set the **Python version** to **3.11** or **3.12**.
5. Click **Deploy!**
   - The deployment will use `app/requirements.txt` which contains only lightweight packages (`streamlit`, `folium`, `streamlit-folium`, `networkx`, `numpy`), completely avoiding heavy GIS compilation issues.
