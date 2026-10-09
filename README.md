# pathfinding-simulator
Graph algorithms, dynamic pathfinding simulation, and performance metrics.
## Interface Contract

### 1. Graph Representation

* The graph is a directed, weighted graph.
* Each directed `(u, v)` pair stores exactly one travel cost.
* Edge weights represent travel time in seconds.
* When converting an OSMnx `MultiDiGraph`, multiple edges between the same nodes must be collapsed by retaining the edge with the minimum travel time.
* If an edge does not exist, it must not be treated as a valid connection.

### 2. Graph API

```python
graph.update_edge(u, v, new_cost)
```

* Updates the cost of the directed edge `(u, v)`.
* `new_cost` must be a non-negative number.
* The updated cost must be used by subsequent planning calls.
* Document the behavior when the specified edge does not exist.

### 3. Algorithm API

```python
path, cost, stats = algo.plan(graph, start, goal)
```

* `path`: list of node IDs from `start` to `goal`, inclusive.
* `cost`: total travel time along the returned path, in seconds.
* `stats`: dictionary containing:

  * `expansions`: number of nodes expanded during the search.
  * `time_ms`: algorithm execution time in milliseconds.
* If the goal is unreachable, return an empty path and `float("inf")` as the cost.
* If `start == goal`, return `[start]` and cost `0`.

### 4. Statistics Format

All algorithms must return the same statistics keys:

```python
{
    "expansions": 0,
    "time_ms": 0.0
}
```

`expansions` must be an integer and `time_ms` must be a non-negative number.

### 5. Testing Requirements

* Test a known shortest path and its cost.
* Test unreachable goals and identical start/goal nodes.
* Test parallel-edge reduction using minimum travel time.
* Test edge updates and their effect on subsequent planning.
* Verify that every returned path uses valid edges.
* Verify that the reported cost equals the sum of the path's edge costs.

### 6. Collaboration Rules

* Shubh works on `shubh/*` branches.
* Tilak works on `tilak/*` branches.
* Never commit directly to `main`.
* Every change must go through a pull request.
* Shubh and Tilak review each other's pull requests.
* Write tests first, then implementation.
* Explain proposed changes before applying them.
