"""D* Lite pathfinding algorithm for directed weighted graphs.

Reference:
    Koenig, S., & Likhachev, M. (2002). D* Lite.
    Eighteenth National Conference on Artificial Intelligence (AAAI-02).
"""

from __future__ import annotations

import heapq
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple
import networkx as nx

# Type alias for the 2-tuple priority key: (k1, k2)
Key = Tuple[float, float]
Node = Any
HeuristicFn = Callable[[Node, Node], float]


class DStarLite:
    """Graph-agnostic D* Lite implementation for directed weighted graphs.

    D* Lite searches backwards from goal to start, allowing it to efficiently
    repair search trees when edge costs change during dynamic traversal.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        start: Node,
        goal: Node,
        heuristic: HeuristicFn | None = None,
    ) -> None:
        self.graph = graph
        self.start = start
        self.goal = goal
        self.heuristic = heuristic

        # g[s]: path cost from vertex s to goal from the previous search
        self.g: Dict[Node, float] = defaultdict(lambda: float("inf"))

        # rhs[s]: one-step lookahead cost based on successors' g-values
        # rhs(s) = min_{s' in Succ(s)} (c(s, s') + g(s'))
        self.rhs: Dict[Node, float] = defaultdict(lambda: float("inf"))

        # Priority queue using min-heap with lazy deletion
        # Heap elements: (key, counter, node)
        self._heap: List[Tuple[Key, int, Node]] = []
        self._open_set: Dict[Node, Key] = {}
        self._counter: int = 0

        # km: Accumulated key modifier accounting for robot movement (0 in initial planning)
        self.km: float = 0.0

        # Performance metric: number of vertex expansions
        self.expansions: int = 0

    # -------------------------------------------------------------------------
    # Priority Queue Helpers (Min-heap with Lazy Deletion)
    # -------------------------------------------------------------------------

    def _insert(self, u: Node, key: Key) -> None:
        """Insert or re-key vertex u in the priority queue with given key."""
        self._open_set[u] = key
        # counter ensures tie-breaking without comparing arbitrary node objects
        heapq.heappush(self._heap, (key, self._counter, u))
        self._counter += 1

    def _remove(self, u: Node) -> None:
        """Lazily remove vertex u from the priority queue."""
        self._open_set.pop(u, None)

    def _top_key(self) -> Key:
        """Return the minimum key in the queue, purging stale lazy entries.

        Returns (inf, inf) if the priority queue is empty.
        """
        while self._heap:
            key, _, u = self._heap[0]
            # Verify if this entry matches the current registered key for u
            if u in self._open_set and self._open_set[u] == key:
                return key
            heapq.heappop(self._heap)  # Discard stale entry
        return (float("inf"), float("inf"))

    def _pop(self) -> Tuple[Node | None, Key]:
        """Pop and return (u, key) with the minimum key in the queue."""
        while self._heap:
            key, _, u = heapq.heappop(self._heap)
            if u in self._open_set and self._open_set[u] == key:
                del self._open_set[u]
                return u, key
        return None, (float("inf"), float("inf"))

    # -------------------------------------------------------------------------
    # Core D* Lite Methods (Koenig & Likhachev 2002)
    # -------------------------------------------------------------------------

    def calculate_key(self, s: Node) -> Key:
        """Calculate the 2-component priority key for vertex s.

        Paper: Line {01} CalculateKey(s)
        Returns:
            [min(g(s), rhs(s)) + h(s_start, s) + km, min(g(s), rhs(s))]
        """
        m = min(self.g[s], self.rhs[s])
        if m == float("inf"):
            # Avoid inf + h arithmetic and prevent NaN comparisons
            return (float("inf"), float("inf"))

        # h(s_start, s): estimated cost from start to s in backward search
        h_val = self.heuristic(self.start, s) if self.heuristic else 0.0
        return (m + h_val + self.km, m)

    def initialize(self) -> None:
        """Initialize search variables and insert the goal vertex into the queue.

        Paper: Lines {02}-{06} Initialize()
        """
        self._heap.clear()
        self._open_set.clear()
        self.km = 0.0
        self.expansions = 0

        # Goal is the destination of the backward search, so rhs(goal) = 0
        self.rhs[self.goal] = 0.0
        self._insert(self.goal, self.calculate_key(self.goal))

    def update_vertex(self, u: Node) -> None:
        """Update the rhs value of vertex u and its membership in the priority queue.

        Paper: Lines {07}-{09} UpdateVertex(u)
        Notice: Because search is backwards from goal to start:
        - rhs(u) evaluates successors s' of u (outgoing edges u -> s')
        - When g(u) changes, updates propagate to predecessors p of u (p -> u)
        """
        if u != self.goal:
            # rhs(u) = min_{s' in Succ(u)} (c(u, s') + g(s'))
            succs = list(self.graph.successors(u))
            if succs:
                # Find minimum one-step lookahead cost to goal through any outgoing edge
                self.rhs[u] = min(
                    self.graph[u][s]["travel_time"] + self.g[s]
                    for s in succs
                )
            else:
                self.rhs[u] = float("inf")

        # Remove u from queue if present (lazy deletion)
        self._remove(u)

        # Inconsistent vertices (g != rhs) must be in the queue to be processed
        if self.g[u] != self.rhs[u]:
            self._insert(u, self.calculate_key(u))

    def compute_shortest_path(self) -> None:
        """Expand inconsistent vertices until start is consistent and top key >= start key.

        Paper: Lines {10}-{21} ComputeShortestPath()
        """
        while True:
            top_key = self._top_key()
            start_key = self.calculate_key(self.start)

            # Termination condition: top key >= start key and start is consistent (g == rhs)
            if not (top_key < start_key or self.rhs[self.start] != self.g[self.start]):
                break

            # If top key is inf, remaining inconsistent vertices cannot reach start
            if top_key == (float("inf"), float("inf")):
                break

            u, k_old = self._pop()
            if u is None:
                break

            k_new = self.calculate_key(u)

            if k_old < k_new:
                # Line {13}-{14}: Outdated key from previous km or updated g/rhs, reinsert
                self._insert(u, k_new)
            elif self.g[u] > self.rhs[u]:
                # Lines {15}-{17}: Overconsistent case (shorter path discovered)
                # Set g(u) = rhs(u) to make u locally consistent
                self.g[u] = self.rhs[u]
                self.expansions += 1
                # Propagate cost decrease to all predecessors that could transition into u
                for p in self.graph.predecessors(u):
                    self.update_vertex(p)
            else:
                # Lines {18}-{21}: Underconsistent case (cost increased/invalidated)
                # Set g(u) = inf to force recomputation of paths through u
                self.g[u] = float("inf")
                self.expansions += 1
                # Propagate cost increase to predecessors and re-evaluate u itself
                for p in list(self.graph.predecessors(u)) + [u]:
                    self.update_vertex(p)

    def extract_path(self) -> Tuple[List[Node], float]:
        """Greedily extract the shortest path from start to goal following optimal successors.

        Follows: argmin_{s' in Succ(curr)} (c(curr, s') + g(s')).
        Returns:
            (path, cost): List of node IDs and total travel time in seconds.
            ([], inf): If goal is unreachable.
        """
        if self.start == self.goal:
            if self.start in self.graph:
                return [self.start], 0.0
            return [], float("inf")

        # If g[start] remains infinite, no path exists from start to goal
        if self.g[self.start] == float("inf"):
            return [], float("inf")

        curr = self.start
        path = [curr]
        total_cost = 0.0
        visited = {curr}
        max_steps = self.graph.number_of_nodes()

        while curr != self.goal:
            succs = list(self.graph.successors(curr))
            if not succs:
                return [], float("inf")

            best_succ = None
            best_val = float("inf")
            best_edge_cost = float("inf")

            for s in succs:
                edge_cost = float(self.graph[curr][s]["travel_time"])
                val = edge_cost + self.g[s]
                if val < best_val:
                    best_val = val
                    best_succ = s
                    best_edge_cost = edge_cost

            # Stop if no valid successor found, infinite path cost, or cycle detected
            if (
                best_succ is None
                or best_val == float("inf")
                or best_succ in visited
                or len(path) > max_steps
            ):
                return [], float("inf")

            curr = best_succ
            visited.add(curr)
            path.append(curr)
            total_cost += best_edge_cost

        return path, total_cost


def plan(
    graph: nx.DiGraph,
    start: Node,
    goal: Node,
    heuristic: HeuristicFn | None = None,
) -> Tuple[List[Node], float, Dict[str, Any]]:
    """Shared algorithm interface contract for D* Lite initial planning.

    Args:
        graph: Directed networkx graph with 'travel_time' edge attribute.
        start: Start node ID.
        goal: Goal node ID.
        heuristic: Optional heuristic callable h(n, goal).

    Returns:
        (path, cost, stats) where:
            path: List of nodes from start to goal inclusive (empty if unreachable).
            cost: Total travel time in seconds (float('inf') if unreachable).
            stats: {"expansions": int, "time_ms": float}.
    """
    t0 = time.perf_counter()

    # Guard for missing nodes in the graph
    if not graph.has_node(start) or not graph.has_node(goal):
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return [], float("inf"), {"expansions": 0, "time_ms": elapsed_ms}

    # Contract requirement: identical start and goal
    if start == goal:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return [start], 0.0, {"expansions": 0, "time_ms": elapsed_ms}

    planner = DStarLite(graph, start, goal, heuristic=heuristic)
    planner.initialize()
    planner.compute_shortest_path()
    path, cost = planner.extract_path()

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    stats = {"expansions": planner.expansions, "time_ms": elapsed_ms}

    return path, cost, stats

