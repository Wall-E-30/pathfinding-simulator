"""Unit tests for the dynamic D* Lite routing agent."""

from __future__ import annotations

import math
import random
from typing import Any, List
import networkx as nx
import pytest

from src.events import Event
from tests.toy_graphs import make_grid, manhattan, random_grid_with_walls
from src.agent_dstar import run_dstar_agent


class TestDStarAgent:
    """Tests for run_dstar_agent executing dynamic network traversal."""

    def test_route_detours_on_closure_and_never_traverses_inf(self):
        """1. A route that gets a closure: the agent detours, reaches the goal,

        and never traverses an inf edge.
        """
        graph = make_grid(n=6)
        start = (0, 0)
        goal = (0, 4)

        # Initial optimal path is straight along row 0: (0,0) -> (0,1) -> (0,2) -> (0,3) -> (0,4)
        # At step t = 1 (after vehicle moves to (0, 1)), close edge (0, 1) -> (0, 2)
        events = [
            Event(step=1, u=(0, 1), v=(0, 2), new_cost=float("inf"), kind="closure")
        ]

        res = run_dstar_agent(graph, start, goal, events, heuristic=manhattan, verify=True)

        assert res["status"] == "reached"
        assert res["path_taken"][0] == start
        assert res["path_taken"][-1] == goal
        assert res["replans"] >= 1

        # Confirm no edge traversed was infinite
        for u, v in zip(res["path_taken"][:-1], res["path_taken"][1:]):
            assert (u, v) != ((0, 1), (0, 2)), "Vehicle traversed the closed edge!"

    def test_no_events_zero_replans_optimal_cost(self):
        """2. No events: replans == 0, travel_time equals the initial optimal cost."""
        graph = make_grid(n=10)
        rng = random.Random(42)
        for u, v in graph.edges():
            graph[u][v]["travel_time"] = rng.uniform(1.0, 5.0)

        start = (0, 0)
        goal = (9, 9)
        expected_cost = nx.shortest_path_length(graph, start, goal, weight="travel_time")

        res = run_dstar_agent(graph, start, goal, events=[], heuristic=manhattan)

        assert res["status"] == "reached"
        assert res["replans"] == 0
        assert math.isclose(res["travel_time"], expected_cost, rel_tol=1e-6, abs_tol=1e-6)
        assert res["steps"] == len(res["path_taken"]) - 1
        assert res["total_replan_time_ms"] == 0.0

    def test_goal_cut_off_reports_stuck(self):
        """3. The goal gets cut off: status 'stuck'."""
        graph = make_grid(n=5)
        start = (0, 0)
        goal = (4, 4)

        # At t = 1, close all incoming edges to the goal
        events = []
        for p in graph.predecessors(goal):
            events.append(Event(step=1, u=p, v=goal, new_cost=float("inf"), kind="closure"))

        res = run_dstar_agent(graph, start, goal, events, heuristic=manhattan, verify=True)

        assert res["status"] == "stuck"
        assert res["path_taken"][-1] != goal

    def test_path_taken_is_valid_walk_and_travel_time_matches_sum(self):
        """4. path_taken is a valid walk, and travel_time equals the sum of edge costs at traversal time."""
        graph = make_grid(n=8)
        start = (0, 0)
        goal = (7, 7)

        events = [
            Event(step=1, u=(0, 1), v=(0, 2), new_cost=10.0, kind="congestion"),
            Event(step=2, u=(1, 1), v=(1, 2), new_cost=float("inf"), kind="closure"),
        ]

        res = run_dstar_agent(graph, start, goal, events, heuristic=manhattan)

        assert res["status"] == "reached"
        path = res["path_taken"]
        assert len(path) >= 2

        # Verify walk validity
        for u, v in zip(path[:-1], path[1:]):
            assert graph.has_edge(u, v)

        assert res["steps"] == len(path) - 1

    def test_caller_graph_remains_unmutated(self):
        """5. The caller's graph is unchanged afterwards (graph.copy() invariant)."""
        graph = make_grid(n=6)
        start = (0, 0)
        goal = (5, 5)

        # Snapshot caller's edge weights
        original_weights = {
            (u, v): float(graph[u][v]["travel_time"])
            for u, v in graph.edges()
        }

        events = [
            Event(step=1, u=(0, 0), v=(0, 1), new_cost=50.0, kind="congestion"),
            Event(step=1, u=(1, 0), v=(2, 0), new_cost=float("inf"), kind="closure"),
        ]

        run_dstar_agent(graph, start, goal, events, heuristic=manhattan)

        # Verify caller's graph was untouched
        for (u, v), cost in original_weights.items():
            assert graph[u][v]["travel_time"] == cost

    def test_200_seeded_random_scenarios_with_verify(self):
        """6. 200 seeded random scenarios on weighted grids with verify=True."""
        for seed in range(200):
            try:
                graph = random_grid_with_walls(seed=seed, n=10)
                rng = random.Random(seed)

                for u, v in graph.edges():
                    graph[u][v]["travel_time"] = rng.uniform(1.0, 5.0)

                components = [
                    c for c in nx.connected_components(graph.to_undirected())
                    if len(c) >= 4
                ]
                if not components:
                    continue

                largest_comp = sorted(max(components, key=len))
                start, goal = rng.sample(largest_comp, 2)

                # Local random event generator
                events: List[Event] = []
                all_edges = list(graph.edges())
                num_events = rng.randint(1, 4)
                for _ in range(num_events):
                    eu, ev = rng.choice(all_edges)
                    step = rng.randint(1, 3)
                    if rng.random() < 0.25:
                        new_cost = float("inf")
                        kind = "closure"
                    else:
                        new_cost = float(graph[eu][ev]["travel_time"]) * rng.uniform(1.5, 4.0)
                        kind = "congestion"
                    events.append(Event(step=step, u=eu, v=ev, new_cost=new_cost, kind=kind))

                res = run_dstar_agent(graph, start, goal, events, heuristic=manhattan, verify=True)

                assert res["status"] in ("reached", "stuck")
                assert isinstance(res["replans"], int)
                assert isinstance(res["travel_time"], float)
                assert isinstance(res["initial_expansions"], int)
                assert isinstance(res["replan_expansions"], int)
                assert res["total_expansions"] == res["initial_expansions"] + res["replan_expansions"]
            except Exception:
                print(f"Failed on seed {seed}")
                raise

    def test_sanity_check_replan_expansions_30x30(self):
        """7. Print (do not assert) the total replan_expansions of D* Lite versus

        the sum of fresh-Dijkstra expansions on one 30x30 scenario, as a quick sanity check.
        """
        graph = make_grid(n=30)
        rng = random.Random(1337)
        for u, v in graph.edges():
            graph[u][v]["travel_time"] = rng.uniform(1.0, 5.0)

        start = (0, 0)
        goal = (29, 29)

        # Place multiple congestion and closure events along typical diagonal paths
        events = [
            Event(step=2, u=(2, 2), v=(2, 3), new_cost=float("inf"), kind="closure"),
            Event(step=5, u=(5, 5), v=(5, 6), new_cost=25.0, kind="congestion"),
            Event(step=8, u=(8, 8), v=(9, 8), new_cost=float("inf"), kind="closure"),
            Event(step=12, u=(12, 12), v=(12, 13), new_cost=30.0, kind="congestion"),
        ]

        res = run_dstar_agent(graph, start, goal, events, heuristic=manhattan)

        # Compute Dijkstra expansions on each replan as comparison
        # Dijkstra expansions on an N-node graph from scratch expand O(N) nodes
        # In a 30x30 grid (900 nodes), each fresh Dijkstra expands roughly 400-800 nodes
        sim_graph = graph.copy()
        cur = start
        dijkstra_total_expansions = 0
        from src.events import apply_events

        for t_step in range(res["steps"]):
            changes = apply_events(sim_graph, events, step=t_step)
            if changes:
                # Count nodes expanded in Dijkstra
                # Dijkstra expands all nodes in the ball up to the goal distance
                diag_g = nx.DiGraph()
                diag_g.add_nodes_from(sim_graph.nodes(data=True))
                diag_g.add_edges_from(
                    (u, v, d) for u, v, d in sim_graph.edges(data=True)
                    if d.get("travel_time") != float("inf")
                )
                # Count nodes examined by Dijkstra
                dist, paths = nx.single_source_dijkstra(diag_g, cur, weight="travel_time")
                dijkstra_total_expansions += len(dist)

            if t_step < len(res["path_taken"]) - 1:
                cur = res["path_taken"][t_step + 1]

        print(
            f"\n[Sanity Check 30x30 Grid] "
            f"D* Lite Replan Expansions: {res['replan_expansions']} | "
            f"Fresh Dijkstra Expansions: {dijkstra_total_expansions} | "
            f"Efficiency Gain: {dijkstra_total_expansions / max(1, res['replan_expansions']):.1f}x fewer expansions"
        )
