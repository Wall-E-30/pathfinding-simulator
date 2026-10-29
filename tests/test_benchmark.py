"""Unit tests for benchmark helper logic (CSV row generation and summary calculations)."""

from __future__ import annotations

import math
import pytest

from scripts.benchmark import build_csv_row, compute_summary


class TestBenchmarkHelpers:
    """Tests for pure benchmark helper functions."""

    def test_build_csv_row_with_replans(self):
        """build_csv_row should extract all required columns and calculate mean latency."""
        sample_result = {
            "status": "reached",
            "path_taken": [10, 20, 30],
            "travel_time": 150.25,
            "steps": 15,
            "replans": 3,
            "initial_expansions": 200,
            "replan_expansions": 60,
            "total_expansions": 260,
            "replan_latencies_ms": [10.0, 15.0, 20.0],
            "total_replan_time_ms": 45.0,
        }

        row = build_csv_row(seed=7, algorithm="baseline", result=sample_result)

        assert row["seed"] == 7
        assert row["algorithm"] == "baseline"
        assert row["status"] == "reached"
        assert row["steps"] == 15
        assert row["replans"] == 3
        assert row["initial_expansions"] == 200
        assert row["replan_expansions"] == 60
        assert row["total_expansions"] == 260
        assert math.isclose(row["travel_time"], 150.25)
        assert math.isclose(row["total_replan_time_ms"], 45.0)
        assert math.isclose(row["mean_replan_latency_ms"], 15.0)

    def test_build_csv_row_zero_replans(self):
        """When replans == 0, mean_replan_latency_ms must be 0.0."""
        sample_result = {
            "status": "reached",
            "path_taken": [10, 20],
            "travel_time": 50.0,
            "steps": 5,
            "replans": 0,
            "initial_expansions": 100,
            "replan_expansions": 0,
            "total_expansions": 100,
            "replan_latencies_ms": [],
            "total_replan_time_ms": 0.0,
        }

        row = build_csv_row(seed=1, algorithm="dstar", result=sample_result)

        assert row["replans"] == 0
        assert row["replan_expansions"] == 0
        assert row["mean_replan_latency_ms"] == 0.0
        assert row["total_replan_time_ms"] == 0.0

    def test_compute_summary_metrics(self):
        """compute_summary calculates mean, median, speed-up ratios, and discrepancy counts."""
        rows = [
            # Seed 0
            {
                "seed": 0,
                "algorithm": "baseline",
                "status": "reached",
                "replan_expansions": 100,
                "total_replan_time_ms": 20.0,
                "mean_replan_latency_ms": 10.0,
                "travel_time": 300.0,
            },
            {
                "seed": 0,
                "algorithm": "dstar",
                "status": "reached",
                "replan_expansions": 20,
                "total_replan_time_ms": 5.0,
                "mean_replan_latency_ms": 2.5,
                "travel_time": 300.0,
            },
            # Seed 1
            {
                "seed": 1,
                "algorithm": "baseline",
                "status": "reached",
                "replan_expansions": 300,
                "total_replan_time_ms": 40.0,
                "mean_replan_latency_ms": 20.0,
                "travel_time": 450.0,
            },
            {
                "seed": 1,
                "algorithm": "dstar",
                "status": "reached",
                "replan_expansions": 40,
                "total_replan_time_ms": 10.0,
                "mean_replan_latency_ms": 5.0,
                "travel_time": 450.0,
            },
            # Seed 2 (one stuck, travel time diff)
            {
                "seed": 2,
                "algorithm": "baseline",
                "status": "stuck",
                "replan_expansions": 50,
                "total_replan_time_ms": 10.0,
                "mean_replan_latency_ms": 5.0,
                "travel_time": 100.0,
            },
            {
                "seed": 2,
                "algorithm": "dstar",
                "status": "stuck",
                "replan_expansions": 10,
                "total_replan_time_ms": 2.0,
                "mean_replan_latency_ms": 1.0,
                "travel_time": 105.0,  # diff > 1e-6
            },
        ]

        summary = compute_summary(rows)

        # Baseline replan expansions: [100, 300, 50] -> mean = 150.0, median = 100.0
        assert math.isclose(summary["baseline_mean_expansions"], 150.0)
        assert math.isclose(summary["baseline_median_expansions"], 100.0)

        # D* Lite replan expansions: [20, 40, 10] -> mean = 23.333..., median = 20.0
        assert math.isclose(summary["dstar_mean_expansions"], 70.0 / 3.0)
        assert math.isclose(summary["dstar_median_expansions"], 20.0)

        # Expansion speed-up: 150.0 / (70 / 3) = 6.42857...
        assert math.isclose(summary["expansions_speedup"], 150.0 / (70.0 / 3.0))

        # Latency speed-up (total replan time): baseline mean (23.333...) / dstar mean (5.666...)
        assert summary["latency_speedup"] > 1.0

        # Unreached status count (Seed 2 is stuck)
        assert summary["unreached_count"] == 1

        # Travel time difference count (Seed 2 diff = 5.0 > 1e-6)
        assert summary["travel_time_diff_count"] == 1
