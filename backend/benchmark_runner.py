"""
AIOS Benchmark System - Backward Compatibility Proxy Module.
Re-exports modularized benchmark runner components from backend.benchmarks.
"""
from backend.benchmarks import (
    BENCHMARK_STATE,
    STATE_LOCK,
    get_default_worker_count,
    update_state,
    add_log,
    run_benchmark,
    run_benchmark_suite,
    stop_benchmark,
    get_benchmark_status
)
from backend.benchmarks.datasets import MOCK_PROBLEMS, COMPARISON_BASELINES, fetch_real_dataset

__all__ = [
    "BENCHMARK_STATE",
    "STATE_LOCK",
    "get_default_worker_count",
    "update_state",
    "add_log",
    "run_benchmark",
    "run_benchmark_suite",
    "stop_benchmark",
    "get_benchmark_status",
    "MOCK_PROBLEMS",
    "COMPARISON_BASELINES",
    "fetch_real_dataset",
]
