"""
Tests for benchmark state handling in backend.benchmarks.runner.
Regression: stop_benchmark() used to call add_log() while holding the
non-reentrant STATE_LOCK, deadlocking /api/cancel forever.
"""
import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.benchmarks import runner  # noqa: E402


def _run_with_deadline(fn, seconds=5):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("r", fn()), daemon=True)
    t.start()
    t.join(seconds)
    return (not t.is_alive()), out.get("r")


class TestStopBenchmark:
    def test_does_not_deadlock(self):
        finished, res = _run_with_deadline(runner.stop_benchmark)
        assert finished, "stop_benchmark deadlocked on STATE_LOCK"
        assert res["status"] == "stopped"

    def test_marks_state_cancelled_and_workers_idle(self):
        with runner.STATE_LOCK:
            runner.BENCHMARK_STATE["active"] = True
            runner.BENCHMARK_STATE["workers"] = [{"status": "Busy", "progress": 50, "task": "t"}]
        finished, _ = _run_with_deadline(runner.stop_benchmark)
        assert finished
        snap = runner.get_benchmark_status()
        assert snap["active"] is False
        assert snap["cancel_requested"] is True
        assert snap["workers"][0] == {"status": "Idle", "progress": 0, "task": "N/A"}
        assert any("cancelled" in line for line in snap["logs"])

    def test_sets_orchestrator_cancel_event(self):
        class Orch:
            cancel_event = threading.Event()
        runner.stop_benchmark(Orch)
        assert Orch.cancel_event.is_set()

    def test_lock_is_released_afterwards(self):
        runner.stop_benchmark()
        assert runner.STATE_LOCK.acquire(timeout=2)
        runner.STATE_LOCK.release()


class TestAddLog:
    def test_log_is_capped_at_200(self):
        for i in range(250):
            runner.add_log(f"line {i}")
        logs = runner.get_benchmark_status()["logs"]
        assert len(logs) == 200
        assert logs[-1].endswith("line 249")
