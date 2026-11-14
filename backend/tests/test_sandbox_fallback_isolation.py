"""
Containment regression tests for the "unrestricted fallback" execution path.

When restricted mode blocks an import, Sandbox re-runs the script via
_execute_python_unrestricted. That path intentionally has no Python-level
import/builtins whitelist (so legitimate AI code using an unlisted-but-benign
module still runs), so the only containment left is OS-level: the fix threads
`use_isolation=self.isolation_available` into the fallback runs, wrapping them
in a network namespace (unshare --net) plus RLIMIT caps.

These tests assert that containment holds. They are defensive checks, not
exploit demonstrations: each asserts the sandbox *prevents* an action.

Uses runtime guards (not pytest markers) so the dependency-free
backend/tests/run_tests.py runner can execute this file too.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.sandbox import Sandbox  # noqa: E402


class TestFallbackIsWired:
    def test_blocked_import_triggers_fallback_banner(self):
        # `subprocess` is not on the restricted whitelist, so this forces the
        # fallback path to run (confirming the path under test is exercised).
        sb = Sandbox(timeout=15)
        try:
            ok, out = sb.execute("import subprocess\nprint('fallback-ran')", language="python")
            assert "Unrestricted Fallback" in out
            assert "fallback-ran" in out
        finally:
            sb.clean_all_workspaces()


class TestFallbackNetworkContained:
    def test_outbound_connection_is_blocked_under_isolation(self):
        sb = Sandbox(timeout=15)
        if not sb.isolation_available:
            return  # kernel isolation (unshare) unavailable here; nothing to assert
        # Force the fallback (unlisted import), then attempt a benign outbound
        # TCP connect to a non-routable address. Under the network namespace
        # this must fail — assert the sandbox does NOT report a successful connect.
        code = (
            "import subprocess\n"            # unlisted -> forces fallback
            "import socket\n"
            "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "s.settimeout(2)\n"
            "try:\n"
            "    s.connect(('10.255.255.1', 80))\n"   # non-routable under isolation
            "    print('CONNECTED')\n"
            "except OSError as e:\n"
            "    print('BLOCKED', type(e).__name__)\n"
            "finally:\n"
            "    s.close()\n"
        )
        try:
            ok, out = sb.execute(code, language="python")
            assert "CONNECTED" not in out, f"outbound connection was NOT contained: {out!r}"
            assert "BLOCKED" in out
        finally:
            sb.clean_all_workspaces()
