"""
Behavioural tests for Sandbox.execute() — these actually run (small) Python
snippets in the sandbox subprocess, so they cover the real isolation layer,
not just the static helpers. Each snippet is tiny and given a short timeout.
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.sandbox import Sandbox  # noqa: E402


class TestPythonExecution:
    def setup_method(self):
        self.sb = Sandbox(timeout=15)

    def teardown_method(self):
        self.sb.clean_all_workspaces()

    def test_prints_output(self):
        ok, out = self.sb.execute("print(2 + 3)", language="python")
        assert ok, out
        assert "5" in out

    def test_runtime_error_reported_as_failure(self):
        ok, out = self.sb.execute("raise ValueError('boom')", language="python")
        assert not ok
        assert "boom" in out

    def test_syntax_error_reported_as_failure(self):
        ok, out = self.sb.execute("def f(:\n  pass", language="python")
        assert not ok

    def test_allowed_stdlib_import_works(self):
        ok, out = self.sb.execute("import math\nprint(math.sqrt(16))", language="python")
        assert ok, out
        assert "4" in out

    def test_unlisted_module_falls_back_to_unrestricted(self):
        # By design a blocked import re-runs the code unrestricted so legitimate
        # AI-written scripts still work. Pin that so a change is deliberate.
        ok, out = self.sb.execute("import subprocess\nprint('imported')", language="python")
        assert "imported" in out
        assert "Unrestricted Fallback" in out

    def test_open_builtin_is_blocked(self):
        ok, out = self.sb.execute("print(open('/etc/passwd').read())", language="python")
        assert "root:" not in out

    def test_eval_exec_builtins_blocked(self):
        ok, out = self.sb.execute("print(eval('1+1'))", language="python")
        assert not (ok and out.strip() == "2")

    def test_timeout_kills_runaway_code(self):
        start = time.time()
        ok, out = self.sb.execute("x = 0\nwhile x >= 0:\n    x += 1 if x < 10 else -1\n",
                                  language="python", timeout=3)
        assert time.time() - start < 20
        assert not ok


class TestLanguageDetection:
    def test_unsupported_language_reports_failure(self):
        ok, out = Sandbox(timeout=5).execute("print(1)", language="cobol")
        assert not ok
        assert "Unsupported" in out

    def test_bash_runs(self):
        sb = Sandbox(timeout=10)
        ok, out = sb.execute("echo hello-sandbox", language="bash")
        assert ok, out
        assert "hello-sandbox" in out


class TestStripAnsi:
    def test_removes_color_codes(self):
        assert Sandbox._strip_ansi("\x1b[31mred\x1b[0m text") == "red text"

    def test_plain_and_empty_pass_through(self):
        assert Sandbox._strip_ansi("plain") == "plain"
        assert Sandbox._strip_ansi("") == ""
        assert Sandbox._strip_ansi(None) is None
