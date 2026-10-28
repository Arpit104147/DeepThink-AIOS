"""
Regression tests for the path-traversal / SSRF protections advertised in the
README: WebSearch.scrape_url, GitAgent.commit_files and Sandbox.execute_workspace.
No real network is used.
"""
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend.git_agent import GitAgent  # noqa: E402
from backend.sandbox import Sandbox  # noqa: E402
from backend.search import WebSearch  # noqa: E402


class TestScrapeUrlSsrf:
    def _scrape(self, url, resolved_ip):
        ws = WebSearch()
        ws._session = mock.Mock()
        with mock.patch.object(socket, "gethostbyname", return_value=resolved_ip):
            out = ws.scrape_url(url)
        return ws, out

    def test_private_loopback_linklocal_are_blocked_before_any_request(self):
        for ip in ("127.0.0.1", "10.0.0.5", "192.168.1.1", "172.16.0.9", "169.254.169.254"):
            ws, out = self._scrape("http://internal.example/", ip)
            assert out == "", ip
            assert not ws._session.get.called, f"request made to {ip}"

    def test_public_ip_is_fetched(self):
        ws = WebSearch()
        resp = mock.Mock(status_code=200, text="<html><body><p>hello world</p></body></html>",
                         content=b"<html><body><p>hello world</p></body></html>")
        ws._session = mock.Mock()
        ws._session.get.return_value = resp
        with mock.patch.object(socket, "gethostbyname", return_value="93.184.216.34"):
            assert "hello world" in ws.scrape_url("https://example.com/")

    def test_blocked_page_markers_return_empty(self):
        ws = WebSearch()
        body = b"<html>Checking your browser before accessing</html>"
        ws._session = mock.Mock()
        ws._session.get.return_value = mock.Mock(status_code=200, text=body.decode(), content=body)
        with mock.patch.object(socket, "gethostbyname", return_value="93.184.216.34"):
            assert ws.scrape_url("https://example.com/") == ""

    def test_non_200_returns_empty(self):
        ws = WebSearch()
        ws._session = mock.Mock()
        ws._session.get.return_value = mock.Mock(status_code=500, text="x", content=b"x")
        with mock.patch.object(socket, "gethostbyname", return_value="93.184.216.34"):
            assert ws.scrape_url("https://example.com/") == ""

    def test_network_error_returns_empty(self):
        ws = WebSearch()
        ws._session = mock.Mock()
        ws._session.get.side_effect = RuntimeError("boom")
        with mock.patch.object(socket, "gethostbyname", return_value="93.184.216.34"):
            assert ws.scrape_url("https://example.com/") == ""


def _git_repo():
    d = tempfile.mkdtemp(prefix="gitagent_test_")
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    subprocess.run(["git", "init", "-q", d], check=True, env=env)
    subprocess.run(["git", "-C", d, "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", d, "config", "user.name", "t"], check=True)
    return d


class TestGitAgentCommitTraversal:
    def setup_method(self):
        self.base = tempfile.mkdtemp(prefix="gitagent_base_")
        self.agent = GitAgent(workspace_base=self.base)
        self.repo = _git_repo()

    def teardown_method(self):
        shutil.rmtree(self.base, ignore_errors=True)
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_dotdot_path_is_blocked_and_nothing_written(self):
        outside = os.path.join(os.path.dirname(self.repo), "pwned_by_test.txt")
        res = self.agent.commit_files(self.repo, {"../pwned_by_test.txt": "x"}, "evil")
        assert res["success"] is False
        assert "traversal" in res["message"].lower()
        assert not os.path.exists(outside)

    def test_absolute_path_is_blocked(self):
        target = os.path.join(tempfile.gettempdir(), "abs_pwned_by_test.txt")
        res = self.agent.commit_files(self.repo, {target: "x"}, "evil")
        assert res["success"] is False
        assert not os.path.exists(target)

    def test_symlink_escape_is_blocked(self):
        outside = tempfile.mkdtemp(prefix="outside_")
        try:
            os.symlink(outside, os.path.join(self.repo, "link"))
            res = self.agent.commit_files(self.repo, {"link/f.txt": "x"}, "evil")
            assert res["success"] is False
            assert not os.path.exists(os.path.join(outside, "f.txt"))
        finally:
            shutil.rmtree(outside, ignore_errors=True)

    def test_legit_nested_file_commits(self):
        res = self.agent.commit_files(self.repo, {"src/a.txt": "hello"}, "add a")
        assert res["success"] is True, res
        assert res.get("sha")
        with open(os.path.join(self.repo, "src", "a.txt")) as f:
            assert f.read() == "hello"


class TestExecuteWorkspaceTraversal:
    def setup_method(self):
        self.sb = Sandbox(timeout=10)

    def teardown_method(self):
        self.sb.clean_all_workspaces()

    def _run(self, files):
        ok, logs, tmp = self.sb.execute_workspace(files, run_cmd="python main.py", timeout=15)
        return ok, logs, tmp

    def test_dotdot_and_absolute_paths_are_refused(self):
        marker = os.path.join(tempfile.gettempdir(), "ws_escape_by_test.txt")
        _, logs, tmp = self._run({
            "main.py": "print('ok')",
            "../ws_escape_by_test.txt": "x",
            marker: "x",
        })
        assert "Refused traversal" in logs
        assert "Refused unsafe path" in logs
        assert not os.path.exists(marker)
        assert not os.path.exists(os.path.join(os.path.dirname(tmp), "ws_escape_by_test.txt"))

    def test_null_byte_path_refused(self):
        _, logs, _ = self._run({"main.py": "print(1)", "a\x00b.py": "x"})
        assert "Refused unsafe path" in logs

    def test_oversize_file_refused(self):
        big = "x" * (self.sb._MAX_FILE_BYTES + 1)
        _, logs, _ = self._run({"main.py": "print(1)", "big.txt": big})
        assert "oversize" in logs

    def test_run_cmd_outside_allowlist_does_not_execute(self):
        marker = os.path.join(tempfile.gettempdir(), "ws_cmd_by_test.txt")
        ok, logs, _ = self.sb.execute_workspace(
            {"main.py": "print(1)"}, run_cmd=f"touch {marker}", timeout=10)
        assert not os.path.exists(marker)
        assert ok is False
