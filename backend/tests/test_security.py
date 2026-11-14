"""
Tests for backend.security: SAST scanner, JWT helpers and air-gap config.

Pure-Python, no network or models. Avoids pytest.raises / parametrize so the
dependency-free runner (run_tests.py) can execute it too.
"""
import base64
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend import security  # noqa: E402
from backend.security import (  # noqa: E402
    scan_code_sast, format_sast_report, create_jwt_token, verify_jwt_token,
    get_user_id_from_token, AirGapConfig,
)


class TestSastPythonBlocked:
    BAD_SNIPPETS = {
        "shell=True": "import subprocess\nsubprocess.run('ls', shell=True)",
        "os.system": "import os\nos.system('rm -rf /tmp/x')",
        "eval(input())": "x = eval(input())",
        "exec(input())": "exec(input())",
        "pickle": "import pickle\npickle.loads(data)",
        "rmtree root": "import shutil\nshutil.rmtree('/etc')",
        "os.remove root": "import os\nos.remove('/etc/passwd')",
        "requests external": "import requests\nrequests.get('https://evil.example')",
        "urlopen": "import urllib.request\nurllib.request.urlopen(u)",
    }

    def test_each_dangerous_snippet_is_blocked(self):
        for name, code in self.BAD_SNIPPETS.items():
            res = scan_code_sast(code, "python")
            assert res["safe"] is False, f"{name} was not blocked"
            assert res["blocked"], name
            assert res["score"] > 0, name

    def test_language_alias_py_is_scanned(self):
        assert scan_code_sast("os.system('x')", "py")["safe"] is False

    def test_multiple_occurrences_raise_score(self):
        one = scan_code_sast("os.system('a')")["score"]
        two = scan_code_sast("os.system('a')\nos.system('b')")["score"]
        assert two > one

    def test_score_is_capped_at_100(self):
        code = "\n".join("os.system('x')" for _ in range(20))
        assert scan_code_sast(code)["score"] == 100


class TestSastPythonSafe:
    def test_benign_code_passes(self):
        code = "def add(a, b):\n    return a + b\nprint(add(1, 2))"
        res = scan_code_sast(code)
        assert res == {"safe": True, "blocked": [], "warnings": [], "score": 0}

    def test_empty_and_whitespace_code_is_safe(self):
        for code in ("", "   \n\t", None):
            assert scan_code_sast(code)["safe"] is True

    def test_warn_patterns_warn_but_do_not_block(self):
        res = scan_code_sast("m = __import__('math')")
        assert res["safe"] is True
        assert res["warnings"]
        assert 0 < res["score"] < 100

    def test_unknown_language_is_not_blocked(self):
        assert scan_code_sast("os.system('x')", "rust")["safe"] is True


class TestSastOtherLanguages:
    def test_c_dangerous_calls_blocked(self):
        for code in ('system("ls");', 'popen("ls","r");', "execve(p, a, e);", "fork();"):
            assert scan_code_sast(code, "c")["safe"] is False, code

    def test_cpp_alias_blocked(self):
        assert scan_code_sast('system("ls");', "cpp")["safe"] is False

    def test_verilog_system_task_blocked(self):
        assert scan_code_sast('initial $system("ls");', "verilog")["safe"] is False

    def test_clean_c_passes(self):
        assert scan_code_sast("int main(){return 0;}", "c")["safe"] is True


class TestSastKnownGaps:
    """Document (not endorse) what regex-based SAST misses, so a future
    hardening change shows up as a deliberate test update."""

    def test_aliased_import_evades_scanner(self):
        code = "import os as o\no.system('id')"
        assert scan_code_sast(code)["safe"] is True

    def test_getattr_evades_scanner(self):
        code = "import os\ngetattr(os, 'sys' + 'tem')('id')"
        assert scan_code_sast(code)["safe"] is True


class TestFormatReport:
    def test_clean_report(self):
        assert "passed" in format_sast_report(scan_code_sast("print(1)"))

    def test_blocked_report_mentions_issue_and_score(self):
        report = format_sast_report(scan_code_sast("os.system('x')"))
        assert "Blocked" in report
        assert "os.system" in report
        assert "Risk Score" in report

    def test_warning_only_report_allows_execution(self):
        report = format_sast_report(scan_code_sast("__import__('x')"))
        assert "Warnings" in report
        assert "Blocked" not in report


def _forge(payload, secret):
    import hashlib
    import hmac
    b64 = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=").decode()
    h = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    p = b64(json.dumps(payload).encode())
    sig = hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{b64(sig)}"


class TestJwt:
    def test_roundtrip(self):
        tok = create_jwt_token("alice", {"role": "admin"})
        payload = verify_jwt_token(tok)
        assert payload["sub"] == "alice"
        assert payload["role"] == "admin"
        assert payload["exp"] > time.time()

    def test_get_user_id(self):
        assert get_user_id_from_token(create_jwt_token("bob")) == "bob"

    def test_tampered_payload_rejected(self):
        h, p, s = create_jwt_token("alice").split(".")
        evil = base64.urlsafe_b64encode(
            json.dumps({"sub": "root", "exp": int(time.time()) + 999}).encode()
        ).rstrip(b"=").decode()
        assert verify_jwt_token(f"{h}.{evil}.{s}") is None

    def test_wrong_secret_rejected(self):
        tok = _forge({"sub": "x", "exp": int(time.time()) + 100}, "not-the-secret")
        assert verify_jwt_token(tok) is None

    def test_expired_token_rejected(self):
        tok = _forge({"sub": "x", "exp": int(time.time()) - 10}, security.JWT_SECRET)
        assert verify_jwt_token(tok) is None

    def test_valid_forged_with_real_secret_accepted(self):
        # Sanity check that _forge itself is correct, so the negatives above mean something.
        tok = _forge({"sub": "x", "exp": int(time.time()) + 100}, security.JWT_SECRET)
        assert verify_jwt_token(tok)["sub"] == "x"

    def test_alg_none_rejected(self):
        b64 = lambda b: base64.urlsafe_b64encode(b).rstrip(b"=").decode()
        h = b64(json.dumps({"alg": "none", "typ": "JWT"}).encode())
        p = b64(json.dumps({"sub": "root", "exp": int(time.time()) + 100}).encode())
        assert verify_jwt_token(f"{h}.{p}.") is None

    def test_garbage_tokens_rejected_without_raising(self):
        for tok in ("", "abc", "a.b", "a.b.c.d", "....", None, 123):
            assert verify_jwt_token(tok) is None

    def test_anonymous_fallback(self):
        assert get_user_id_from_token("garbage") == "anonymous"


class TestAirGap:
    def _make(self, value):
        old = os.environ.get("AIOS_AIR_GAP")
        os.environ["AIOS_AIR_GAP"] = value
        try:
            return AirGapConfig()
        finally:
            if old is None:
                os.environ.pop("AIOS_AIR_GAP", None)
            else:
                os.environ["AIOS_AIR_GAP"] = old

    def test_disabled_allows_everything(self):
        cfg = self._make("0")
        assert cfg.allow_web_search and cfg.allow_model_download and cfg.allow_external_api

    def test_enabled_blocks_everything(self):
        cfg = self._make("1")
        assert not (cfg.allow_web_search or cfg.allow_model_download or cfg.allow_external_api)
        assert cfg.to_dict()["air_gap_enabled"] is True
