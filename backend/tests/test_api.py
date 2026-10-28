"""
HTTP-level tests for the FastAPI app (backend.app) using TestClient.

No models are loaded and no network is used. The app creates ./outputs,
./workspaces and a memory DB relative to the CWD at import time, so we import
it from a throwaway directory to keep the repo clean.
"""
import os
import sys
import tempfile

import pytest

try:
    from fastapi.testclient import TestClient
except Exception as e:  # fastapi/httpx not installed
    pytest.skip(f"fastapi TestClient unavailable: {e}", allow_module_level=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_ORIG_CWD = os.getcwd()
_TMP = tempfile.mkdtemp(prefix="aios_api_test_")
os.chdir(_TMP)
try:
    from backend import app as app_module  # noqa: E402
finally:
    os.chdir(_ORIG_CWD)

client = TestClient(app_module.app)


class TestStatusAndSecurity:
    def test_status_shape(self):
        r = client.get("/api/status")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, dict) and body

    def test_security_status(self):
        r = client.get("/api/security/status")
        assert r.status_code == 200
        body = r.json()
        assert body["sast_enabled"] is True
        assert set(body["air_gap"]) >= {"air_gap_enabled", "web_search", "model_download"}

    def test_vulkan_status_does_not_500(self):
        assert client.get("/api/vulkan/status").status_code == 200

    def test_workspace_status(self):
        r = client.get("/api/workspace/status")
        assert r.status_code == 200
        assert "available" in r.json()


class TestAuth:
    def test_login_when_auth_disabled(self):
        if app_module.AUTH_ENABLED:
            return
        r = client.post("/api/auth/login", json={"username": "u", "password": "p"})
        assert r.status_code == 200
        assert r.json()["token"] == "auth_disabled"

    def test_login_missing_fields_is_422(self):
        assert client.post("/api/auth/login", json={"username": "u"}).status_code == 422

    def test_wrong_password_rejected_when_auth_enabled(self):
        from unittest import mock
        with mock.patch.object(app_module, "AUTH_ENABLED", True), \
             mock.patch.dict(os.environ, {"AIOS_ADMIN_PASSWORD": "s3cret"}):
            r = client.post("/api/auth/login", json={"username": "u", "password": "nope"})
            assert r.status_code == 401
            ok = client.post("/api/auth/login", json={"username": "u", "password": "s3cret"})
            assert ok.status_code == 200
            assert app_module.verify_jwt_token(ok.json()["token"])["sub"] == "u"


class TestModelEndpoints:
    def test_download_unknown_model_is_400(self):
        r = client.post("/api/download/definitely_not_a_model")
        assert r.status_code == 400

    def test_remove_unknown_custom_model_is_400(self):
        assert client.delete("/api/models/custom/definitely_not_a_model").status_code == 400

    def test_add_custom_model_requires_fields(self):
        r = client.post("/api/models/custom/add",
                        json={"key": "", "name": "n", "repo_id": "", "filename": ""})
        assert r.status_code == 400

    def test_add_custom_model_validation_error_on_missing_body_fields(self):
        assert client.post("/api/models/custom/add", json={"key": "k"}).status_code == 422

    def test_roles_get(self):
        r = client.get("/api/models/roles")
        assert r.status_code == 200
        assert "roles" in r.json()

    def test_delete_unknown_model_files_is_400(self):
        assert client.delete("/api/models/delete/definitely_not_a_model").status_code == 400


class TestSettings:
    def test_valid_settings_update(self):
        r = client.post("/api/settings", json={
            "context_length": 4096, "max_tokens": 512, "temperature": 0.2,
            "device_mode": "cpu", "gpu_layers": 0, "search_mode": "off"})
        assert r.status_code == 200
        assert r.json() == {"status": "updated"}
        orch = app_module.orchestrator
        assert orch.max_tokens == 512
        assert orch.temperature == 0.2

    def test_invalid_types_rejected(self):
        r = client.post("/api/settings", json={
            "context_length": "lots", "max_tokens": 1, "temperature": 0.1})
        assert r.status_code == 422

    def test_missing_required_fields_rejected(self):
        assert client.post("/api/settings", json={"temperature": 0.1}).status_code == 422


class TestChatValidation:
    def test_missing_prompt_is_422(self):
        assert client.post("/api/chat", json={"mode": "auto"}).status_code == 422

    def test_missing_mode_is_422(self):
        assert client.post("/api/chat", json={"prompt": "hi"}).status_code == 422

    def test_wrong_type_is_422(self):
        r = client.post("/api/chat", json={"prompt": "hi", "mode": "auto", "max_tokens": "many"})
        assert r.status_code == 422


class TestMemoryEndpoints:
    def test_count_is_int(self):
        r = client.get("/api/memory/count")
        assert r.status_code == 200
        assert isinstance(r.json()["count"], int)

    def test_list_returns_memories_key(self):
        r = client.get("/api/memory/list")
        assert r.status_code == 200
        assert "memories" in r.json()

    def test_delete_requires_memory_id(self):
        assert client.post("/api/memory/delete", json={}).status_code == 422


class TestWorkspaceEndpoints:
    def test_commit_requires_body(self):
        assert client.post("/api/workspace/commit", json={}).status_code == 422

    def test_commit_traversal_is_not_written(self):
        # Valid git repo inside tmp; a ../ path must be refused by GitAgent.
        import subprocess
        repo = tempfile.mkdtemp(prefix="api_repo_")
        subprocess.run(["git", "init", "-q", repo], check=True)
        r = client.post("/api/workspace/commit", json={
            "workspace_path": repo, "files": {"../api_escape.txt": "x"}, "message": "m"})
        assert r.status_code == 200
        assert r.json()["success"] is False
        assert not os.path.exists(os.path.join(os.path.dirname(repo), "api_escape.txt"))


class TestCancel:
    def test_cancel_sets_flag(self):
        r = client.post("/api/cancel")
        assert r.status_code == 200
        assert app_module.generation_cancel.is_set()
        app_module.generation_cancel.clear()


class TestCors:
    def test_cors_header_present(self):
        r = client.get("/api/security/status", headers={"Origin": "http://localhost:5173"})
        assert r.headers.get("access-control-allow-origin") == "*"
