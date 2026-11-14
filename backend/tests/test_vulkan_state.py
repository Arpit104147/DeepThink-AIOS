"""
Regression test for the missing `import json` in backend.vulkan_engine.

Before the fix, load_installed_state()/save_installed_state() referenced an
undefined `json`; the NameError was swallowed by a broad `except`, so install
state silently never persisted (load always returned defaults, save was a
no-op). These tests assert the round-trip actually works.
"""
import importlib
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import backend.vulkan_engine as ve  # noqa: E402


class TestInstalledStatePersistence:
    def setup_method(self):
        self._dir = tempfile.mkdtemp(prefix="vulkan_state_test_")
        self._orig_dir = ve.VULKAN_DIR
        self._orig_file = ve.INSTALLED_STATE_FILE
        ve.VULKAN_DIR = self._dir
        ve.INSTALLED_STATE_FILE = os.path.join(self._dir, "installed_state.json")

    def teardown_method(self):
        ve.VULKAN_DIR = self._orig_dir
        ve.INSTALLED_STATE_FILE = self._orig_file
        import shutil
        shutil.rmtree(self._dir, ignore_errors=True)

    def test_json_is_importable_in_module(self):
        assert hasattr(ve, "json") and ve.json is importlib.import_module("json")

    def test_save_actually_writes_a_file(self):
        ve.save_installed_state({"cuda": True, "vulkan": False, "metal": True})
        assert os.path.exists(ve.INSTALLED_STATE_FILE), "state file was not written"

    def test_round_trip_preserves_values(self):
        ve.save_installed_state({"cuda": True, "vulkan": False, "metal": True})
        loaded = ve.load_installed_state()
        assert loaded == {"cuda": True, "vulkan": False, "metal": True}

    def test_load_returns_defaults_when_absent(self):
        loaded = ve.load_installed_state()
        assert loaded == {"cuda": False, "vulkan": False, "metal": False}
