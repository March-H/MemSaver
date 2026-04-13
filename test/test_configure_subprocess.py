"""Regression test: configure_subprocess() must preserve existing LD_PRELOAD entries."""
import os
from unittest.mock import patch

import pytest

FAKE_LIB = "/fake/torch_memory_saver_hook_mode_preload.so"


def _invoke(initial_ld_preload):
    """Call configure_subprocess() and return the LD_PRELOAD seen inside the context."""
    from torch_memory_saver.hooks.mode_preload import configure_subprocess

    env_patch = {} if initial_ld_preload is None else {"LD_PRELOAD": initial_ld_preload}
    with patch.dict(os.environ, env_patch, clear=False):
        if initial_ld_preload is None:
            os.environ.pop("LD_PRELOAD", None)
        with patch(
            "torch_memory_saver.hooks.mode_preload.get_binary_path_from_package",
            return_value=FAKE_LIB,
        ):
            with configure_subprocess():
                return os.environ.get("LD_PRELOAD", "")


def test_sets_ld_preload_when_unset():
    """LD_PRELOAD unset → should be set to the lib path only."""
    result = _invoke(None)
    assert result == FAKE_LIB


def test_prepends_without_overwriting_existing():
    """LD_PRELOAD already set → TMS lib is prepended, existing entry preserved."""
    existing = "/other/lib.so"
    result = _invoke(existing)
    parts = result.split(":")
    assert parts[0] == FAKE_LIB, "TMS lib must come first"
    assert existing in parts, "Pre-existing LD_PRELOAD entry must not be dropped"


def test_restores_ld_preload_after_context():
    """After the context exits, LD_PRELOAD must be restored to its original value."""
    from torch_memory_saver.hooks.mode_preload import configure_subprocess

    original = "/original/lib.so"
    with patch.dict(os.environ, {"LD_PRELOAD": original}):
        with patch(
            "torch_memory_saver.hooks.mode_preload.get_binary_path_from_package",
            return_value=FAKE_LIB,
        ):
            with configure_subprocess():
                assert os.environ["LD_PRELOAD"] != original
        assert os.environ.get("LD_PRELOAD") == original


def test_prepends_cuda_runtime_lib_dir_to_ld_library_path():
    """When CUDA runtime lib dir is found, it should be prepended to LD_LIBRARY_PATH."""
    from torch_memory_saver.hooks.mode_preload import configure_subprocess

    existing = "/existing/lib/path"
    runtime_lib = "/fake/nvidia/cuda_runtime/lib"
    with patch.dict(os.environ, {"LD_LIBRARY_PATH": existing}, clear=False):
        with (
            patch(
                "torch_memory_saver.hooks.mode_preload.get_binary_path_from_package",
                return_value=FAKE_LIB,
            ),
            patch(
                "torch_memory_saver.hooks.mode_preload._find_cuda_runtime_lib_dir",
                return_value=runtime_lib,
            ),
        ):
            with configure_subprocess():
                parts = os.environ.get("LD_LIBRARY_PATH", "").split(":")
                assert parts[0] == runtime_lib
                assert existing in parts


def test_keeps_ld_library_path_when_no_cuda_runtime_lib_dir():
    """When no CUDA runtime lib dir is found, LD_LIBRARY_PATH should remain unchanged."""
    from torch_memory_saver.hooks.mode_preload import configure_subprocess

    existing = "/existing/lib/path"
    with patch.dict(os.environ, {"LD_LIBRARY_PATH": existing}, clear=False):
        with (
            patch(
                "torch_memory_saver.hooks.mode_preload.get_binary_path_from_package",
                return_value=FAKE_LIB,
            ),
            patch(
                "torch_memory_saver.hooks.mode_preload._find_cuda_runtime_lib_dir",
                return_value="",
            ),
        ):
            with configure_subprocess():
                assert os.environ.get("LD_LIBRARY_PATH") == existing
