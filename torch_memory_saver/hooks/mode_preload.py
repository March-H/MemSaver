import logging
import os
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
from torch_memory_saver.hooks.base import HookUtilBase
from torch_memory_saver.utils import get_binary_path_from_package, change_env

logger = logging.getLogger(__name__)


class HookUtilModePreload(HookUtilBase):
    def get_path_binary(self):
        env_ld_preload = os.environ.get("LD_PRELOAD", "")
        
        interest_paths = [p for p in env_ld_preload.split(":") if "torch_memory_saver" in p]
        assert len(interest_paths) == 1, (
            f"TorchMemorySaver observes invalid LD_PRELOAD. "
            f"You can use configure_subprocess() utility, "
            f"or directly specify `LD_PRELOAD=/path/to/torch_memory_saver_cpp.some-postfix.so python your_script.py. "
            f'(LD_PRELOAD="{env_ld_preload}" process_id={os.getpid()})'
        )
        return interest_paths[0]


def _find_cuda_runtime_lib_dir() -> str:
    # Prefer the CUDA runtime bundled with the installed Python packages (e.g. torch/nvidia wheels),
    # to avoid loading an older system libcudart before importing torch.
    spec = find_spec("nvidia.cuda_runtime")
    if spec is None or not spec.submodule_search_locations:
        return ""
    package_dir = Path(next(iter(spec.submodule_search_locations)))
    lib_dir = package_dir / "lib"
    if lib_dir.is_dir():
        return str(lib_dir)
    return ""


@contextmanager
def configure_subprocess():
    """Configure environment variables for subprocesses. Only needed for hook_mode=preload."""
    lib_path = str(get_binary_path_from_package("torch_memory_saver_hook_mode_preload"))

    current_preload = os.environ.get("LD_PRELOAD", "")
    new_preload = f"{lib_path}:{current_preload}" if current_preload else lib_path

    runtime_lib_dir = _find_cuda_runtime_lib_dir()
    current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_ld_library_path = current_ld_library_path
    if runtime_lib_dir:
        new_ld_library_path = (
            f"{runtime_lib_dir}:{current_ld_library_path}"
            if current_ld_library_path else runtime_lib_dir
        )

    with change_env("LD_PRELOAD", new_preload):
        with change_env("LD_LIBRARY_PATH", new_ld_library_path):
            yield
