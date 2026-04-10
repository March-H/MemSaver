import logging
import os
import shutil
from pathlib import Path
import setuptools
from setuptools import setup

logger = logging.getLogger(__name__)


# copy & modify from torch/utils/cpp_extension.py
def _find_cuda_home():
    """Find the install path for CUDA."""
    home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if home is None:
        compiler_path = shutil.which("nvcc")
        if compiler_path is not None:
            home = os.path.dirname(os.path.dirname(compiler_path))
        else:
            home = '/usr/local/cuda'
    return home


def _create_ext_modules():
    """Create CUDA extension modules."""
    sources = [
        'csrc/api_forwarder.cpp',
        'csrc/core.cpp',
        'csrc/entrypoint.cpp',
    ]

    common_macros = [('Py_LIMITED_API', '0x03090000')]
    extra_compile_args = ['-std=c++17', '-O3']
    cuda_home = Path(_find_cuda_home())
    include_dirs = [str((cuda_home / 'include').resolve())]
    library_dirs = [
        str((cuda_home / 'lib64').resolve()),
        str((cuda_home / 'lib64/stubs').resolve()),
    ]
    libraries = ['cuda', 'cudart']
    platform_macros = [('USE_CUDA', '1')]

    ext_modules = [
        setuptools.Extension(
            name,
            sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            define_macros=[
                *common_macros,
                *platform_macros,
                *extra_macros,
            ],
            py_limited_api=True,
            extra_compile_args=extra_compile_args,
        )
        for name, extra_macros in [
            ('torch_memory_saver_hook_mode_preload', [('TMS_HOOK_MODE_PRELOAD', '1')]),
            ('torch_memory_saver_hook_mode_torch', [('TMS_HOOK_MODE_TORCH', '1')]),
        ]
    ]
    
    return ext_modules


ext_modules = _create_ext_modules()

setup(
    name='torch_memory_saver',
    version='0.0.9',
    ext_modules=ext_modules,
    python_requires=">=3.9",
    packages=setuptools.find_packages(include=["torch_memory_saver", "torch_memory_saver.*"]),
)
