"""Not to be used by end users, but only for tests of the package itself."""

import torch


def get_and_print_gpu_memory(message, gpu_id=0):
    """Print GPU memory usage with optional message."""
    mem = torch.cuda.device_memory_used(gpu_id)
    print(f"GPU {gpu_id} memory: {mem / 1024 ** 3:.2f} GB ({message})")
    return mem
