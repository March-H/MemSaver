"""Benchmark model reload speed with torch_memory_saver on a real LLM."""

from __future__ import annotations

import gc
import json
import multiprocessing
import os
import subprocess
import tempfile
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from json import JSONDecodeError
from pathlib import Path
from unittest import SkipTest

import pytest
import torch
import torch_memory_saver

transformers = pytest.importorskip("transformers")
AutoModelForCausalLM = transformers.AutoModelForCausalLM
AutoTokenizer = transformers.AutoTokenizer


RUN_FLAG = "TMS_RUN_MODEL_RELOAD_TEST"
DEFAULT_MODEL_PATH = "/root/share/models/Qwen3-1.7B"
DEFAULT_PROMPT = "Please summarize how torch_memory_saver can pause and resume model weights."
DEFAULT_MAX_INPUT_TOKENS = 64
DEFAULT_HOOK_MODE = "preload"

pytestmark = pytest.mark.skipif(
    os.environ.get(RUN_FLAG) != "1",
    reason=(
        "This is a heavyweight benchmark. "
        f"Set {RUN_FLAG}=1 to run it explicitly."
    ),
)


@dataclass
class BenchmarkConfig:
    model_path: str
    prompt: str
    device: str
    hook_mode: str
    max_input_tokens: int

    @classmethod
    def from_env(cls) -> "BenchmarkConfig":
        return cls(
            model_path=os.environ.get("TMS_MODEL_RELOAD_MODEL_PATH", DEFAULT_MODEL_PATH),
            prompt=os.environ.get("TMS_MODEL_RELOAD_PROMPT", DEFAULT_PROMPT),
            device=os.environ.get("TMS_MODEL_RELOAD_DEVICE", _select_default_device()),
            hook_mode=os.environ.get("TMS_MODEL_RELOAD_HOOK_MODE", DEFAULT_HOOK_MODE),
            max_input_tokens=int(
                os.environ.get("TMS_MODEL_RELOAD_MAX_INPUT_TOKENS", str(DEFAULT_MAX_INPUT_TOKENS))
            ),
        )


def test_model_reload_qwen3_1_7b():
    config = BenchmarkConfig.from_env()
    result = _run_in_subprocess(config)
    metrics = result["metrics"]

    print(json.dumps(metrics, indent=2, sort_keys=True))

    assert metrics["model_bytes"] > 0
    assert metrics["cold_load_seconds"] > 0
    assert metrics["pause_seconds"] >= 0
    assert metrics["resume_seconds"] > 0
    assert metrics["reload_seconds"] > 0
    assert metrics["cpu_backup_available"]
    assert metrics["resume_signature_max_abs_diff"] <= 1e-3
    assert metrics["reload_signature_max_abs_diff"] <= 1e-3


def _select_default_device() -> str:
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "cuda:0"

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return "cuda:0"

    candidates = []
    for line in output.splitlines():
        if not line.strip():
            continue
        index_str, free_mem_str = [part.strip() for part in line.split(",", maxsplit=1)]
        candidates.append((int(index_str), int(free_mem_str)))

    if not candidates:
        return "cuda:0"

    best_index, _ = max(candidates, key=lambda item: item[1])
    return f"cuda:{best_index}"


def _run_in_subprocess(config: BenchmarkConfig) -> dict:
    result_fd, result_path = tempfile.mkstemp(prefix="tms_model_reload_", suffix=".json")
    os.close(result_fd)

    ctx_manager = (
        torch_memory_saver.configure_subprocess()
        if config.hook_mode == "preload"
        else nullcontext()
    )

    with ctx_manager:
        ctx = multiprocessing.get_context("spawn")
        proc = ctx.Process(
            target=_subprocess_entry,
            args=(asdict(config), result_path),
        )
        proc.start()
        proc.join(timeout=30 * 60)

    try:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
            pytest.fail("Model reload benchmark timed out after 30 minutes.")

        if not os.path.exists(result_path):
            pytest.fail(f"Benchmark subprocess exited without writing {result_path}.")

        try:
            with open(result_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except JSONDecodeError as exc:
            pytest.fail(
                f"Benchmark subprocess exited with code {proc.exitcode} but wrote invalid JSON: {exc}"
            )

        if payload["status"] == "skip":
            pytest.skip(payload["reason"])

        if payload["status"] == "error":
            pytest.fail(payload["traceback"])

        assert payload["status"] == "ok", payload
        assert proc.exitcode == 0, f"Benchmark subprocess exited with code {proc.exitcode}"
        return payload
    finally:
        if os.path.exists(result_path):
            os.remove(result_path)


def _subprocess_entry(config_dict: dict, result_path: str) -> None:
    try:
        metrics = _run_benchmark(BenchmarkConfig(**config_dict))
    except SkipTest as exc:
        _write_result(result_path, {"status": "skip", "reason": str(exc)})
        return
    except Exception:
        _write_result(
            result_path,
            {
                "status": "error",
                "traceback": traceback.format_exc(),
            },
        )
        raise

    _write_result(result_path, {"status": "ok", "metrics": metrics})


def _run_benchmark(config: BenchmarkConfig) -> dict:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model_path = Path(config.model_path)
    if not model_path.exists():
        raise SkipTest(f"Model path does not exist: {model_path}")

    if not torch.cuda.is_available():
        raise SkipTest("CUDA is not available in the benchmark subprocess.")

    torch_memory_saver.torch_memory_saver.hook_mode = config.hook_mode

    device = torch.device(config.device)
    if device.type != "cuda":
        raise SkipTest(f"Only CUDA devices are supported, got {device}.")

    if device.index is not None and device.index >= torch.cuda.device_count():
        raise SkipTest(
            f"Requested device {device} but only {torch.cuda.device_count()} CUDA devices are visible."
        )

    torch.cuda.set_device(device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoded = tokenizer(
        config.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_input_tokens,
    )
    batch = {key: value.to(device) for key, value in encoded.items()}

    model, cold_load_seconds = _load_model_to_gpu(
        model_path=model_path,
        device=device,
        enable_cpu_backup=True,
    )
    reference_signature = _forward_signature(model, batch)
    tracked_param, reference_weight_slice = _capture_tracked_parameter(model)
    model_bytes = _model_nbytes(model)

    free_before_pause, total_memory = torch.cuda.mem_get_info(device)
    pause_start = time.perf_counter()
    torch_memory_saver.torch_memory_saver.pause()
    torch.cuda.synchronize(device)
    pause_seconds = time.perf_counter() - pause_start
    free_after_pause, _ = torch.cuda.mem_get_info(device)

    cpu_backup = torch_memory_saver.torch_memory_saver.get_cpu_backup(tracked_param)
    cpu_backup_available = cpu_backup is not None
    if not cpu_backup_available:
        raise AssertionError("CPU backup for model weights is missing after pause().")
    torch.testing.assert_close(
        cpu_backup.flatten()[: reference_weight_slice.numel()].float(),
        reference_weight_slice,
        rtol=0,
        atol=0,
    )

    resume_start = time.perf_counter()
    torch_memory_saver.torch_memory_saver.resume()
    torch.cuda.synchronize(device)
    resume_seconds = time.perf_counter() - resume_start
    free_after_resume, _ = torch.cuda.mem_get_info(device)

    resumed_signature = _forward_signature(model, batch)
    resume_signature_max_abs_diff = float(
        (reference_signature - resumed_signature).abs().max().item()
    )
    torch.testing.assert_close(reference_signature, resumed_signature, rtol=1e-3, atol=1e-3)

    del cpu_backup
    del tracked_param
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    reload_model, reload_seconds = _load_model_to_gpu(
        model_path=model_path,
        device=device,
        enable_cpu_backup=False,
    )
    reloaded_signature = _forward_signature(reload_model, batch)
    reload_signature_max_abs_diff = float(
        (reference_signature - reloaded_signature).abs().max().item()
    )
    torch.testing.assert_close(reference_signature, reloaded_signature, rtol=1e-3, atol=1e-3)

    del reload_model
    del batch
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    return {
        "model_path": str(model_path),
        "prompt": config.prompt,
        "prompt_num_tokens": int(encoded["input_ids"].shape[-1]),
        "device": str(device),
        "hook_mode": config.hook_mode,
        "model_bytes": model_bytes,
        "model_gib": model_bytes / 1024 ** 3,
        "cold_load_seconds": cold_load_seconds,
        "pause_seconds": pause_seconds,
        "resume_seconds": resume_seconds,
        "reload_seconds": reload_seconds,
        "resume_effective_gib_per_s": _bytes_per_second_to_gib(model_bytes, resume_seconds),
        "reload_effective_gib_per_s": _bytes_per_second_to_gib(model_bytes, reload_seconds),
        "resume_vs_reload_speedup": reload_seconds / resume_seconds,
        "cpu_backup_available": cpu_backup_available,
        "gpu_total_memory_bytes": total_memory,
        "gpu_free_before_pause_bytes": free_before_pause,
        "gpu_free_after_pause_bytes": free_after_pause,
        "gpu_free_after_resume_bytes": free_after_resume,
        "resume_signature_max_abs_diff": resume_signature_max_abs_diff,
        "reload_signature_max_abs_diff": reload_signature_max_abs_diff,
    }


def _load_model_to_gpu(model_path: Path, device: torch.device, enable_cpu_backup: bool):
    context = (
        torch_memory_saver.torch_memory_saver.region(enable_cpu_backup=enable_cpu_backup)
        if enable_cpu_backup
        else nullcontext()
    )

    start = time.perf_counter()
    with context:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            low_cpu_mem_usage=False,
        )
        model.to(device)

    model.eval()
    torch.cuda.synchronize(device)
    return model, time.perf_counter() - start


def _forward_signature(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.inference_mode():
        outputs = model(**batch)
    logits_tail = outputs.logits[:, -1, :16].float().cpu()
    return logits_tail


def _capture_tracked_parameter(model):
    for param in model.parameters():
        if param.is_cuda and param.is_contiguous() and param.numel() >= 16:
            return param, param.detach().flatten()[:16].float().cpu()
    raise RuntimeError("Could not find a contiguous CUDA parameter to validate CPU backup.")


def _model_nbytes(model) -> int:
    total = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        total += tensor.numel() * tensor.element_size()
    return total


def _bytes_per_second_to_gib(num_bytes: int, seconds: float) -> float:
    if seconds <= 0:
        return float("inf")
    return num_bytes / seconds / 1024 ** 3


def _write_result(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
