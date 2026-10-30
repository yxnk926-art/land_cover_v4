"""Evidence capture for numerical validation failures; never supplies metrics.

The snapshot is deliberately not a resumable training checkpoint. Replays inspect
the exact failed model and batch without an optimizer step or precision fallback.
"""
import json
import math
import os
import random
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NumericalValidationError(FloatingPointError):
    def __init__(self, message, *, stage, batch_index, sample_names, images, labels, amp):
        super().__init__(message)
        self.stage = str(stage)
        self.batch_index = int(batch_index)
        self.sample_names = [str(name) for name in sample_names]
        self.images = images.detach().cpu().clone()
        self.labels = labels.detach().cpu().clone()
        self.amp = bool(amp)


def _snapshot(value):
    """Keep payloads compatible with torch.load(weights_only=True)."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _json_safe(value):
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "saved_in": "snapshot.pth"}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def tensor_stats(tensor):
    value = tensor.detach()
    finite = torch.isfinite(value)
    count = int(finite.sum().item())
    stats = {"shape": list(value.shape), "dtype": str(value.dtype),
             "numel": value.numel(), "finite_count": count,
             "all_finite": count == value.numel()}
    if count:
        work = value[finite].double()
        stats.update(min=float(work.min().item()), max=float(work.max().item()),
                     abs_max=float(work.abs().max().item()))
    return stats


def inspect_model_state(model):
    """Scan every parameter/buffer and BN's eval-time effective gain."""
    report = {"parameters": {}, "buffers": {}, "batchnorm": {}, "all_finite": True}
    for kind, items in (("parameters", model.named_parameters()),
                        ("buffers", model.named_buffers())):
        for name, tensor in items:
            stats = tensor_stats(tensor)
            report[kind][name] = stats
            report["all_finite"] &= stats["all_finite"]
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            entry = {"eps": float(module.eps), "track_running_stats": module.track_running_stats}
            if module.running_var is not None:
                variance = module.running_var.detach().double()
                weight = (module.weight.detach().double() if module.affine
                          else torch.ones_like(variance))
                entry["running_var"] = tensor_stats(variance)
                entry["negative_variance_count"] = int((variance < 0).sum().item())
                entry["effective_gain"] = tensor_stats(weight / torch.sqrt(variance + module.eps))
            if module.running_mean is not None:
                entry["running_mean"] = tensor_stats(module.running_mean)
            report["batchnorm"][name or "<root>"] = entry
    return report


def _tensors(value, path="value"):
    if torch.is_tensor(value):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _tensors(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _tensors(item, f"{path}[{index}]")


@torch.no_grad()
def replay_validation_failure(model, images, labels, *, amp):
    """Report one replay, restoring module modes, buffers and all RNG state.

    Hooks cover composite modules as well as leaves. They identify the first
    non-finite *module output*, not every internal functional operation.
    """
    from common.engine import autocast_ctx, primary_logits

    device = next((item.device for item in model.parameters()),
                  next((item.device for item in model.buffers()), images.device))
    requested_amp = bool(amp)
    use_amp = requested_amp and device.type == "cuda"
    if requested_amp and not use_amp:
        return {"requested_amp": True, "status": "skipped", "reason": "CUDA AMP requires CUDA"}
    if not use_amp and any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        return {"requested_amp": False, "status": "skipped", "reason": "FP32 replay requires FP32 model weights; model was not mutated"}

    report = {"requested_amp": requested_amp, "effective_amp": use_amp,
              "device": str(device), "model_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
              "first_nonfinite_module": None, "status": "pending"}
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    rng_python, rng_numpy = random.getstate(), np.random.get_state()
    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):
            if report["first_nonfinite_module"] is not None:
                return
            if any(not bool(torch.isfinite(item).all()) for _, item in _tensors(output)):
                report["first_nonfinite_module"] = {
                    "name": name or "<root>", "type": type(module).__name__,
                    "inputs": {key: tensor_stats(item) for key, item in _tensors(inputs)},
                    "outputs": {key: tensor_stats(item) for key, item in _tensors(output)},
                }
        return hook

    try:
        model.eval()
        for name, module in model.named_modules():
            hooks.append(module.register_forward_hook(make_hook(name)))
        with autocast_ctx(device.type, enabled=use_amp):
            logits = primary_logits(model(images.to(device)))
            report["logits"] = tensor_stats(logits)
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            report["resized_logits"] = tensor_stats(logits)
        report["status"] = ("finite" if report["logits"]["all_finite"]
                            and report["resized_logits"]["all_finite"] else "non_finite")
    except Exception as exc:
        report.update(status="error", error_type=type(exc).__name__, error=str(exc))
    finally:
        for hook in hooks:
            hook.remove()
        for buffer, original in buffers:
            buffer.copy_(original)
        for module, training in modes:
            module.training = training
        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)
        random.setstate(rng_python)
        np.random.set_state(rng_numpy)
    return report


def _atomic_save(path, writer):
    fd, temporary = tempfile.mkstemp(prefix="tmp_", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_validation_failure(model, error, output_dir, *, context, optimizer=None, scaler=None):
    """Save snapshot first, then a diagnostic report; return report.json Path.

    Callers must re-raise the original NumericalValidationError even if saving
    fails. A replay never substitutes for the failed validation or its metrics.
    """
    destination = Path(output_dir) / ("numerical_failure_" + uuid.uuid4().hex[:12])
    destination.mkdir(parents=True, exist_ok=False)
    failure = {"message": str(error), "stage": error.stage, "batch_index": error.batch_index,
               "sample_names": error.sample_names, "amp": error.amp}
    payload = {"kind": "numerical_failure", "format_version": 1,
               "diagnosis_only": True, "failure": failure,
               "model": _snapshot(model.state_dict()), "images": error.images,
               "labels": error.labels, "context": _snapshot(context),
               "module_training_modes": {name: module.training for name, module in model.named_modules()},
               "rng_torch_cpu": torch.get_rng_state(),
               "rng_torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
               "rng_python": _snapshot(random.getstate()), "rng_numpy": _snapshot(np.random.get_state())}
    if optimizer is not None:
        payload["optimizer"] = _snapshot(optimizer.state_dict())
    if scaler is not None:
        payload["scaler"] = _snapshot(scaler.state_dict())
    _atomic_save(destination / "snapshot.pth", lambda path: torch.save(payload, path))
    report = {"kind": "numerical_failure", "format_version": 1, "diagnosis_only": True,
              "snapshot": "snapshot.pth", "failure": failure, "context": _json_safe(payload["context"]),
              "torch_version": str(torch.__version__),
              "cuda_version": torch.version.cuda,
              "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32,
                       "cudnn": torch.backends.cudnn.allow_tf32},
              "inputs": tensor_stats(error.images), "labels": tensor_stats(error.labels)}
    report_path = destination / "report.json"

    def save_report():
        def write(path):
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(_json_safe(report), stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        _atomic_save(report_path, write)

    save_report()
    try:
        report["model_state"] = inspect_model_state(model)
        report["replays"] = {
            "amp": replay_validation_failure(model, error.images, error.labels, amp=True),
            "fp32": replay_validation_failure(model, error.images, error.labels, amp=False),
        }
    except Exception as exc:
        report["diagnostic_error"] = {"type": type(exc).__name__, "message": str(exc)}
    save_report()
    return report_path
