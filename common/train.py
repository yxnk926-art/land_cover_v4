# -*- coding: utf-8 -*-
"""统一细类协议 v4 的训练入口。

示例：
    python common/train.py --config config_unified_unet_r50
    python common/train.py --config config_unified_ddrnet --run-tag unified_v2_s42 --seed 42

配置必须显式指定。输出由各 leaf 的 ``EXP_NAME``、``CKPT_DIR`` 与 ``OUT_DIR`` 隔离。
"""
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJECT_ROOT)
import argparse
import hashlib
import importlib
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

from common.dataset import build_unified_dataloaders
from common.engine import (adjust_learning_rate, AverageMeter, autocast_ctx, make_grad_scaler,
                           build_optimizer, build_criterion, primary_logits)
from common.early_stopping import EarlyStopping
from common.models import build_model
from common.native_datasets import check_selection_lists
from common.numerical_diagnostics import (
    NumericalValidationError, save_validation_failure,
)
from common.pretrained import apply_pretrained, resolve_pretrained
from common.unified_protocol import evaluate_unified, format_unified_scores
from common.unified_taxonomy import load_taxonomy

# Set when run as a script; tests and tools assign their own.
cfg = None
cli_args = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_reproducibility(config):
    """配置确定性执行；DETERMINISTIC 与 cuDNN benchmark 始终互斥。

    需要严格可复现的实验应显式设置 ``DETERMINISTIC=True``；若某算子没有确定性实现，
    默认立即报错，也可用 ``DETERMINISTIC_WARN_ONLY=True`` 临时改为警告。
    """
    deterministic = bool(getattr(config, "DETERMINISTIC", False))
    requested_benchmark = bool(getattr(config, "CUDNN_BENCHMARK", True))
    warn_only = bool(getattr(config, "DETERMINISTIC_WARN_ONLY", False))

    if deterministic:
        # 必须在 CUDA 上下文建立前提供，供确定性 cuBLAS 路径使用。
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = requested_benchmark and not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    return {
        "deterministic": deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "warn_only": warn_only,
    }


def _load_checkpoint_file(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_initial_weights(model, path):
    """Partially load a model/backbone checkpoint and report exact matches.

    Supported payloads are a raw state_dict or a dict containing ``model`` or
    ``state_dict``.  Common ``module.``/``model.`` wrappers are stripped; a raw
    backbone state can also match by adding ``backbone.``.  Shape mismatches
    are intentionally skipped rather than silently reshaped.
    """
    payload = _load_checkpoint_file(path)
    if isinstance(payload, dict) and payload.get("kind") == "numerical_failure":
        raise ValueError("数值失败快照仅供诊断，禁止作为初始化权重")
    if isinstance(payload, dict) and "model" in payload:
        payload = payload["model"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a state_dict mapping")

    current = model.state_dict()
    matched = {}
    skipped = []
    for raw_key, value in payload.items():
        if not torch.is_tensor(value):
            continue
        key = raw_key
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        candidates = (key, f"backbone.{key}")
        destination = next(
            (candidate for candidate in candidates
             if candidate in current and current[candidate].shape == value.shape),
            None,
        )
        if destination is None:
            skipped.append(raw_key)
        else:
            matched[destination] = value
    if not matched:
        raise ValueError(
            f"{path} has no parameter names/shapes compatible with {type(model).__name__}"
        )
    incompatible = model.load_state_dict(matched, strict=False)
    return len(matched), len(skipped), len(incompatible.missing_keys)


def config_snapshot(module):
    """Keep simple resolved config values inside checkpoints for provenance."""
    snapshot = {}
    for name in dir(module):
        if not name.isupper():
            continue
        value = getattr(module, name)
        if isinstance(value, (str, int, float, bool, type(None), list, tuple, dict)):
            snapshot[name] = value
    return snapshot


# Fields that must not drift within one checkpoint lineage.  A key a leaf does
# not define is skipped, so each model only adds its own keys.
RESUME_GUARD_KEYS = (
    "PROTOCOL_ID", "MODEL_NAME", "IN_CHANNELS", "NUM_CLASSES",
    "IGNORE_INDEX", "CROP_SIZE", "SCALE_RANGE",
    "MEAN", "STD", "IMAGE_PAD_MODE", "HFLIP_PROB", "VFLIP_PROB",
    "ROT90_PROB", "AUG_LEGACY_ROT90", "BRIGHTNESS", "CONTRAST",
    "SATURATION", "BLUR_PROB", "BLUR_SIGMA", "CUTOUT_PROB",
    "CUTOUT_FRAC", "CUTOUT_LABEL_MODE", "CUTOUT_N", "LOSS_NAME",
    "DEEP_SUP_WEIGHTS", "DICE_WEIGHT", "BATCH_SIZE", "LR",
    "WEIGHT_DECAY", "BACKBONE_LR_MULT", "NO_DECAY_BN_BIAS",
    "CLIP_GRAD", "WARMUP_ITERS", "POLY_POWER", "SEED",
    "DETERMINISTIC", "DETERMINISTIC_DATA_LOADER",
    "AMP", "EVAL_AMP", "SCHEDULE_EPOCHS", "EPOCHS", "EVAL_INTERVAL",
    "EARLY_STOP_PATIENCE", "EARLY_STOP_MIN_DELTA", "DETERMINISTIC_WARN_ONLY",
    "VAL_BATCH_SIZE", "NUM_WORKERS", "PAD_DIVISOR",
    "DTYPE_SCALE", "STRICT_LABEL_SCHEMA",
    "ITERS_PER_EPOCH", "EVAL_WINDOW", "EVAL_STRIDE",
    # Model structure and ImageNet initialisation.
    "MOBILENET_WIDTH_MULT", "OUTPUT_STRIDE", "ASPP_RATES", "ASPP_CHANNELS",
    "DECODER_CHANNELS", "LOW_LEVEL_CHANNELS", "MODEL_DROPOUT", "LRASPP_CHANNELS",
    "LRASPP_CONTEXT_POOL", "SEGFORMER_VARIANT", "SEGFORMER_DECODER_DIM",
    "DROP_PATH_RATE", "PIDNET_VARIANT", "PIDNET_AUXILIARY",
    "BOUNDARY_WEIGHT", "BOUNDARY_AWARE_WEIGHT", "BOUNDARY_THRESHOLD",
    "BOUNDARY_DILATION", "ENCODER_NAME", "UNET_DECODER_CHANNELS", "DEEPLAB_BACKBONE",
    "SWIN_VARIANT", "SWIN_WINDOW_SIZE", "UPERNET_CHANNELS",
    "UPERNET_POOL_SCALES", "UPERNET_AUXILIARY", "AUX_CHANNELS",
    "DDRNET_VARIANT", "DDRNET_AUXILIARY", "PRETRAINED", "PRETRAINED_SHA256",
    "MOBILENET_REDUCED_TAIL", "MOBILENET_BN_EPS",
    # Taxonomy, training datasets (with their split and list identities), label
    # stacking and sampling.
    "UNIFIED_TAXONOMY_ID", "UNIFIED_TAXONOMY_SHA256", "FINE_CLASS_NAMES",
    "UNIFIED_DATASET_IDS", "UNIFIED_DATASETS", "LABEL_OFFSETS", "SET_MEMBERSHIP",
    "DATASET_WEIGHTS", "SELECTION_METRIC",
)


def protocol_snapshot(module):
    """Resolved fields that must not drift within one checkpoint lineage."""
    full = config_snapshot(module)
    return {key: full[key] for key in RESUME_GUARD_KEYS if key in full}


def protocol_fingerprint(snapshot):
    payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_resume_checkpoint(ckpt, current_fingerprint, module):
    """Fail closed for unverified or protocol-incompatible resume payloads."""
    if ckpt.get("kind") == "numerical_failure":
        raise RuntimeError("数值失败快照仅供诊断，禁止作为续训 checkpoint")
    saved = ckpt.get("protocol_fingerprint")
    if saved is None:
        if not getattr(module, "ALLOW_UNVERIFIED_RESUME", False):
            raise RuntimeError(
                "checkpoint 缺少 protocol_fingerprint，拒绝不受验证的续训。"
                "如仅为迁移历史 checkpoint，需显式设置 ALLOW_UNVERIFIED_RESUME=True。"
            )
    elif (saved != current_fingerprint
          and not getattr(module, "ALLOW_CONFIG_MISMATCH_ON_RESUME", False)):
        raise RuntimeError(
            "拒绝恢复协议不一致的 checkpoint："
            f"saved={saved[:12]} current={current_fingerprint[:12]}。"
            "请使用新的 EXP_NAME，或在确认风险后显式设置 "
            "ALLOW_CONFIG_MISMATCH_ON_RESUME=True。"
        )
    if "rng_state" not in ckpt and not getattr(module, "ALLOW_UNVERIFIED_RESUME", False):
        raise RuntimeError(
            "checkpoint 缺少 rng_state，无法保证续训随机轨迹；拒绝续训。"
            "历史迁移需显式设置 ALLOW_UNVERIFIED_RESUME=True。"
        )


def capture_rng_state(train_loader=None, val_loader=None):
    np_state = np.random.get_state()
    payload = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": np_state[0],
            "state": np_state[1].tolist(),
            "position": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    for name, loader in (("train_loader", train_loader), ("val_loader", val_loader)):
        generator = getattr(loader, "generator", None)
        payload[name] = generator.get_state() if generator is not None else None
    return payload


def restore_rng_state(payload, train_loader=None, val_loader=None):
    if not isinstance(payload, dict):
        raise RuntimeError("rng_state 格式非法")
    random.setstate(tuple(payload["python"]))
    np_state = payload["numpy"]
    np.random.set_state((
        np_state["bit_generator"],
        np.asarray(np_state["state"], dtype=np.uint32),
        int(np_state["position"]),
        int(np_state["has_gauss"]),
        float(np_state["cached_gaussian"]),
    ))
    torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available() and payload.get("torch_cuda"):
        torch.cuda.set_rng_state_all(payload["torch_cuda"])
    for name, loader in (("train_loader", train_loader), ("val_loader", val_loader)):
        state = payload.get(name)
        generator = getattr(loader, "generator", None)
        if state is not None and generator is not None:
            generator.set_state(state)


def best_checkpoint_payload(model, epoch, scores, module, resolved):
    """Build a provenance-bearing best checkpoint without training state.

    The final evaluation must know that a checkpoint uses the exact taxonomy,
    label stacking and splits, so the state sits under ``model`` next to immutable
    selection metadata.  Test metrics are intentionally absent.
    """
    payload = {
        "format_version": 2,
        "kind": "best_model",
        "epoch": int(epoch),
        "model": model.state_dict(),
        "source_selection_metric": "miou",
        "source_selection_value": float(scores["miou"]),
        "config": config_snapshot(module),
        "protocol": resolved["protocol"],
        "protocol_fingerprint": resolved["protocol_fingerprint"],
        "source_selection_by_dataset": {
            name: float(value["miou"]) for name, value in scores["datasets"].items()},
    }
    return payload


def write_resolved_config(out_dir, module):
    """Atomically persist the effective config used by this run."""
    protocol = protocol_snapshot(module)
    payload = {
        "config_module": module.__name__,
        "config_file": os.path.abspath(module.__file__),
        "resolved": config_snapshot(module),
        "protocol": protocol,
        "protocol_fingerprint": protocol_fingerprint(protocol),
    }
    path = os.path.join(out_dir, "resolved_config.json")
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, path)
    return payload


def validate_training_domain(module):
    """Reject anything but train and selection lists before creating any output."""
    role = getattr(module, "DATASET_ROLE", None)
    if role is not None and role != "source":
        raise ValueError("DATASET_ROLE 必须为 source")
    for entry in module.UNIFIED_DATASETS.values():
        check_selection_lists(entry["root"], entry["train_list"], entry["val_list"])
    if load_taxonomy(module.TAXONOMY_PATH).sha256 != module.UNIFIED_TAXONOMY_SHA256:
        raise ValueError("统一细类文件在配置读入后被改动，拒绝训练")


def apply_run_overrides(module, args):
    """Isolate fresh baselines without modifying the imported leaf files."""
    if args.seed is not None:
        if not 0 <= args.seed < 2 ** 32:
            raise ValueError("seed 必须在 [0, 2**32)")
        module.SEED = args.seed
    if args.run_tag is None:
        return
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_tag):
        raise ValueError("run-tag 必须以字母或数字开头，且仅含字母、数字、点、下划线和连字符")
    status = getattr(module, "UNIFIED_TAXONOMY_STATUS", None)
    if status is not None and status != "final":
        raise ValueError(f"统一细类体系仍是 {status}，不能做正式（--run-tag）训练；"
                         "审查定稿后把细类文件的 status 改为 final")
    if getattr(module, "RESUME", False) or getattr(module, "INIT_CHECKPOINT", None):
        raise ValueError("新 baseline 必须从头训练，禁止 RESUME/INIT_CHECKPOINT")
    for key, root in (("OUT_DIR", module.OUTPUT_ROOT),
                      ("CKPT_DIR", module.CHECKPOINT_ROOT)):
        parent = Path(root).resolve() / "baselines" / args.run_tag
        path = parent / module.MODEL_SLUG
        if path.resolve() != path or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
            raise ValueError(f"baseline 输出已存在或路径发生重定向，拒绝覆盖：{path}")
        setattr(module, key, str(path))
    module.RUN_TAG = args.run_tag
    module.EXP_NAME = f"{module.EXP_NAME}_{args.run_tag}"


def resolve_pretrained_identity(module):
    """Fail fast on a missing/altered checkpoint and pin its hash in the config."""
    name = getattr(module, "PRETRAINED", None)
    if not name:
        return None
    spec, _ = resolve_pretrained(name, getattr(module, "PRETRAINED_DIR", None))
    module.PRETRAINED_SHA256 = spec.sha256
    return spec


def main():
    apply_run_overrides(cfg, cli_args)
    validate_training_domain(cfg)
    resolve_pretrained_identity(cfg)
    reproducibility = configure_reproducibility(cfg)
    set_seed(cfg.SEED)
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_path = os.path.join(cfg.OUT_DIR, "train_log.txt")
    log_file = open(log_path, "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    resolved = write_resolved_config(cfg.OUT_DIR, cfg)

    # ------------------------------------------------------------ 数据与模型
    # One train loader over every dataset; val_loader maps dataset -> loader.
    train_loader, val_loader = build_unified_dataloaders(cfg)
    iters_per_epoch = len(train_loader)
    # 调度视野可以长于实际训练长度：试跑只训练前 40 个 epoch 时，仍沿用完整调度的
    # 学习率轨迹，结果才能和正式训练的同一时刻比较。
    schedule_epochs = cfg.SCHEDULE_EPOCHS
    max_iter = iters_per_epoch * schedule_epochs

    model = build_model(cfg).to(device)
    last_ckpt = os.path.join(cfg.CKPT_DIR, "checkpoint_last.pth")
    pretrained_name = getattr(cfg, "PRETRAINED", None)
    if pretrained_name and not (cfg.RESUME and os.path.exists(last_ckpt)):
        report = apply_pretrained(model, pretrained_name,
                                  getattr(cfg, "PRETRAINED_DIR", None))
        log(f"[*] 预训练权重 {report['name']} ({report['sha256'][:12]}): "
            f"载入 {report['loaded_tensors']}, 丢弃分类头 {report['dropped_tensors']}, "
            f"新初始化 {report['new_tensors']}")
    init_checkpoint = getattr(cfg, "INIT_CHECKPOINT", None)
    if init_checkpoint and not (cfg.RESUME and os.path.exists(last_ckpt)):
        if not os.path.isfile(init_checkpoint):
            raise FileNotFoundError(f"INIT_CHECKPOINT 不存在: {init_checkpoint}")
        matched, skipped, missing = load_initial_weights(model, init_checkpoint)
        log(f"[*] 初始化权重 {init_checkpoint}: 匹配 {matched}, "
            f"跳过 {skipped}, 模型未覆盖 {missing}")
    criterion = build_criterion(cfg).to(device)
    optimizer = build_optimizer(model, cfg)  # 分组学习率/decay 由 config 的开关决定，默认单组
    scaler = make_grad_scaler(enabled=cfg.AMP and device.type == "cuda")
    eval_amp = bool(getattr(cfg, "EVAL_AMP", False)) and device.type == "cuda"

    # patience 按“验证次数”计数；EVAL_INTERVAL=1 时才等同于 epoch 数。
    early_stop_patience = getattr(cfg, "EARLY_STOP_PATIENCE", None)
    early_stop_min_delta = getattr(cfg, "EARLY_STOP_MIN_DELTA", 0.0)
    if early_stop_patience in (None, 0):
        early_stopper = None
    else:
        early_stopper = EarlyStopping(early_stop_patience, early_stop_min_delta)

    # ------------------------------------------------------------ 断点续训
    start_epoch, best_miou = 0, 0.0
    if cfg.RESUME and os.path.exists(last_ckpt):
        ckpt = _load_checkpoint_file(last_ckpt)
        current_fingerprint = resolved["protocol_fingerprint"]
        validate_resume_checkpoint(ckpt, current_fingerprint, cfg)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        if "rng_state" in ckpt:
            restore_rng_state(ckpt["rng_state"], train_loader, val_loader)
        best_miou = ckpt.get("best_miou", 0.0)
        log(f"[*] 从 {last_ckpt} 续训: epoch {start_epoch}, best mIoU {best_miou:.6f}")
        if early_stopper is not None:
            restored = early_stopper.load_state_dict(ckpt.get("early_stopping"))
            if not restored:
                # 旧 checkpoint 没有早停状态，或用户修改了 min_delta：安全地从
                # 已知 raw best 开始一个新的 patience 窗口，而不是猜测历史计数。
                early_stopper.reset(best_miou)
                log("[*] checkpoint 无兼容早停状态；从当前 best mIoU 重新计 patience")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"[*] config: {cfg.__file__}")
    log(f"[*] precision: train_amp={scaler.is_enabled()} | eval_amp={eval_amp}")
    log("[*] reproducibility: "
        f"deterministic={reproducibility['deterministic']} | "
        f"cudnn_benchmark={reproducibility['cudnn_benchmark']} | "
        f"warn_only={reproducibility['warn_only']}")
    log(f"[*] 模型 {cfg.MODEL_NAME} | 设备 {device} | 参数量 {n_params:.2f}M | "
        f"每 epoch {iters_per_epoch} iter | 共 {cfg.EPOCHS} epoch"
        f"（调度视野 {schedule_epochs} epoch） | "
        f"batch {cfg.BATCH_SIZE} | crop {cfg.CROP_SIZE}")
    # 旧格式 checkpoint 续训时 load_state_dict 会丢掉 initial_lr，这里用 cfg.LR 兜底
    log("[*] optimizer 参数组: " + " | ".join(
        f"lr={pg.get('initial_lr', cfg.LR):.1e} wd={pg['weight_decay']} n={len(pg['params'])}"
        for pg in optimizer.param_groups))
    if early_stopper is not None:
        log(f"[*] early stopping: patience={early_stopper.patience} 次验证 | "
            f"min_delta={early_stopper.min_delta:.6f}")
        if early_stopper.should_stop:
            log("[*] checkpoint 已满足早停条件；无需继续训练。若要继续，请增大 "
                "EARLY_STOP_PATIENCE 或将其设为 None")
            log_file.close()
            return

    # ------------------------------------------------------------ 训练循环
    for epoch in range(start_epoch, cfg.EPOCHS):
        model.train()
        # Fixed-length epochs (ITERS_PER_EPOCH): the epoch selects the slice of
        # the permutation stream, which also keeps a resumed run on that stream.
        sampler = getattr(train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        loss_meter = AverageMeter()
        t0 = time.time()

        for it, (img, lab, _) in enumerate(train_loader):
            cur_iter = epoch * iters_per_epoch + it
            lr = adjust_learning_rate(optimizer, cfg.LR, cur_iter, max_iter,
                                      warmup_iters=cfg.WARMUP_ITERS, power=cfg.POLY_POWER)

            img = img.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(device.type, enabled=scaler.is_enabled()):
                logits_list = model(img)
                loss = criterion(logits_list, lab)

            # GradScaler can skip an optimizer update when gradients overflow,
            # but it cannot undo BatchNorm running-stat updates that already
            # happened during a non-finite forward pass.  Stop immediately so
            # the epoch-end checkpoint cannot persist a poisoned model state.
            logits = primary_logits(logits_list).detach()
            forward_finite = (
                torch.isfinite(loss) & torch.isfinite(logits).all()
            ).item()
            if not forward_finite:
                finite = torch.isfinite(logits)
                bad_count = int((~finite).sum().item())
                total_count = logits.numel()
                message = (
                    f"non-finite forward at epoch={epoch}, iter={it}, "
                    f"global_iter={cur_iter}; non-finite logits "
                    f"{bad_count}/{total_count}; "
                    f"loss_finite={torch.isfinite(loss).item()}; "
                    f"AMP={scaler.is_enabled()}"
                )
                log(f"[ERROR] {message}")
                raise FloatingPointError(message)

            scaler.scale(loss).backward()
            if cfg.CLIP_GRAD is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD)
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.size(0))
            if it % cfg.LOG_INTERVAL == 0:
                done = cur_iter + 1
                eta = (time.time() - t0) / (it + 1) * (iters_per_epoch - it - 1)
                component_text = ""
                if getattr(criterion, "last_components", None):
                    component_text = " | " + " ".join(
                        f"{name}={value.item():.4f}"
                        for name, value in criterion.last_components.items()
                        if name != "total"
                    )
                log(f"E[{epoch}/{cfg.EPOCHS - 1}] I[{it}/{iters_per_epoch}] "
                    f"loss {loss.item():.4f} (avg {loss_meter.avg:.4f}) lr {lr:.2e} "
                    f"eta {eta / 60:.1f}min iter {done}/{max_iter}{component_text}")

        log(f"== epoch {epoch} 完成: 平均 loss {loss_meter.avg:.4f}, 耗时 {(time.time() - t0) / 60:.1f}min")

        # -------------------------------------------------------- 验证与保存
        if (epoch + 1) % cfg.EVAL_INTERVAL == 0 or epoch == cfg.EPOCHS - 1:
            try:
                scores = evaluate_unified(model, val_loader, device, cfg, amp=eval_amp)
            except NumericalValidationError as exc:
                # Save this exact post-update model, not the previous epoch's
                # last checkpoint.  Diagnostics can never become a best model.
                try:
                    report_path = save_validation_failure(
                        model, exc, Path(cfg.OUT_DIR) / "diagnostics" / f"epoch_{epoch:04d}",
                        context={
                            "epoch": epoch,
                            "config": config_snapshot(cfg),
                            "protocol": resolved["protocol"],
                            "protocol_fingerprint": resolved["protocol_fingerprint"],
                            "rng_state": capture_rng_state(train_loader, val_loader),
                        },
                        optimizer=optimizer, scaler=scaler,
                    )
                    log(f"[ERROR] 验证数值异常，诊断已保存：{report_path}")
                except Exception as diagnostic_error:
                    log(f"[ERROR] 保存诊断失败：{type(diagnostic_error).__name__}: {diagnostic_error}")
                log_file.close()
                raise
            log(f"[val] epoch {epoch}\n" + format_unified_scores(scores, cfg))

            # raw best 与“足以重置 patience 的有效提升”是两个概念：微小提升仍保存
            # best_model，但只有超过 min_delta 才延长训练。
            if scores["miou"] > best_miou:
                best_miou = scores["miou"]
                best_path = os.path.join(cfg.CKPT_DIR, "best_model.pth")
                torch.save(best_checkpoint_payload(model, epoch, scores, cfg, resolved), best_path)
                log(f"[*] 新最优模型: epoch {epoch}, val mIoU {best_miou:.6f} -> {best_path}")

            should_stop = False
            if early_stopper is not None:
                should_stop, meaningful_gain = early_stopper.step(scores["miou"])
                status = "有效提升，patience 已重置" if meaningful_gain else "未达到 min_delta"
                log(f"[*] early stopping: {status} | ref "
                    f"{early_stopper.reference_score:.6f} | "
                    f"bad {early_stopper.bad_evals}/{early_stopper.patience}")

            torch.save({
                "format_version": 3,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "best_miou": best_miou,
                "miou": scores["miou"],
                "early_stopping": (early_stopper.state_dict()
                                   if early_stopper is not None else None),
                "config": config_snapshot(cfg),
                "protocol": resolved["protocol"],
                "protocol_fingerprint": resolved["protocol_fingerprint"],
                "rng_state": capture_rng_state(train_loader, val_loader),
            }, last_ckpt)

            if should_stop:
                log(f"[*] 触发早停: 连续 {early_stopper.bad_evals} 次验证未取得 "
                    f"> {early_stopper.min_delta:.6f} 的有效提升；"
                    f"停止于 epoch {epoch}，raw best mIoU {best_miou:.6f}")
                break

    log(f"[*] 训练结束，best val mIoU = {best_miou:.6f}")
    log_file.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="configs/ 下的 leaf 文件名，不带 .py")
    parser.add_argument("--run-tag", help="为全新 baseline 隔离输出；拒绝覆盖已有结果")
    parser.add_argument("--seed", type=int, help="显式训练随机种子")
    return parser.parse_args(argv)


if __name__ == "__main__":
    cli_args = parse_args()
    cfg = importlib.import_module(f"configs.{cli_args.config}")
    main()
