#!/usr/bin/env python3
"""Train the unified fine-class models sequentially, with isolated artifacts.

Without ``--models`` all ten leaves run, fastest first.  ``LAND_COVER_UNIFIED_DATASETS``
narrows the training datasets.

This launcher only reads existing leaf configurations.  Training owns the final
protocol checks and writes the resolved configuration/checkpoint fingerprint.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import unified_protocol  # noqa: E402


@dataclass(frozen=True)
class BaselineRun:
    model: str
    config: str
    command: tuple[str, ...]
    output_dir: Path
    checkpoint_dir: Path
    log_path: Path
    # The train and selection lists of every dataset.
    lists: tuple
    summary: dict
    resume: bool = False


def run_tag(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError(
            "run-tag must start with a letter/digit and contain only letters, "
            "digits, underscores, dots or hyphens"
        )
    return value


def train_seed(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be an integer") from exc
    if not 0 <= seed <= 2**32 - 1:
        raise argparse.ArgumentTypeError("seed must be in [0, 2**32 - 1]")
    return seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(unified_protocol.MODEL_CONFIGS),
                        help="default: all ten leaves, fastest first")
    parser.add_argument("--run-tag", type=run_tag, required=True)
    parser.add_argument("--seed", type=train_seed, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="continue this run-tag after an interruption: a model with "
                             "checkpoint_last.pth resumes (same fingerprint only), a finished "
                             "one exits at once, one without a checkpoint starts from zero")
    parser.add_argument("--dry-run", action="store_true",
                        help="print commands and resolved protocol without training or writing")
    return parser


def build_runs(models: Sequence[str], tag: str, seed: int,
               resume: bool = False) -> list[BaselineRun]:
    """Resolve plans without mutating the imported modules or touching outputs."""
    run_tag(tag)
    train_seed(str(seed))
    if len(models) != len(set(models)):
        raise ValueError("duplicate model names are not allowed")
    # The training child reads the same variable, so plan and run agree.
    unified_datasets = os.environ.get(unified_protocol.DATASETS_ENV) or None
    runs = []
    for model in models:
        config = unified_protocol.MODEL_CONFIGS[model]
        cfg = unified_protocol.import_unified_config(
            config, unified_datasets and unified_datasets.split(","))
        if cfg.DATASET_ROLE != "source":
            raise ValueError(f"{config}: baseline training must use source data")
        if getattr(cfg, "EVAL_AMP", None) is not False:
            raise ValueError(f"{config}: this baseline requires explicit EVAL_AMP=False")
        if cfg.RESUME or cfg.INIT_CHECKPOINT is not None:
            raise ValueError(f"{config}: baseline requires RESUME=False and INIT_CHECKPOINT=None")
        output_dir = Path(cfg.OUTPUT_ROOT) / "baselines" / tag / cfg.MODEL_SLUG
        checkpoint_dir = Path(cfg.CHECKPOINT_ROOT) / "baselines" / tag / cfg.MODEL_SLUG
        log_path = Path(cfg.OUTPUT_ROOT) / "baselines" / tag / "launcher" / f"{model}.log"
        lists = tuple(Path(entry[key]) for entry in cfg.UNIFIED_DATASETS.values()
                      for key in ("train_list", "val_list"))
        summary = {
            "model": model,
            "protocol": cfg.PROTOCOL_ID,
            "seed": seed,
            "epochs_cap": cfg.EPOCHS,
            "schedule_epochs": cfg.SCHEDULE_EPOCHS,
            "early_stop_patience": cfg.EARLY_STOP_PATIENCE,
            "early_stop_min_delta": cfg.EARLY_STOP_MIN_DELTA,
            "train_amp": cfg.AMP,
            "eval_amp": cfg.EVAL_AMP,
            "scale_range": cfg.SCALE_RANGE,
            "batch_size": cfg.BATCH_SIZE,
            "lr": cfg.LR,
            "weight_decay": cfg.WEIGHT_DECAY,
            "poly_power": cfg.POLY_POWER,
            "output_dir": str(output_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "log_path": str(log_path),
            "datasets": list(cfg.UNIFIED_DATASET_IDS),
            "taxonomy": cfg.UNIFIED_TAXONOMY_ID,
            "taxonomy_status": cfg.UNIFIED_TAXONOMY_STATUS,
            "fine_classes": cfg.NUM_CLASSES,
            "iters_per_epoch": cfg.ITERS_PER_EPOCH,
            "max_iters": cfg.ITERS_PER_EPOCH * cfg.SCHEDULE_EPOCHS,
            "crop_size": cfg.CROP_SIZE,
            "eval_interval": cfg.EVAL_INTERVAL,
        }
        command = (sys.executable, "-u", str(PROJECT_ROOT / "common" / "train.py"),
                   "--config", config, "--run-tag", tag, "--seed", str(seed))
        if resume:
            command += ("--resume",)
        runs.append(BaselineRun(model, config, command, output_dir, checkpoint_dir,
                                log_path, lists, summary, resume))
    return runs


def preflight(runs: Sequence[BaselineRun], *, check_data: bool = True) -> None:
    """Check every selected run before starting even the first child."""
    for run in runs:
        for path in (run.output_dir, run.checkpoint_dir):
            if path.is_symlink():
                raise ValueError(f"refusing a symlink artifact directory: {path}")
            if path.exists() and (not path.is_dir() or (any(path.iterdir()) and not run.resume)):
                raise ValueError(f"artifact path is already occupied; choose a new run-tag: {path}")
        if run.log_path.is_symlink() or (run.log_path.exists() and not run.resume):
            raise ValueError(f"launcher log already exists; choose a new run-tag: {run.log_path}")
        if check_data:
            for path in run.lists:
                if not path.is_file():
                    raise ValueError(f"source list is missing: {path}")
                if not path.read_text(encoding="utf-8").strip():
                    raise ValueError(f"source list is empty: {path}")


def _stop_process_group(proc: subprocess.Popen) -> None:
    """Stop the training child and its workers when the launcher is interrupted."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        proc.wait(timeout=15)
    except ProcessLookupError:
        return
    except (subprocess.TimeoutExpired, OSError):
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.wait()


def run_one(run: BaselineRun) -> int:
    run.log_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also closes the preflight/start race for this log.
    with run.log_path.open("a" if run.resume else "x", encoding="utf-8", buffering=1) as log:
        log.write(json.dumps(run.summary, ensure_ascii=False, sort_keys=True) + "\n")
        kwargs = {"start_new_session": True} if os.name == "posix" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
        }
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            run.command, cwd=PROJECT_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, **kwargs,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                print(line, end="", flush=True)
            returncode = proc.wait()
            log.write(f"\nlauncher_returncode={returncode}\n")
            return returncode
        except KeyboardInterrupt:
            _stop_process_group(proc)
            log.write("\nlauncher_interrupted=true\n")
            raise
        except BaseException:
            _stop_process_group(proc)
            raise
        finally:
            if proc.stdout is not None:
                proc.stdout.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models = args.models or list(unified_protocol.MODEL_CONFIGS)
    try:
        runs = build_runs(models, args.run_tag, args.seed, args.resume)
        preflight(runs, check_data=not args.dry_run)
    except (ValueError, OSError) as exc:
        print(f"Baseline preflight failed: {exc}", file=sys.stderr)
        return 2
    for run in runs:
        print(json.dumps(run.summary, ensure_ascii=False, sort_keys=True), flush=True)
        print(shlex.join(run.command), flush=True)
    if args.dry_run:
        print("Dry-run: no training started and no files written; data-file checks are deferred.")
        return 0
    draft = [run.model for run in runs if run.summary["taxonomy_status"] != "final"]
    if draft:
        print("The unified fine-class taxonomy is still a draft; set its status to final after "
              f"review before training ({', '.join(draft)} not started).", file=sys.stderr)
        return 2
    for run in runs:
        print(f"Starting {run.model}; full log: {run.log_path}", flush=True)
        try:
            returncode = run_one(run)
        except KeyboardInterrupt:
            print("Baseline launcher interrupted; remaining models were not started.", file=sys.stderr)
            return 130
        except OSError as exc:
            print(f"Could not run {run.model}: {exc}; remaining models were not started.", file=sys.stderr)
            return 1
        if returncode != 0:
            print(f"{run.model} failed (exit={returncode}); inspect {run.log_path}. "
                  "Remaining models were not started.", file=sys.stderr)
            return 1
    print(f"Completed {len(runs)} baseline models; run-tag={args.run_tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
