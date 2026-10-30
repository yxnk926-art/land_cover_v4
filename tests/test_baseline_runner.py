from dataclasses import replace
import io
import os
from pathlib import Path
import tempfile
from unittest import mock
import unittest

from common import unified_protocol as up
from tests.test_unified_protocol import SYNTHETIC, UnifiedCopies
from tools import run_baselines as runner


class BaselineRunnerTest(UnifiedCopies):
    def setUp(self):
        super().setUp()
        os.environ[up.DATASETS_ENV] = ",".join(SYNTHETIC)

    def make_runs(self, root):
        plans = runner.build_runs(["mobilenetv3_lraspp_in1k", "pidnet_s_in1k"], "test_s42", 42)
        train_list, val_list = root / "train.txt", root / "val.txt"
        train_list.write_text("train_sample\n", encoding="utf-8")
        val_list.write_text("val_sample\n", encoding="utf-8")
        return [replace(plan, output_dir=root / "outputs" / plan.model,
                        checkpoint_dir=root / "checkpoints" / plan.model,
                        log_path=root / "launcher" / f"{plan.model}.log",
                        lists=(train_list, val_list)) for plan in plans]

    def test_default_queue_is_all_nine_models_fastest_first(self):
        args = runner.build_parser().parse_args(["--run-tag", "t"])
        self.assertIsNone(args.models)
        self.assertEqual(len(up.MODEL_CONFIGS), 9)
        self.assertEqual(list(up.MODEL_CONFIGS)[:2], ["ddrnet", "mobilenetv3_lraspp_in1k"])
        self.assertEqual(list(up.MODEL_CONFIGS)[-1], "swin_upernet")
        with mock.patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            runner.build_parser().parse_args([])  # a run tag is required

    def test_all_plans_preserve_leaf_settings_and_isolate_outputs(self):
        configs = [up.import_unified_config(name, list(SYNTHETIC))
                   for name in up.MODEL_CONFIGS.values()]
        before = [(cfg.OUT_DIR, cfg.CKPT_DIR, cfg.EXP_NAME, cfg.SEED) for cfg in configs]
        plans = runner.build_runs(list(up.MODEL_CONFIGS), "baseline_fp32_s123", 123)
        self.assertEqual(len(plans), len(up.MODEL_CONFIGS))
        self.assertEqual(len({plan.output_dir for plan in plans}), len(plans))
        self.assertEqual(len({plan.checkpoint_dir for plan in plans}), len(plans))
        for plan, cfg in zip(plans, configs):
            self.assertEqual(plan.output_dir, Path(cfg.OUTPUT_ROOT) / "baselines" /
                             "baseline_fp32_s123" / cfg.MODEL_SLUG)
            self.assertEqual(plan.checkpoint_dir, Path(cfg.CHECKPOINT_ROOT) / "baselines" /
                             "baseline_fp32_s123" / cfg.MODEL_SLUG)
            self.assertEqual(plan.summary["lr"], cfg.LR)
            self.assertEqual(plan.summary["weight_decay"], cfg.WEIGHT_DECAY)
            self.assertEqual(plan.summary["poly_power"], cfg.POLY_POWER)
            self.assertEqual(plan.command[-4:], ("--run-tag", "baseline_fp32_s123", "--seed", "123"))
            self.assertFalse(plan.summary["eval_amp"])
            self.assertEqual(plan.lists[0].name, "train.txt")
        self.assertEqual(before, [(cfg.OUT_DIR, cfg.CKPT_DIR, cfg.EXP_NAME, cfg.SEED)
                                  for cfg in configs])

    def test_later_occupied_model_prevents_any_training(self):
        with tempfile.TemporaryDirectory() as directory:
            plans = self.make_runs(Path(directory))
            plans[1].checkpoint_dir.mkdir(parents=True)
            (plans[1].checkpoint_dir / "evidence.pth").write_bytes(b"keep")
            with mock.patch.object(runner, "build_runs", return_value=plans), \
                    mock.patch.object(runner, "run_one") as run_one, \
                    mock.patch("sys.stderr", new=io.StringIO()):
                self.assertEqual(runner.main(["--run-tag", "test_s42"]), 2)
                run_one.assert_not_called()
            self.assertFalse(plans[0].output_dir.exists())

    def test_dry_run_never_starts_process_or_creates_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plans = self.make_runs(root)
            plans[0].lists[0].unlink()
            with mock.patch.object(runner, "build_runs", return_value=plans), \
                    mock.patch.object(runner, "run_one") as run_one, \
                    mock.patch("sys.stdout", new=io.StringIO()) as output:
                self.assertEqual(runner.main(["--run-tag", "test_s42", "--dry-run"]), 0)
                run_one.assert_not_called()
                self.assertIn("--run-tag", output.getvalue())
            self.assertFalse((root / "launcher").exists())

    def test_failure_stops_remaining_models(self):
        with tempfile.TemporaryDirectory() as directory:
            plans = self.make_runs(Path(directory))
            with mock.patch.object(runner, "build_runs", return_value=plans), \
                    mock.patch.object(runner, "run_one", return_value=1) as run_one, \
                    mock.patch("sys.stdout", new=io.StringIO()), \
                    mock.patch("sys.stderr", new=io.StringIO()):
                self.assertEqual(runner.main(["--run-tag", "test_s42"]), 1)
                run_one.assert_called_once_with(plans[0])

    def test_child_error_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.make_runs(Path(directory))[0]
            proc = mock.Mock()
            proc.stdout = io.StringIO("epoch 48\nFloatingPointError: bad logits\n")
            proc.wait.return_value = 1
            with mock.patch.object(runner.subprocess, "Popen", return_value=proc) as popen, \
                    mock.patch("sys.stdout", new=io.StringIO()):
                self.assertEqual(runner.run_one(plan), 1)
                self.assertEqual(popen.call_args.kwargs["stderr"], runner.subprocess.STDOUT)
            log = plan.log_path.read_text(encoding="utf-8")
            self.assertIn("FloatingPointError: bad logits", log)
            self.assertIn("launcher_returncode=1", log)
            with self.assertRaisesRegex(ValueError, "log already exists"):
                runner.preflight([plan])

    def test_interruption_stops_child_group_and_keeps_log(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = self.make_runs(Path(directory))[0]
            proc = mock.Mock()
            proc.stdout = mock.MagicMock()
            proc.stdout.__iter__.side_effect = KeyboardInterrupt()
            with mock.patch.object(runner.subprocess, "Popen", return_value=proc), \
                    mock.patch.object(runner, "_stop_process_group") as stop:
                with self.assertRaises(KeyboardInterrupt):
                    runner.run_one(plan)
                stop.assert_called_once_with(proc)
            self.assertIn("launcher_interrupted=true", plan.log_path.read_text(encoding="utf-8"))

    def test_source_lists_are_required_before_real_run(self):
        with tempfile.TemporaryDirectory() as directory:
            plans = self.make_runs(Path(directory))
            plans[0].lists[1].unlink()
            with self.assertRaisesRegex(ValueError, "source list is missing"):
                runner.preflight(plans)

    def test_reject_duplicate_models_and_invalid_run_tags(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            runner.build_runs(["ddrnet", "ddrnet"], "test_s42", 42)
        for tag in ("..", "../old", "a/b", "a\\b", ""):
            with self.subTest(tag=tag), self.assertRaises(runner.argparse.ArgumentTypeError):
                runner.run_tag(tag)
        for seed in ("-1", str(2**32), "x"):
            with self.subTest(seed=seed), self.assertRaises(runner.argparse.ArgumentTypeError):
                runner.train_seed(seed)


if __name__ == "__main__":
    unittest.main()
