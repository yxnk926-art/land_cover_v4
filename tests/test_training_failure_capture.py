"""Exercise failure capture through the real training entry point."""
import io
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from common import train
from common.numerical_diagnostics import NumericalValidationError


class FiniteTrainBadEval(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Conv2d(3, 6, kernel_size=1)
        with torch.no_grad():
            self.head.weight.fill_(0.1)
            self.head.bias.zero_()

    def forward(self, image):
        logits = self.head(image)
        return logits if self.training else logits * float("nan")


class TrainingFailureCaptureTest(unittest.TestCase):
    def config(self, directory):
        return SimpleNamespace(
            __name__="configs.test_leaf", __file__=str(directory / "leaf.py"),
            PROTOCOL_ID="test", MODEL_NAME="tiny", NUM_CLASSES=6, IGNORE_INDEX=255,
            UNIFIED_DATASETS={"d": {"num_classes": 6, "eval_window": None, "eval_stride": None,
                                    "class_names": list("abcdef")}},
            LABEL_OFFSETS={"d": 0}, SET_MEMBERSHIP=torch.eye(6, dtype=torch.int64).tolist(),
            OUT_DIR=str(directory / "outputs"), CKPT_DIR=str(directory / "checkpoints"),
            EPOCHS=1, SCHEDULE_EPOCHS=1, EVAL_INTERVAL=1, LOG_INTERVAL=1, SEED=1,
            LR=1e-3, WEIGHT_DECAY=0.01, WARMUP_ITERS=0, POLY_POWER=0.9, CLIP_GRAD=1.0,
            DICE_WEIGHT=0.0, DEEP_SUP_WEIGHTS=(1.0,),
            AMP=False, EVAL_AMP=False, BATCH_SIZE=1, CROP_SIZE=4,
            EARLY_STOP_PATIENCE=0, INIT_CHECKPOINT=None, RESUME=False,
        )

    def run_failed_training(self, config, model, diagnostic_error=None):
        images = torch.ones(1, 3, 4, 4)
        labels = torch.zeros(1, 4, 4, dtype=torch.long)
        loaders = ([(images, labels, ["train_tile"])], {"d": [(images, labels, ["bad_val_tile"])]})
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(train, "cfg", config))
            stack.enter_context(mock.patch.object(train, "cli_args", SimpleNamespace(run_tag=None, seed=None)))
            stack.enter_context(mock.patch.object(train, "validate_training_domain"))
            stack.enter_context(mock.patch.object(train, "configure_reproducibility", return_value={
                "deterministic": False, "cudnn_benchmark": False, "warn_only": False,
            }))
            stack.enter_context(mock.patch.object(train, "build_unified_dataloaders", return_value=loaders))
            stack.enter_context(mock.patch.object(train, "build_model", return_value=model))
            stack.enter_context(mock.patch("torch.cuda.is_available", return_value=False))
            if diagnostic_error is not None:
                stack.enter_context(mock.patch.object(train, "save_validation_failure", side_effect=diagnostic_error))
            stack.enter_context(redirect_stdout(io.StringIO()))
            with self.assertRaisesRegex(NumericalValidationError, "evaluation logits contain non-finite") as raised:
                train.main()
        self.assertEqual(raised.exception.sample_names, ["bad_val_tile"])
        self.assertEqual(raised.exception.stage, "logits")
        self.assertFalse((Path(config.CKPT_DIR) / "best_model.pth").exists())
        self.assertFalse((Path(config.CKPT_DIR) / "checkpoint_last.pth").exists())

    def test_real_training_saves_exact_post_update_failure_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary))
            model = FiniteTrainBadEval()
            before = {name: value.clone() for name, value in model.state_dict().items()}
            self.run_failed_training(config, model)
            snapshots = list(Path(config.OUT_DIR).glob("diagnostics/epoch_0000/numerical_failure_*/snapshot.pth"))
            self.assertEqual(len(snapshots), 1)
            payload = torch.load(snapshots[0], map_location="cpu", weights_only=True)
            self.assertEqual(payload["kind"], "numerical_failure")
            self.assertTrue(payload["diagnosis_only"])
            self.assertEqual(payload["context"]["epoch"], 0)
            self.assertIn("protocol_fingerprint", payload["context"])
            self.assertIn("scaler", payload)
            self.assertTrue(payload["optimizer"]["state"])
            self.assertTrue(any(not torch.equal(before[name], value) for name, value in model.state_dict().items()))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(payload["model"][name], value.cpu()))
            self.assertTrue((snapshots[0].parent / "report.json").is_file())

    def test_diagnostic_write_error_preserves_original_validation_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(Path(temporary))
            self.run_failed_training(config, FiniteTrainBadEval(), diagnostic_error=OSError("disk unavailable"))
            log = (Path(config.OUT_DIR) / "train_log.txt").read_text(encoding="utf-8")
            self.assertIn("OSError: disk unavailable", log)


if __name__ == "__main__":
    unittest.main()
