"""Training entry point: fingerprints, checkpoints, resume guards, run tags."""
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from common import train


class TrainingProtocolTest(unittest.TestCase):
    def test_protocol_fingerprint_changes_on_critical_drift(self):
        first = SimpleNamespace(
            PROTOCOL_ID="v1", MODEL_NAME="unet", CROP_SIZE=768,
            SCALE_RANGE=(0.75, 1.5), LR=5e-4, AMP=True,
        )
        second = SimpleNamespace(**{**vars(first), "SCALE_RANGE": (0.8, 1.25)})
        fp1 = train.protocol_fingerprint(train.protocol_snapshot(first))
        self.assertNotEqual(fp1, train.protocol_fingerprint(train.protocol_snapshot(second)))
        second = SimpleNamespace(**{**vars(first), "AMP": False})
        self.assertNotEqual(fp1, train.protocol_fingerprint(train.protocol_snapshot(second)))
        first.EVAL_AMP = False
        second = SimpleNamespace(**{**vars(first), "EVAL_AMP": True})
        self.assertNotEqual(train.protocol_fingerprint(train.protocol_snapshot(first)),
                            train.protocol_fingerprint(train.protocol_snapshot(second)))

    def test_best_checkpoint_carries_selection_provenance(self):
        model = torch.nn.Conv2d(3, 6, kernel_size=1)
        cfg = SimpleNamespace(PROTOCOL_ID="unified-fine-v4", MODEL_NAME="tiny", NUM_CLASSES=6,
                              UNIFIED_TAXONOMY_SHA256="a" * 64)
        protocol = train.protocol_snapshot(cfg)
        resolved = {"protocol": protocol, "protocol_fingerprint": train.protocol_fingerprint(protocol)}
        self.assertIn("UNIFIED_TAXONOMY_SHA256", protocol)
        scores = {"miou": 0.625, "datasets": {"a": {"miou": 0.5}, "b": {"miou": 0.75}}}
        payload = train.best_checkpoint_payload(model, epoch=4, scores=scores, module=cfg,
                                                resolved=resolved)
        self.assertEqual(payload["kind"], "best_model")
        self.assertEqual(payload["source_selection_metric"], "miou")
        self.assertEqual(payload["source_selection_value"], 0.625)
        self.assertEqual(payload["source_selection_by_dataset"], {"a": 0.5, "b": 0.75})
        self.assertEqual(payload["config"]["UNIFIED_TAXONOMY_SHA256"], "a" * 64)

        with tempfile.TemporaryDirectory() as temp:
            rich_path, raw_path = Path(temp) / "best-rich.pth", Path(temp) / "best-raw.pth"
            torch.save(payload, rich_path)
            torch.save(model.state_dict(), raw_path)
            for path in (rich_path, raw_path):
                restored = torch.nn.Conv2d(3, 6, kernel_size=1)
                matched, skipped, missing = train.load_initial_weights(restored, path)
                self.assertEqual((matched, skipped, missing), (len(model.state_dict()), 0, 0))
                for expected, actual in zip(model.parameters(), restored.parameters()):
                    self.assertTrue(torch.equal(expected, actual))

    def test_training_rejects_a_target_role(self):
        with self.assertRaisesRegex(ValueError, "DATASET_ROLE"):
            train.validate_training_domain(SimpleNamespace(DATASET_ROLE="target"))

    def test_resume_is_fail_closed_and_rng_roundtrip(self):
        cfg = SimpleNamespace(ALLOW_UNVERIFIED_RESUME=False, ALLOW_CONFIG_MISMATCH_ON_RESUME=False)
        with self.assertRaisesRegex(RuntimeError, "仅供诊断"):
            train.validate_resume_checkpoint({"kind": "numerical_failure"}, "a" * 64, cfg)
        with self.assertRaisesRegex(RuntimeError, "protocol_fingerprint"):
            train.validate_resume_checkpoint({}, "a" * 64, cfg)
        with self.assertRaisesRegex(RuntimeError, "协议不一致"):
            train.validate_resume_checkpoint(
                {"protocol_fingerprint": "b" * 64, "rng_state": {}}, "a" * 64, cfg)
        with self.assertRaisesRegex(RuntimeError, "rng_state"):
            train.validate_resume_checkpoint({"protocol_fingerprint": "a" * 64}, "a" * 64, cfg)

        train.set_seed(123)
        generator = torch.Generator().manual_seed(456)
        loader = SimpleNamespace(generator=generator)
        state = train.capture_rng_state(loader, None)
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "rng-state.pth"
            torch.save({"rng_state": state}, checkpoint)
            state = train._load_checkpoint_file(checkpoint)["rng_state"]
        expected = (random.random(), float(np.random.rand()), torch.rand(1),
                    torch.rand(1, generator=generator))
        train.restore_rng_state(state, loader, None)
        actual = (random.random(), float(np.random.rand()), torch.rand(1),
                  torch.rand(1, generator=generator))
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])
        self.assertTrue(torch.equal(expected[2], actual[2]))
        self.assertTrue(torch.equal(expected[3], actual[3]))

    def test_run_tag_isolates_outputs_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            config = SimpleNamespace(
                OUTPUT_ROOT=str(root / "outputs"), CHECKPOINT_ROOT=str(root / "checkpoints"),
                MODEL_SLUG="unet", EXP_NAME="unified_unet", SEED=1,
                RESUME=False, INIT_CHECKPOINT=None,
            )
            args = SimpleNamespace(run_tag="unified_s42", seed=42)
            train.apply_run_overrides(config, args)
            self.assertEqual(config.SEED, 42)
            self.assertEqual(Path(config.OUT_DIR), root / "outputs/baselines/unified_s42/unet")
            Path(config.OUT_DIR).mkdir(parents=True)
            evidence = Path(config.OUT_DIR) / "train_log.txt"
            evidence.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "拒绝覆盖"):
                train.apply_run_overrides(config, args)
            self.assertEqual(evidence.read_text(), "keep")
            with self.assertRaisesRegex(ValueError, "run-tag"):
                train.apply_run_overrides(config, SimpleNamespace(run_tag="../escape", seed=42))
            diagnostic = root / "failure.pth"
            torch.save({"kind": "numerical_failure", "model": {}}, diagnostic)
            with self.assertRaisesRegex(ValueError, "仅供诊断"):
                train.load_initial_weights(torch.nn.Conv2d(3, 6, 1), diagnostic)


if __name__ == "__main__":
    unittest.main()
