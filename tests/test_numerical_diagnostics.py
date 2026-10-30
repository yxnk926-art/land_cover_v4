import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn

from common.engine import evaluate
from common.numerical_diagnostics import (
    NumericalValidationError, inspect_model_state, replay_validation_failure,
    save_validation_failure,
)


class BadActivation(nn.Module):
    def forward(self, image):
        return image * float("nan")


class TestNumericalDiagnostics(unittest.TestCase):
    def setUp(self):
        self.images = torch.ones(1, 2, 2, 2)
        self.labels = torch.zeros(1, 2, 2, dtype=torch.long)
        self.loader = [(self.images, self.labels, ["tile_17"])]

    def test_finite_evaluation_still_returns_metrics(self):
        scores = evaluate(nn.Identity(), self.loader, torch.device("cpu"), 2, ignore_index=255)
        self.assertEqual(scores["miou"], 1.0)

    def test_nonfinite_input_captures_named_batch_without_running_model(self):
        self.images[0, 0, 0, 0] = float("inf")
        model = mock.Mock()
        with self.assertRaises(NumericalValidationError) as raised:
            evaluate(model, self.loader, torch.device("cpu"), 2)
        error = raised.exception
        self.assertEqual(error.stage, "input")
        self.assertEqual(error.sample_names, ["tile_17"])
        self.assertEqual(error.batch_index, 0)
        model.assert_not_called()
        self.images.zero_()
        self.assertTrue(torch.isinf(error.images[0, 0, 0, 0]))

    def test_nonfinite_logits_never_become_metrics(self):
        with self.assertRaisesRegex(NumericalValidationError, "evaluation logits contain non-finite") as raised:
            evaluate(BadActivation(), self.loader, torch.device("cpu"), 2)
        self.assertEqual(raised.exception.stage, "logits")

    def test_resize_nonfinite_is_guarded(self):
        large_labels = torch.zeros(1, 4, 4, dtype=torch.long)
        with mock.patch("common.engine.F.interpolate", return_value=torch.full((1, 2, 4, 4), float("nan"))):
            with self.assertRaises(NumericalValidationError) as raised:
                evaluate(nn.Identity(), [(self.images, large_labels)], torch.device("cpu"), 2)
        self.assertEqual(raised.exception.stage, "resized_logits")

    def test_bn_buffers_and_first_bad_module_are_reported(self):
        model = nn.Sequential(nn.Conv2d(2, 2, 1), nn.BatchNorm2d(2), BadActivation())
        model.train()
        model[1].eval()  # Preserve mixed module modes, not only model.training.
        modes = [module.training for module in model.modules()]
        state = {name: value.clone() for name, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        replay = replay_validation_failure(model, self.images, self.labels, amp=False)
        self.assertEqual(replay["status"], "non_finite")
        self.assertEqual(replay["first_nonfinite_module"]["name"], "2")
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, state[name]))
        model[1].running_var[0] = float("nan")
        model[1].running_var[1] = -1
        report = inspect_model_state(model)
        self.assertFalse(report["all_finite"])
        self.assertEqual(report["batchnorm"]["1"]["negative_variance_count"], 1)
        self.assertFalse(report["batchnorm"]["1"]["effective_gain"]["all_finite"])

    def test_diagnosis_snapshot_is_weights_only_safe_and_saved_before_replay(self):
        model = nn.Sequential(nn.Conv2d(2, 2, 1), BadActivation())
        optimizer = torch.optim.AdamW(model.parameters())
        with self.assertRaises(NumericalValidationError) as raised:
            evaluate(model, self.loader, torch.device("cpu"), 2)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("common.numerical_diagnostics.replay_validation_failure", side_effect=RuntimeError("diagnostic failed")):
                path = save_validation_failure(model, raised.exception, tmp,
                                               context={"epoch": 48, "protocol": "test"}, optimizer=optimizer)
            report = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(report["diagnostic_error"]["message"], "diagnostic failed")
            payload = torch.load(path.parent / "snapshot.pth", map_location="cpu", weights_only=True)
            self.assertEqual(payload["kind"], "numerical_failure")
            self.assertTrue(payload["diagnosis_only"])
            self.assertEqual(payload["context"]["epoch"], 48)
            self.assertIn("optimizer", payload)
            self.assertTrue(torch.equal(payload["images"], self.images))
            self.assertNotIn("scores", payload)
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(payload["model"][name], value.cpu()))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA FP16 autocast")
    def test_cuda_amp_overflow_and_fp32_replay_have_distinct_results(self):
        model = nn.Conv2d(2, 2, 1, bias=False).cuda()
        with torch.no_grad():
            model.weight.fill_(40000.0)
        with self.assertRaises(NumericalValidationError) as raised:
            evaluate(model, self.loader, torch.device("cuda"), 2, amp=True)
        self.assertTrue(raised.exception.amp)
        amp = replay_validation_failure(model, self.images, self.labels, amp=True)
        fp32 = replay_validation_failure(model, self.images, self.labels, amp=False)
        self.assertEqual(amp["status"], "non_finite")
        self.assertEqual(fp32["status"], "finite")
        self.assertEqual(amp["first_nonfinite_module"]["type"], "Conv2d")
        scores = evaluate(model, self.loader, torch.device("cuda"), 2, ignore_index=255, amp=False)
        self.assertEqual(scores["miou"], 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_validation_failure(model, raised.exception, tmp, context={"epoch": 48})
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(report["replays"]["amp"]["status"], "non_finite")
            self.assertEqual(report["replays"]["fp32"]["status"], "finite")


if __name__ == "__main__":
    unittest.main()
