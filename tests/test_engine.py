import unittest

import numpy as np
import torch

from common.engine import ConfusionMatrix, SegLoss, evaluate


class _SevenLogitModel(torch.nn.Module):
    def forward(self, image):
        return torch.zeros(
            image.shape[0], 7, image.shape[2], image.shape[3],
            device=image.device,
        )


class CommonV3EngineTests(unittest.TestCase):
    def test_six_logit_ce_supervises_class_zero_and_ignores_255(self):
        logits = torch.zeros(1, 6, 1, 2, requires_grad=True)
        target = torch.tensor([[[0, 255]]])
        loss = SegLoss(
            ignore_index=255, dice_weight=0.0, deep_sup_weights=(1.0,)
        )(logits, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertLess(float(logits.grad[0, 0, 0, 0]), 0.0)
        self.assertTrue(torch.all(logits.grad[0, 1:, 0, 0] > 0))
        self.assertTrue(torch.equal(
            logits.grad[0, :, 0, 1], torch.zeros(6, dtype=logits.grad.dtype)
        ))

    def test_all_ignore_loss_is_differentiable_zero(self):
        logits = torch.randn(1, 6, 2, 2, requires_grad=True)
        target = torch.full((1, 2, 2), 255, dtype=torch.long)
        loss = SegLoss(
            ignore_index=255, dice_weight=0.0, deep_sup_weights=(1.0,)
        )(logits, target)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_external_ignore_keeps_exactly_six_metric_classes(self):
        matrix = ConfusionMatrix(num_classes=6, ignore_index=255)
        matrix.update(
            np.array([[0, 1, 5], [2, 3, 4]]),
            np.array([[0, 1, 255], [2, 3, 4]]),
        )
        scores = matrix.scores()
        self.assertEqual(scores["n_keep_classes"], 6)
        self.assertAlmostEqual(scores["miou_fixed"], 5 / 6)
        self.assertEqual(int(scores["per_class_support"].sum()), 5)

    def test_evaluate_rejects_old_seven_logit_head(self):
        image = torch.zeros(1, 3, 2, 2)
        target = torch.zeros(1, 2, 2, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, r"\[B,6,H,W\]"):
            evaluate(
                _SevenLogitModel(), [(image, target)], torch.device("cpu"),
                num_classes=6, ignore_index=255, amp=False,
            )


if __name__ == "__main__":
    unittest.main()
