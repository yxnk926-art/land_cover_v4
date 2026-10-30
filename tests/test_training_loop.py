from pathlib import Path
import tempfile
import unittest

import torch

from common.engine import SegLoss, adjust_learning_rate, make_grad_scaler


class TrainingLoopTest(unittest.TestCase):
    def test_two_epoch_smoke_loss_lr_amp_and_resume_state(self):
        torch.manual_seed(7)
        model = torch.nn.Conv2d(3, 2, kernel_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        scaler = make_grad_scaler(enabled=False)
        criterion = SegLoss(
            ignore_index=255, dice_weight=0.5, deep_sup_weights=(1.0,)
        )
        images = torch.ones(2, 3, 4, 4)
        labels = torch.ones(2, 4, 4, dtype=torch.long)
        losses = []
        lrs = []
        for epoch in range(2):
            lr = adjust_learning_rate(
                optimizer, 0.2, epoch, max_iter=4, warmup_iters=0, power=0.9
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            lrs.append(lr)
        self.assertLess(losses[1], losses[0])
        self.assertAlmostEqual(lrs[0], 0.2)
        self.assertAlmostEqual(lrs[1], 0.2 * (1 - 1 / 4) ** 0.9)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "roundtrip.pth"
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict()}, path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            restored = torch.nn.Conv2d(3, 2, kernel_size=1)
            restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.2)
            restored.load_state_dict(payload["model"])
            restored_optimizer.load_state_dict(payload["optimizer"])
            for expected, actual in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(expected, actual))

    def test_dice_known_value(self):
        criterion = SegLoss(
            ignore_index=255, dice_weight=1.0, deep_sup_weights=(1.0,)
        )
        logits = torch.zeros(1, 2, 1, 2)
        target = torch.tensor([[[0, 1]]])
        # Uniform logits: CE=ln(2), Dice loss=0.5 for the two present classes.
        self.assertAlmostEqual(
            float(criterion(logits, target)), float(torch.log(torch.tensor(2.0))) + 0.5,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
