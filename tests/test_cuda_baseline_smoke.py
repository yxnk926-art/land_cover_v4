"""Small CUDA integration check, not a convergence or full-run stability test."""
import gc
import unittest

import torch

from common import unified_protocol as up
from common.engine import (autocast_ctx, build_criterion, build_optimizer,
                           evaluate, make_grad_scaler)
from common.models import build_model
from tests.test_unified_protocol import UnifiedCopies


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaBaselineSmokeTests(UnifiedCopies):
    def test_all_models_amp_update_then_fp32_whole_image_validation(self):
        device = torch.device("cuda")
        for name in up.MODEL_CONFIGS:
            with self.subTest(model=name):
                torch.manual_seed(42)
                cfg = self.unified(name)
                self.assertFalse(cfg.EVAL_AMP)
                model = build_model(cfg).to(device).train()
                criterion = build_criterion(cfg).to(device)
                optimizer = build_optimizer(model, cfg)
                scaler = make_grad_scaler(enabled=cfg.AMP)
                images = torch.randn(2, 3, 128, 128, device=device)
                labels = torch.randint(0, len(cfg.SET_MEMBERSHIP), (2, 128, 128), device=device)
                labels[:, :2, :2] = cfg.IGNORE_INDEX
                optimizer.zero_grad(set_to_none=True)
                with autocast_ctx("cuda", enabled=cfg.AMP):
                    loss = criterion(model(images), labels)
                self.assertTrue(torch.isfinite(loss).item())
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD)
                scaler.step(optimizer)
                scaler.update()
                val = [(torch.randn(1, 3, 128, 160),
                        torch.randint(0, cfg.NUM_CLASSES, (1, 128, 160)),
                        ["synthetic-validation"])]
                scores = evaluate(model, val, device, cfg.NUM_CLASSES,
                                  ignore_index=cfg.IGNORE_INDEX, amp=cfg.EVAL_AMP)
                self.assertTrue(0.0 <= scores["miou"] <= 1.0)
                self.assertEqual(int(scores["per_class_support"].sum()), 128 * 160)
                del model, criterion, optimizer, scaler, loss, images, labels
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
