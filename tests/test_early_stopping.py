# -*- coding: utf-8 -*-
"""Checks for the reusable early-stopping state machine."""
import unittest

from common.early_stopping import EarlyStopping


class EarlyStoppingTests(unittest.TestCase):
    def test_tiny_raw_best_does_not_reset_patience(self):
        tracker = EarlyStopping(patience=3, min_delta=1e-3)
        self.assertEqual(tracker.step(0.6212), (False, True))
        self.assertEqual(tracker.step(0.6213), (False, False))
        self.assertEqual(tracker.bad_evals, 1)

    def test_cumulative_gain_resets_patience(self):
        tracker = EarlyStopping(patience=3, min_delta=1e-3)
        tracker.step(0.6212)
        tracker.step(0.6217)
        should_stop, improved = tracker.step(0.6223)
        self.assertFalse(should_stop)
        self.assertTrue(improved)
        self.assertEqual(tracker.bad_evals, 0)
        self.assertAlmostEqual(tracker.reference_score, 0.6223)

    def test_stops_on_exact_patience_boundary(self):
        tracker = EarlyStopping(patience=2, min_delta=1e-3)
        tracker.step(0.6)
        self.assertFalse(tracker.step(0.6001)[0])
        self.assertTrue(tracker.step(0.6002)[0])

    def test_state_round_trip(self):
        tracker = EarlyStopping(patience=4, min_delta=1e-3)
        tracker.step(0.6)
        tracker.step(0.6001)
        restored = EarlyStopping(patience=4, min_delta=1e-3)
        self.assertTrue(restored.load_state_dict(tracker.state_dict()))
        self.assertEqual(restored.bad_evals, 1)
        self.assertAlmostEqual(restored.reference_score, 0.6)

    def test_changed_delta_rejects_old_counter(self):
        tracker = EarlyStopping(patience=4, min_delta=1e-3)
        tracker.step(0.6)
        changed = EarlyStopping(patience=4, min_delta=5e-4)
        self.assertFalse(changed.load_state_dict(tracker.state_dict()))


if __name__ == "__main__":
    unittest.main()

