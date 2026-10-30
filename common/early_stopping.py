# -*- coding: utf-8 -*-
"""Validation-based early-stopping state with checkpoint round-trip support.

The raw best score used for ``best_model.pth`` intentionally remains separate
from this tracker.  A tiny raw improvement can therefore still be preserved
without resetting the early-stopping patience counter.
"""
import math


class EarlyStopping:
    """Stop after ``patience`` validations without a meaningful improvement.

    ``min_delta`` is measured against the last meaningful reference score, not
    the immediately preceding validation.  Several small gains may therefore
    accumulate and eventually reset the counter once their total exceeds the
    configured threshold.
    """

    def __init__(self, patience, min_delta=0.0):
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError("EARLY_STOP_PATIENCE 必须是正整数或 None/0（关闭）")
        if not math.isfinite(float(min_delta)) or float(min_delta) < 0:
            raise ValueError("EARLY_STOP_MIN_DELTA 必须是有限的非负数")
        self.patience = patience
        self.min_delta = float(min_delta)
        self.reference_score = None
        self.bad_evals = 0

    @property
    def should_stop(self):
        return self.bad_evals >= self.patience

    def reset(self, reference_score=None):
        if reference_score is not None:
            reference_score = float(reference_score)
            if not math.isfinite(reference_score):
                raise ValueError("early-stopping reference score must be finite")
        self.reference_score = reference_score
        self.bad_evals = 0

    def step(self, score):
        """Update the tracker and return ``(should_stop, meaningful_gain)``."""
        score = float(score)
        if not math.isfinite(score):
            self.bad_evals += 1
            return self.should_stop, False

        if self.reference_score is None:
            self.reference_score = score
            self.bad_evals = 0
            return False, True

        if score > self.reference_score + self.min_delta:
            self.reference_score = score
            self.bad_evals = 0
            return False, True

        self.bad_evals += 1
        return self.should_stop, False

    def state_dict(self):
        return {
            "reference_score": self.reference_score,
            "bad_evals": self.bad_evals,
            "min_delta": self.min_delta,
            "patience": self.patience,
        }

    def load_state_dict(self, state):
        """Restore dynamic state.

        Returns ``False`` when the saved ``min_delta`` differs from the active
        config.  In that case the caller should start a fresh patience window,
        because the old counter has a different meaning.
        """
        if not isinstance(state, dict):
            return False
        saved_delta = float(state.get("min_delta", self.min_delta))
        if not math.isclose(saved_delta, self.min_delta, rel_tol=0.0, abs_tol=1e-15):
            return False

        reference = state.get("reference_score")
        if reference is not None:
            reference = float(reference)
            if not math.isfinite(reference):
                return False
        bad_evals = int(state.get("bad_evals", 0))
        if bad_evals < 0:
            return False

        self.reference_score = reference
        self.bad_evals = bad_evals
        return True
