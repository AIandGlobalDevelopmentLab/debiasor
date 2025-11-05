from __future__ import annotations
import numpy as np
from scipy.stats import gaussian_kde
from .base import BaseDebiaser

class TweedieCorrection(BaseDebiaser):
    """
    Tweedie's correction
    """

    def __init__(self, delta: float = 1e-5):
        self.delta = delta

    def fit(self, cal_predictions: np.ndarray, cal_targets: np.ndarray) -> "TweedieCorrection":
        self.cal_predictions_ = np.asarray(cal_predictions)
        self.cal_targets_ = np.asarray(cal_targets)

        self.sigma_ = np.std(self.cal_predictions_ - self.cal_targets_)
        self.kde_ = gaussian_kde(self.cal_predictions_)
        return self

    def _score(self, y: np.ndarray) -> np.ndarray:
        log_p_plus = self.kde_.logpdf(y + self.delta)
        log_p_minus = self.kde_.logpdf(y - self.delta)
        return (log_p_plus - log_p_minus) / (2 * self.delta)

    def debiased_mean(self, predictions: np.ndarray) -> float:
        if not hasattr(self, "sigma_"):
            raise RuntimeError("Call .fit() before .debiased_mean().")

        mean_pred = np.mean(predictions)
        scores = self._score(predictions)
        return float(mean_pred - self.sigma_**2 * np.mean(scores))
