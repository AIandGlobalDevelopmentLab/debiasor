from __future__ import annotations
import numpy as np
from sklearn.linear_model import LinearRegression
from .base import BaseDebiaser

class LinearCalibrationCorrection(BaseDebiaser):
    """
    Linear Calibration Correction (LCC)
    """

    def __init__(self):
        pass

    def fit(self, cal_predictions: np.ndarray, cal_targets: np.ndarray) -> "LinearCalibrationCorrection":
        cal_predictions_ = np.asarray(cal_predictions)
        cal_targets_ = np.asarray(cal_targets)

        model = LinearRegression()
        model.fit(cal_targets_.reshape(-1, 1), cal_predictions_)

        self.intercept_ = model.intercept_
        self.slope_ = model.coef_[0]
        return self

    def debiased_mean(self, predictions: np.ndarray) -> float:
        if not hasattr(self, "slope_"):
            raise RuntimeError("Call .fit() before .debiased_mean().")

        mean_pred = np.mean(predictions)
        return float((mean_pred - self.intercept_) / self.slope_)
