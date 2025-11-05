import numpy as np
import pytest

@pytest.fixture
def no_noise_data():
    # Predictions == targets
    preds = np.linspace(0, 1, 1000)
    return preds, preds.copy()

@pytest.fixture
def noisy_data():
    rng = np.random.default_rng(0)
    n_cal = 5000
    n_inf = 3000
    noise = 0.2

    cal_preds = rng.random(n_cal)
    cal_targets = cal_preds + rng.normal(0, noise, n_cal)

    preds = rng.random(n_inf)
    targets = preds + rng.normal(0, noise, n_inf)

    # Introduce population shift (bias)
    mask = targets > 0.8
    return (cal_preds, cal_targets, preds[mask], targets[mask])
