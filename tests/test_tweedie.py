import numpy as np
import pytest
from debiasor import TweedieDebiaser

def test_no_noise(no_noise_data):
    preds, targets = no_noise_data
    tweedie = TweedieDebiaser().fit(preds, targets)
    corrected = tweedie.debiased_mean(preds)
    assert np.isclose(corrected, preds.mean())

def test_bias_reduction(noisy_data):
    cal_preds, cal_targets, preds, targets = noisy_data
    tweedie = TweedieDebiaser().fit(cal_preds, cal_targets)

    naive = preds.mean()
    corrected = tweedie.debiased_mean(preds)
    true = targets.mean()

    assert abs(corrected - true) < abs(naive - true)

def test_params_roundtrip():
    tweedie = TweedieDebiaser(delta=1e-4)
    params = tweedie.get_params()
    tweedie.set_params(**params)
    assert tweedie.get_params() == params


def test_sigma_from_separate_calibration():
    rng = np.random.default_rng(1)
    n_cal = 1000
    n_sigma = 800
    noise_cal = 0.05
    noise_sigma = 0.5

    cal_preds = rng.random(n_cal)
    cal_targets = cal_preds + rng.normal(0, noise_cal, n_cal)

    sigma_preds = rng.random(n_sigma)
    sigma_targets = sigma_preds + rng.normal(0, noise_sigma, n_sigma)

    tweedie = TweedieDebiaser().fit(
        cal_preds,
        cal_targets,
        cal_predictions_sigma=sigma_preds,
        cal_targets_sigma=sigma_targets,
    )

    # sigma_ should reflect the noise level in the separate sigma calibration set
    expected_sigma = np.std(sigma_preds - sigma_targets)
    assert pytest.approx(expected_sigma, rel=1e-2) == tweedie.sigma_


def test_sigma_args_must_both_be_provided(noisy_data):
    cal_preds, cal_targets, _, _ = noisy_data
    with pytest.raises(ValueError):
        TweedieDebiaser().fit(cal_preds, cal_targets, cal_predictions_sigma=np.ones_like(cal_preds))
    with pytest.raises(ValueError):
        TweedieDebiaser().fit(cal_preds, cal_targets, cal_targets_sigma=np.ones_like(cal_targets))


def test_debiased_predictions_mean_consistency(noisy_data):
    cal_preds, cal_targets, preds, targets = noisy_data
    tweedie = TweedieDebiaser().fit(cal_preds, cal_targets)

    dp = tweedie.debiased_predictions(preds)
    dm = tweedie.debiased_mean(preds)

    # mean of per-element debiased predictions should equal debiased_mean
    assert np.isclose(dp.mean(), dm)
