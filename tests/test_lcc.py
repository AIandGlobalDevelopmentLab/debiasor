import numpy as np
from debiasor import LccDebiaser

def test_no_noise(no_noise_data):
    preds, targets = no_noise_data
    lcc = LccDebiaser().fit(preds, targets)
    corrected = lcc.debiased_mean(preds)
    assert np.isclose(corrected, preds.mean())

def test_bias_reduction(noisy_data):
    cal_preds, cal_targets, preds, targets = noisy_data
    lcc = LccDebiaser().fit(cal_preds, cal_targets)

    naive = preds.mean()
    corrected = lcc.debiased_mean(preds)
    true = targets.mean()

    assert abs(corrected - true) < abs(naive - true)

def test_params_roundtrip():
    lcc = LccDebiaser()
    params = lcc.get_params()
    lcc.set_params(**params)
    assert lcc.get_params() == params


def test_debiased_predictions_mean_consistency(noisy_data):
    cal_preds, cal_targets, preds, targets = noisy_data
    lcc = LccDebiaser().fit(cal_preds, cal_targets)

    dp = lcc.debiased_predictions(preds)
    dm = lcc.debiased_mean(preds)

    # mean of the per-element debiased predictions should equal debiased_mean
    assert np.isclose(dp.mean(), dm)
