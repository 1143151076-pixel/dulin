import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from credit_risk.data import FEATURES, engineer, validate_features
from credit_risk.model import pipeline, partitions, select_threshold, metrics


def sample(n=120):
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(100, 20, (n, len(FEATURES))), columns=FEATURES)
    x['LIMIT_BAL'] = 10000
    return x


def test_duplicate_groups_never_cross_splits():
    x = pd.concat([sample(), sample()], ignore_index=True)
    tr, va, te, groups = partitions(x)
    assert len(tr) + len(va) + len(te) == len(x)
    assert not set(groups[tr]) & set(groups[va])
    assert not set(groups[tr]) & set(groups[te])
    assert not set(groups[va]) & set(groups[te])


@pytest.mark.parametrize('problem', ['target', 'missing', 'inf', 'zero_limit', 'empty', 'duplicate', 'text'])
def test_reject_bad_input(problem):
    x = sample()
    if problem == 'target': x['default payment next month'] = 0
    if problem == 'missing': x = x.drop(columns='PAY_0')
    if problem == 'inf': x.loc[0, 'PAY_0'] = np.inf
    if problem == 'zero_limit': x.loc[0, 'LIMIT_BAL'] = 0
    if problem == 'empty': x = x.iloc[:0]
    if problem == 'duplicate': x = pd.concat([x, x[['PAY_0']]], axis=1)
    if problem == 'text': x['PAY_0'] = 'bad'
    with pytest.raises((ValueError, TypeError)): validate_features(x)


def test_engineering_does_not_mutate_and_is_order_invariant():
    x = sample()
    copy = x.copy(deep=True)
    a = engineer(x)
    pd.testing.assert_frame_equal(x, copy)
    pd.testing.assert_frame_equal(a, engineer(x[x.columns[::-1]]))
    assert a.shape[1] == len(FEATURES) + 6
    assert np.allclose(a['bill_to_limit'], x['BILL_AMT1'] / x['LIMIT_BAL'])


def test_imputation_is_training_only_and_model_roundtrip(tmp_path):
    x = sample()
    y = np.tile([0, 1], 60)
    x.loc[0, 'PAY_0'] = np.nan
    m = pipeline(LogisticRegression(max_iter=1000)).fit(x, y)
    before = m['imputer'].statistics_.copy()
    holdout = sample(5); holdout['BILL_AMT1'] = 1e12
    p = m.predict_proba(holdout)
    assert np.isfinite(p).all()
    np.testing.assert_array_equal(before, m['imputer'].statistics_)
    path = tmp_path / 'model.joblib'; joblib.dump(m, path)
    np.testing.assert_allclose(p, joblib.load(path).predict_proba(holdout))


def test_threshold_cost_and_metric_counts():
    y = np.array([0, 0, 1, 1]); p = np.array([.1, .2, .3, .4])
    t = select_threshold(y, p)
    assert .2 < t <= .3
    assert metrics(y, p, t)['confusion_matrix'] == [[2, 0], [0, 2]]
