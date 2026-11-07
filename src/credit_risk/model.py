"""Training-only preprocessing, grouped partitions and validation-only decisions."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             precision_score, recall_score, confusion_matrix)
from .data import engineer

SEED = 42

def partitions(x):
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy()
    first = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED)
    dev, test = next(first.split(x, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=SEED + 1)
    tr, va = next(second.split(x.iloc[dev], groups=groups[dev]))
    return dev[tr], dev[va], test, groups

def pipeline(estimator):
    return Pipeline([('features', FunctionTransformer(engineer)),
                     ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                     ('scale', StandardScaler()), ('model', estimator)])

def candidates(x_train, y_train, groups):
    booster = lambda: pipeline(HistGradientBoostingClassifier(
        max_iter=160, learning_rate=.06, max_leaf_nodes=15, l2_regularization=5,
        early_stopping=False, random_state=SEED))
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
                 .split(x_train, y_train, groups))
    return {
        'prior_baseline': pipeline(DummyClassifier(strategy='prior')),
        'logistic_regression': pipeline(LogisticRegression(C=1, max_iter=2500, random_state=SEED)),
        'gradient_boosting': booster(),
        'calibrated_gradient_boosting': CalibratedClassifierCV(booster(), method='sigmoid', cv=folds),
    }

def metrics(y, p, threshold=.5):
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'roc_auc': float(roc_auc_score(y, p)),
            'average_precision': float(average_precision_score(y, p)),
            'brier_score': float(brier_score_loss(y, p)),
            'precision': float(precision_score(y, pred, zero_division=0)),
            'recall': float(recall_score(y, pred, zero_division=0)),
            'flag_rate': float(pred.mean()),
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
            'illustrative_cost_per_1000': float(1000 * (fp + 5 * fn) / len(y))}

def select_threshold(y, p):
    thresholds = np.linspace(0, 1, 501)
    costs = [np.mean((p >= t) & (y == 0)) + 5 * np.mean((p < t) & (y == 1)) for t in thresholds]
    return float(thresholds[int(np.argmin(costs))])

def bootstrap_ci(y, p, groups, repeats=200):
    # Resample complete duplicate-feature groups, not individual repeated rows.
    rng = np.random.default_rng(SEED)
    unique = np.unique(groups)
    lookup = {g: np.flatnonzero(groups == g) for g in unique}
    values = []
    for _ in range(repeats):
        idx = np.concatenate([lookup[g] for g in rng.choice(unique, len(unique), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        values.append([roc_auc_score(y[idx], p[idx]), average_precision_score(y[idx], p[idx])])
    return dict(zip(['roc_auc', 'average_precision'], np.quantile(values, [.025, .975], axis=0).T.tolist()))
