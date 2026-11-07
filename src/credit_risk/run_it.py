"""Run with python -m credit_risk.run_it train or predict."""
import argparse
import json
import platform
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.inspection import permutation_importance
from .data import load_data, validate_features, FEATURES, TARGET, URL
from .model import candidates, partitions, metrics, select_threshold, bootstrap_ci


def train(root):
    started = time.perf_counter()
    root = Path(root)
    reports, models = root / 'reports', root / 'models'
    reports.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    df, checksum = load_data(root / 'data/uci.zip')
    x = validate_features(df[FEATURES])
    y = df[TARGET].to_numpy()
    tr, va, te, groups = partitions(x)
    assert not (set(groups[tr]) & set(groups[va]) or set(groups[tr]) & set(groups[te]) or set(groups[va]) & set(groups[te]))
    models_by_name = candidates(x.iloc[tr], y[tr], groups[tr])
    results = {}
    for name, estimator in models_by_name.items():
        begin = time.perf_counter()
        estimator.fit(x.iloc[tr], y[tr])
        p = estimator.predict_proba(x.iloc[va])[:, 1]
        results[name] = metrics(y[va], p)
        results[name]['fit_seconds'] = round(time.perf_counter() - begin, 3)
        print(name, json.dumps(results[name]), flush=True)
    # Pick now. The test set is not a suggestion box.
    selected = max(results, key=lambda name: (results[name]['average_precision'], -results[name]['brier_score']))
    estimator = models_by_name[selected]
    threshold = select_threshold(y[va], estimator.predict_proba(x.iloc[va])[:, 1])
    importance = permutation_importance(estimator, x.iloc[va], y[va], scoring='average_precision',
                                      n_repeats=3, random_state=42, n_jobs=1)
    pd.DataFrame({'feature': FEATURES, 'validation_ap_decrease': importance.importances_mean,
                  'std': importance.importances_std}).sort_values('validation_ap_decrease', ascending=False).to_csv(reports / 'what_mattered.csv', index=False)
    # Decisions locked. Time for the awkward bit: the holdout.
    p = estimator.predict_proba(x.iloc[te])[:, 1]
    test = metrics(y[te], p, threshold)
    report = {'source_url': URL, 'source_sha256': checksum, 'rows': len(df),
              'raw_features': len(FEATURES), 'engineered_features': 6,
              'excluded': ['ID', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE'],
              'seed': 42, 'selected_model': selected, 'threshold': threshold,
              'selection_metric': 'validation average precision; lower Brier breaks ties',
              'cost_assumption': 'False negative = 5 units; false positive = 1 unit; illustrative only',
              'splits': {name: {'rows': len(idx), 'positive_rate': float(y[idx].mean()),
                                'groups': len(set(groups[idx]))} for name, idx in [('train',tr),('validation',va),('test',te)]},
              'validation': results, 'test': test,
              'test_group_bootstrap_95_ci': bootstrap_ci(y[te], p, groups[te]),
              'runtime': {'python': platform.python_version(), 'sklearn': sklearn.__version__},
              'elapsed_seconds': round(time.perf_counter() - started, 2)}
    (reports / 'the_numbers.json').write_text(json.dumps(report, indent=2))
    # IDs facilitate audit, with raw data and row predictions excluded from publication.
    pd.DataFrame({'id': df['ID'], 'split': [''] * len(df)}).assign(
        split=np.select([df.index.isin(tr), df.index.isin(va)], ['train','validation'], default='test')
    ).to_csv(root / 'data/split_manifest.csv', index=False)
    joblib.dump({'model': estimator, 'threshold': threshold, 'features': FEATURES,
                 'source_sha256': checksum}, models / 'model.joblib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fpr, tpr, _ = roc_curve(y[te], p)
    axes[0].plot(fpr, tpr, color='#147d92', label=f"AUC {test['roc_auc']:.3f}")
    axes[0].plot([0,1], [0,1], '--', color='gray'); axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='Holdout ROC'); axes[0].legend()
    precision, recall, _ = precision_recall_curve(y[te], p)
    axes[1].plot(recall, precision, color='#147d92', label=f"AP {test['average_precision']:.3f}")
    axes[1].axhline(y[te].mean(), linestyle='--', color='gray', label='Prevalence')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='Holdout precision-recall'); axes[1].legend()
    observed, predicted = calibration_curve(y[te], p, n_bins=10, strategy='quantile')
    axes[2].plot(predicted, observed, 'o-', color='#147d92'); axes[2].plot([0,1], [0,1], '--', color='gray')
    axes[2].set(xlabel='Mean predicted probability', ylabel='Observed default rate', title='Calibration (quantile bins)')
    fig.tight_layout(); fig.savefig(reports / 'reality_check.png', dpi=160); plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)


def predict(model_path, input_path, output_path):
    # Load only artifacts you trust: joblib/pickle may execute code on load.
    artifact = joblib.load(model_path)
    x = validate_features(pd.read_csv(input_path))
    p = artifact['model'].predict_proba(x)[:, 1]
    pd.DataFrame({'default_probability': p, 'review_flag': p >= artifact['threshold']}).to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('train'); fit.add_argument('--root', default='.')
    pred = commands.add_parser('predict')
    pred.add_argument('--model', default='models/model.joblib')
    pred.add_argument('--input', required=True); pred.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'train': train(args.root)
    else: predict(args.model, args.input, args.output)

if __name__ == '__main__':
    main()
