# Credit risk, with receipts

Can a model spot credit-card defaults without sneaking a look at the answers?

That's the project. Thirty thousand public records, a few ordinary models, and enough checks to make cheating inconvenient. No mystery neural net. No claim that a weekend benchmark has fixed lending.

The final holdout: **ROC-AUC 0.805 · AP 0.600 · Brier 0.129**, on **6,013 records**. Default prevalence is **22.25%**. These are offline results.

![The holdout has entered the chat](reports/reality_check.png)

## Run it before believing it

Tested with Python 3.14.7 and scikit-learn 1.9.1. In this folder:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps -e .
pytest -q
credit-risk train
credit-risk predict --input examples/made_up_people.csv --output examples/what_the_model_said.csv
```

The first training run downloads the official UCI archive. After that, it uses the local copy. You'll need internet for the initial download and packages.

Training saves:

- `reports/the_numbers.json` — scores, splits, threshold, data checksum. The receipts.
- `reports/reality_check.png` — ROC, precision-recall, and calibration plots.
- `reports/what_mattered.csv` — permutation importance on validation data.
- `models/model.joblib` — the fitted model and threshold. Generated locally, kept out of Git.

The people in `examples/made_up_people.csv` are invented. Nobody's credit application is being posted here. Only load joblib files you trust: they can execute code.

## The rules of this little experiment

**No copy-paste classmates across splits.** All 30,000 rows stay. Rows with identical values for the 19 financial inputs get grouped together before splitting. That leaves 17,962 training, 6,025 validation, and 6,013 test rows. Roughly 60/20/20; grouped, not stratified.

**No preprocessing clairvoyance.** Median imputation, missing-value indicators, and scaling are fitted inside each training pipeline. Validation and test data don't contribute statistics. There are six added features: current and average bill/payment-to-limit ratios, maximum delay, and mean delay.

**The baseline gets a seat at the table.** Compare a prior-only dummy, logistic regression, histogram gradient boosting, and a three-fold sigmoid-calibrated boosting ensemble. Calibration folds respect the same duplicate-feature groups.

**Pick a winner before opening the envelope.** Highest validation average precision wins; lower Brier breaks ties. Pick the threshold on validation too. Keep that fitted model rather than retraining on validation, then evaluate it once on the test set. The 200 group-bootstrap resamples estimate uncertainty for this fixed model.

AP here means scikit-learn's non-interpolated average precision. It isn't trapezoidal PR-AUC wearing a different hat.

## The model bake-off

These are **validation** scores:

| Model | ROC-AUC | AP | Brier ↓ |
|---|---:|---:|---:|
| Prior-only baseline | 0.500 | 0.224 | 0.174 |
| Logistic regression | 0.731 | 0.493 | 0.147 |
| Gradient boosting | 0.779 | 0.549 | 0.136 |
| Calibrated boosting | 0.780 | 0.552 | 0.136 |

Calibrated boosting won on AP, barely. Its Brier score was actually a little **worse** than plain boosting: 0.136277 vs 0.136022. So no, this isn't a “calibration made everything better” victory lap. This comparison also mixes calibration with ensembling; it doesn't isolate either effect or establish a significant improvement.

## The test set gets the last word

| What was measured | Result |
|---|---:|
| ROC-AUC | 0.8054 |
| ROC-AUC 95% group-bootstrap interval | 0.7912–0.8197 |
| Average precision | 0.6000 |
| AP 95% group-bootstrap interval | 0.5695–0.6278 |
| Brier score | 0.1288 |
| Threshold | 0.194 |
| Precision / recall | 42.63% / 74.14% |
| Flagged for review | 38.70% |

Threshold selection assumes a missed default costs five units and a false alarm costs one: minimize validation `FP + 5 × FN`. That's an illustrative assumption, not dollars, not a Ramp policy, and definitely not money this repo has saved anyone.

At that threshold: **3,340 true negatives, 1,335 false positives, 346 false negatives, 992 true positives**. That's a lot of false alarms. An actual review team would probably have questions. Fair questions.

## Things that can ruin a nice-looking score

The data is historical Taiwanese **consumer** credit data, not modern business underwriting data. One cohort, random group holdout. No future-month test. Temporal drift, reject inference, selection bias, customer-level longitudinal leakage, subgroup fairness, and deployment monitoring haven't been tested.

ID, sex, education, marriage, and age are excluded from prediction. That doesn't magically make the model fair: financial inputs can still act as proxies.

Permutation importance shows predictive association, not causation. Bootstrap intervals describe resampling uncertainty for the fitted model, not every uncertainty involved in training and choosing it. This isn't a production lending system or an approval policy.

## Files, without the treasure hunt

| File | What it does |
|---|---|
| `src/credit_risk/data.py` | Download, validate, and prepare features |
| `src/credit_risk/model.py` | Split data, compare models, score predictions |
| `src/credit_risk/run_it.py` | Training and batch prediction commands |
| `tests/test_no_peeking.py` | Check splits, bad inputs, preprocessing, saved models, and thresholds |
| `.github/workflows/keep_me_honest.yml` | Run the tests on GitHub |
| `reports/` | Results and plots, including the less flattering bits |

There are 11 tests. They check specific failure modes; they don't issue a certificate of statistical sainthood. [The original Linux run passed](https://github.com/1143151076-pixel/dulin/actions/runs/42417818253); current runs are on the [Actions page](https://github.com/1143151076-pixel/dulin/actions).

## Where the data came from

I. Yeh (2009), *Default of Credit Card Clients*, UCI Machine Learning Repository. [Dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) · [DOI](https://doi.org/10.24432/C55S3H). CC BY 4.0. Financial history covers April–September 2005. This project selects financial columns, adds features, and groups records for evaluation. Raw data stays out of Git.

Useful references: [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), [calibration](https://scikit-learn.org/stable/modules/calibration.html).

## Who made this

Du Lin's personal portfolio project, built with coding-assistant help for implementation, execution, and documentation. Not a Ramp project or a university research deliverable. The code and results are here to inspect and reproduce; understanding them still takes actual work. Annoying, but true.

## A second lab: receipts, please

[`expense_lab/`](expense_lab/) adds a real, manually run Gemini receipt-extraction smoke test, a SQLite point-in-time transaction query, and company-level experiment analysis on synthetic data. It includes frozen examples, saved model output, baseline comparison and boundary tests. This is a small evaluation exercise, not a production LLM service or online A/B test.

Run `python expense_lab/run_checks.py` to reproduce the saved-response scoring and simulations. The full suite now has **22 local passing tests** (11 original credit-risk tests + 11 expense-lab tests); CI for this addition should be checked on the latest commit.
