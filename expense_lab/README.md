# Receipts, please

A small expense-review lab: one actual Gemini run, SQL history features, and a simulated experiment. No real customer data, payments, or measured business savings.

## What ran

- **LLM extraction:** 24 hand-authored synthetic receipts, frozen before the first model response. Gemini **3.1 Pro**, as labelled in the Gemini web app on November 7, 2026, returned all four scored fields correctly for 24/24 records. A simple keyword/first-total baseline matched 17/24. Four cases contain embedded instructions; all four were flagged for review. These are descriptive smoke-test counts, not a statistically established model advantage or a security guarantee.
- **SQL:** SQLite query over 30,000 synthetic transactions / 500 synthetic users. The prior-30-day mean excludes the current timestamp and all same-time peers. An indexed correlated lookup avoids accidentally reading future transactions. Evaluation excludes 3,064 unlabeled rows and rejects duplicate labels. Synthetic precision/recall and illustrative costs are in `sql_results.json`; they say nothing about actual fraud savings.
- **Experiment analysis:** 2,000 simulated companies assigned once to control/treatment. Welch interval and test on company-level review time, plus a sample-ratio-mismatch gate. We inject a two-minute effect. Recovering it checks the analysis; it is not evidence that an LLM saved anyone two minutes. No online A/B test happened.

## Run the receipts

From the repository root, after installing the existing requirements:

```sh
python expense_lab/run_checks.py
pytest -q
```

This reproduces scoring and the SQL/statistical experiments **from the saved model response**. It does not call a model. To obtain a fresh response, start a new Gemini conversation, paste `prompt.txt` without `cases.json` (which contains the labels), save the returned JSON to a new file, and score it. Web-app model versions, hidden system prompts, sampling parameters and latency are not controlled. This is not an API integration, a deployed service, or a reproducible claim about future model output.

`gemini_response.json` was transcribed from the actual visible web-app response; only JSON whitespace was normalized. The submitted input JSON used compact whitespace; the stored prompt pretty-prints the same records and identical policy. Labels were not sent to Gemini. `frozen_inputs.json` stores hashes of the pre-response fixtures and prompt. Evaluation and implementation were prepared with coding-assistant support; the candidate still needs to explain and reproduce them personally.

## What each file does

- `cases.json`: synthetic examples and intended answers; keep out of model input.
- `prompt.txt`: policy and unlabeled input records.
- `gemini_response.json`: actual one-run model outputs, not reconstructed from labels.
- `run_checks.py`: baseline, strict schema validation, field scoring, SQL execution and experiment analysis.
- `last_30_days.sql`: point-in-time history feature and fixed rule.
- `*_results.json`: measured local outputs. Costs and treatment effects are synthetic.
- `../tests/test_expense_lab.py`: boundary, malformed-response, label-join, scoring and interval tests.

## Before calling this production

The 24 examples were authored by the same implementation process as the policy. They are small, English-only, text-only, and intentionally simple. There is no OCR, independent annotation, distributional sampling, latency study, cost measurement, API reliability evaluation, broad adversarial assessment, or warehouse deployment. The 4/4 injection checks are not a defense certification. The response validator raises on invalid data; an actual service would catch that error and route the request to human review. No automated expense approval is implemented.

For a real experiment: randomize by company before exposure, preregister review-time and error-rate guardrails, estimate power from pilot variance, set a fixed horizon and maturation window, check assignment integrity and spillover, analyze intent-to-treat at the randomization unit, and do not stop early on a convenient p-value. The current simulation covers only assignment, SRM and a single company-level continuous outcome; it does not implement the full operational design.

SQLite demonstrates local SQL logic. It does not establish Snowflake/BigQuery experience or distributed performance. Moving to a warehouse requires dialect-specific time windows, partitions, query plans and measured cost/latency.
