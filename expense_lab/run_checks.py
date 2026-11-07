"""Small, inspectable expense experiments. Synthetic data cannot pay real bills."""
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chisquare, ttest_ind

ROOT = Path(__file__).resolve().parent
FIELDS = ('category', 'amount', 'currency', 'needs_review')
CATEGORIES = {'meals', 'travel', 'lodging', 'software', 'other', 'unknown'}


def validate_response(rows, ids):
    if not isinstance(rows, list) or len(rows) != len(ids):
        raise ValueError('Wrong response size')
    seen = set()
    for r in rows:
        if not isinstance(r, dict) or set(r) != {'id', *FIELDS}:
            raise ValueError('Unexpected response schema')
        if r['id'] not in ids or r['id'] in seen:
            raise ValueError('Missing, duplicate or unknown ID')
        seen.add(r['id'])
        if r['category'] not in CATEGORIES or type(r['needs_review']) is not bool:
            raise ValueError('Invalid category or review flag')
        if r['amount'] is not None and (type(r['amount']) not in (int, float) or not math.isfinite(r['amount'])):
            raise ValueError('Invalid amount')
        if r['currency'] is not None and (not isinstance(r['currency'], str) or not re.fullmatch('[A-Z]{3}', r['currency'])):
            raise ValueError('Invalid currency')
    return {r['id']: r for r in rows}


def baseline(cases):
    # Deliberately simple keyword/first-total baseline, not tuned against model errors.
    keys = {'meals': r'lunch|dinner|meal|cafe', 'travel': r'taxi|train|airline',
            'lodging': r'hotel', 'software': r'software|license|subscription',
            'other': r'paper|stationery'}
    out = []
    for c in cases:
        s = c['receipt']
        matches = [k for k,v in keys.items() if re.search(v,s,re.I)]
        cat = matches[0] if len(matches)==1 else 'unknown'
        m = re.search(r'\btotal\s+(?:([A-Z]{3})\s+)?\$?(-?\d+(?:\.\d+)?)',s,re.I)
        amount = float(m[2]) if m else None
        currency = m[1].upper() if m and m[1] else None
        review = cat=='unknown' or amount is None or currency is None or amount<0 or amount>500 or 'personal' in s.lower()
        out.append(dict(id=c['id'],category=cat,amount=amount,currency=currency,needs_review=review))
    return out


def score(cases, rows):
    pred = validate_response(rows, {c['id'] for c in cases})
    correct = {k:0 for k in FIELDS}; exact=0; errors=[]; tp=fp=fn=0
    for c in cases:
        p,g=pred[c['id']],c['gold']
        bad=[k for k in FIELDS if p[k]!=g[k]]
        for k in FIELDS: correct[k]+=int(k not in bad)
        exact+=not bad
        if bad: errors.append({'id':c['id'],'fields':bad,'predicted':p,'expected':g})
        tp+=p['needs_review'] and g['needs_review']
        fp+=p['needs_review'] and not g['needs_review']
        fn+=not p['needs_review'] and g['needs_review']
    return {'n':len(cases),'exact_records':exact,'correct_fields':correct,
            'review_precision':tp/(tp+fp) if tp+fp else None,
            'review_recall':tp/(tp+fn) if tp+fn else None,'false_safe':fn,'errors':errors}


def query_history(transactions, method="reference"):
    if method not in {"reference", "window"}:
        raise ValueError("Unknown query method")
    expected=['transaction_id','user_id','ts','amount_cents']
    if set(transactions.columns)!=set(expected) or transactions.empty:
        raise ValueError('Transaction schema or empty input')
    if transactions.isna().any().any() or transactions.transaction_id.duplicated().any():
        raise ValueError('Missing values or duplicate transaction ID')
    for name in ['ts','amount_cents']:
        if not pd.api.types.is_integer_dtype(transactions[name]):
            raise ValueError('Time and cents must be integers')
    if (transactions.amount_cents<0).any():
        raise ValueError('Refunds require a separate rule')
    with sqlite3.connect(':memory:') as db:
        transactions.to_sql('transactions',db,index=False)
        db.execute('CREATE INDEX history_idx ON transactions(user_id,ts)')
        query = 'last_30_days.sql' if method == 'reference' else 'last_30_days_window.sql'
        return pd.read_sql_query((ROOT/query).read_text(),db)


def evaluate_rule(scored, labels):
    if scored.transaction_id.duplicated().any() or labels.transaction_id.duplicated().any():
        raise ValueError('Duplicate ID could multiply losses')
    if not labels.is_fraud.dropna().isin([0,1]).all():
        raise ValueError('Invalid label')
    joined=scored.merge(labels,on='transaction_id',how='left',validate='one_to_one')
    known=joined[joined.is_fraud.notna()]
    if len(known)==0: raise ValueError('No labeled transactions')
    y=known.is_fraud.astype(bool); flag=known.is_anomaly.astype(bool)
    tp=int((y&flag).sum()); fp=int((~y&flag).sum()); fn=int((y&~flag).sum())
    return {'evaluated_rows':len(known),'unlabeled_rows':len(joined)-len(known),
            'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,
            'illustrative_cost_cents':500*fp+int(known.loc[y&~flag,'amount_cents'].sum()),
            'note':'Synthetic labels and costs, not measured fraud savings.'}


def experiment(seed=20261107,n=2000,effect=-2.0):
    # One row per independent company, assignment before outcome, fixed horizon.
    rng=np.random.default_rng(seed)
    treatment=rng.integers(0,2,n)
    baseline_minutes=2.0+rng.lognormal(np.log(13)-0.5*0.6**2,0.6,n)
    outcomes=baseline_minutes+treatment*effect
    control=outcomes[treatment==0]; treated=outcomes[treatment==1]
    test=ttest_ind(treated,control,equal_var=False)
    ci=test.confidence_interval(.95)
    srm=float(chisquare([len(control),len(treated)]).pvalue)
    return {'data':'synthetic independent company-level outcomes','companies':n,
            'assignment_counts':[len(control),len(treated)],'seed':seed,
            'outcome_generator':'2 + lognormal with log-sigma 0.6, mean about 15 minutes before treatment',
            'injected_effect_minutes':effect,'estimated_treatment_minus_control_minutes':float(treated.mean()-control.mean()),
            'welch_95_ci':[float(ci.low),float(ci.high)],'two_sided_p_value':float(test.pvalue),
            'sample_ratio_mismatch_p':srm,'analysis_valid_under_srm_gate':srm>=.01,
            'interpretation':'Checks the analysis implementation only. No online experiment or real time saving occurred.'}


def main():
    frozen=json.loads((ROOT/'frozen_inputs.json').read_text())
    assert hashlib.sha256((ROOT/'cases.json').read_bytes()).hexdigest()==frozen['cases_sha256']
    cases=json.loads((ROOT/'cases.json').read_text())
    model=json.loads((ROOT/'gemini_response.json').read_text())
    llm={'design':frozen['design'],'baseline':score(cases,baseline(cases)), 'gemini':score(cases,model)}
    (ROOT/'llm_results.json').write_text(json.dumps(llm,indent=2))
    rng=np.random.default_rng(20261107); n=30000
    tx=pd.DataFrame({'transaction_id':np.arange(n),'user_id':rng.integers(0,500,n),
                     'ts':rng.integers(0,90*86400,n),'amount_cents':rng.integers(100,20000,n)})
    fraud=rng.random(n)<.03
    tx.loc[fraud,'amount_cents']*=10
    labels=pd.DataFrame({'transaction_id':np.arange(n),'is_fraud':fraud.astype(float)})
    labels.loc[rng.random(n)<.1,'is_fraud']=np.nan
    scored=query_history(tx)
    window_scored=query_history(tx, method='window')
    pd.testing.assert_frame_equal(scored,window_scored)
    sql={'reference_window_equivalence':True,'rows':n,'users':int(tx.user_id.nunique()),'seed':20261107,
         'window':'[timestamp-30 days,timestamp), excluding same-time peers',
         'evaluation':evaluate_rule(scored,labels)}
    (ROOT/'sql_results.json').write_text(json.dumps(sql,indent=2))
    ab=experiment()
    (ROOT/'experiment_results.json').write_text(json.dumps(ab,indent=2))
    print(json.dumps({'llm_exact':llm['gemini']['exact_records'],'baseline_exact':llm['baseline']['exact_records'],'sql':sql,'experiment':ab},indent=2))

if __name__=='__main__':main()
