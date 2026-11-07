import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]/'expense_lab'
spec=importlib.util.spec_from_file_location('expense_checks',ROOT/'run_checks.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_time_boundaries_same_time_and_future():
    day=86400
    tx=pd.DataFrame({'transaction_id':[1,2,3,4,5,6], 'user_id':[1]*5+[2],
                     'ts':[0,day,31*day,31*day,32*day,31*day],
                     'amount_cents':[10000,100,400,900,100000,50]})
    out=m.query_history(tx).set_index('transaction_id')
    assert out.loc[3,'past_mean_cents']==100
    assert out.loc[4,'past_mean_cents']==100
    assert out.loc[3,'is_anomaly']==1
    assert np.isnan(out.loc[6,'past_mean_cents'])
    assert out.loc[6,'is_anomaly']==0


def test_labels_no_silent_missing_or_duplicate_join():
    scored=pd.DataFrame({'transaction_id':[1,2,3,4],'amount_cents':[100,200,300,400],'is_anomaly':[1,0,1,0]})
    labels=pd.DataFrame({'transaction_id':[1,2,3],'is_fraud':[0,1,1]})
    r=m.evaluate_rule(scored,labels)
    assert (r['tp'],r['fp'],r['fn'],r['unlabeled_rows'])==(1,1,1,1)
    assert r['illustrative_cost_cents']==700
    with pytest.raises(ValueError):m.evaluate_rule(scored,pd.concat([labels,labels]))


@pytest.mark.parametrize('bad',['duplicate','boolean_amount','nan_amount','extra_key','string_flag','missing_id'])
def test_llm_schema_fails_closed(bad):
    rows=[dict(id='A',category='meals',amount=1,currency='USD',needs_review=False)]
    if bad=='duplicate':rows*=2
    if bad=='boolean_amount':rows[0]['amount']=True
    if bad=='nan_amount':rows[0]['amount']=float('nan')
    if bad=='extra_key':rows[0]['approved']=True
    if bad=='string_flag':rows[0]['needs_review']='false'
    if bad=='missing_id':rows[0]['id']='B'
    with pytest.raises(ValueError):m.validate_response(rows,{'A'})


def test_known_scoring_error_is_counted():
    cases=[{'id':'A','gold':dict(category='meals',amount=1,currency='USD',needs_review=True)}]
    pred=[dict(id='A',category='meals',amount=1,currency='USD',needs_review=False)]
    r=m.score(cases,pred)
    assert r['exact_records']==0 and r['false_safe']==1 and r['review_recall']==0


def test_experiment_shift_and_interval():
    a=m.experiment(effect=0);b=m.experiment(effect=-2)
    assert b['estimated_treatment_minus_control_minutes']==pytest.approx(a['estimated_treatment_minus_control_minutes']-2)
    np.testing.assert_allclose(b['welch_95_ci'],np.array(a['welch_95_ci'])-2)
    assert sum(a['assignment_counts'])==2000


def test_frozen_fixture_checksum():
    import hashlib
    frozen=json.loads((ROOT/'frozen_inputs.json').read_text())
    assert hashlib.sha256((ROOT/'cases.json').read_bytes()).hexdigest()==frozen['cases_sha256']
