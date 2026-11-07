"""Get the data. Check the columns. Refuse obvious nonsense."""
from pathlib import Path
import hashlib
import urllib.request
import zipfile
import pandas as pd
import numpy as np

URL = 'https://archive.ics.uci.edu/static/public/350/default%2Bof%2Bcredit%2Bcard%2Bclients.zip'
TARGET = 'default payment next month'
PAY = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
BILL = [f'BILL_AMT{i}' for i in range(1, 7)]
PAID = [f'PAY_AMT{i}' for i in range(1, 7)]
FEATURES = ['LIMIT_BAL'] + PAY + BILL + PAID

def load_data(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(URL, timeout=90) as response:
            path.write_bytes(response.read())
    with zipfile.ZipFile(path) as archive:
        member = next(n for n in archive.namelist() if n.endswith('.xls'))
        with archive.open(member) as stream:
            df = pd.read_excel(stream, header=1)
    if len(df) != 30000 or not df['ID'].is_unique:
        raise ValueError('Unexpected UCI dataset shape or nonunique IDs')
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError('Invalid labels')
    return df, hashlib.sha256(path.read_bytes()).hexdigest()

def validate_features(frame):
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError('Expected nonempty data with unique columns')
    if set(frame.columns) != set(FEATURES):
        raise ValueError(f'Expected exactly these columns: {FEATURES}')
    x = frame.loc[:, FEATURES].apply(pd.to_numeric, errors='raise').astype(float)
    if np.isinf(x.to_numpy()).any():
        raise ValueError('Infinite inputs are not supported')
    # Missing numeric values are permitted and imputed inside the fitted pipeline.
    if (x['LIMIT_BAL'].dropna() <= 0).any():
        raise ValueError('LIMIT_BAL must be positive when supplied')
    return x

def engineer(frame):
    x = validate_features(frame).copy()
    x['bill_to_limit'] = x['BILL_AMT1'] / x['LIMIT_BAL']
    x['payment_to_limit'] = x['PAY_AMT1'] / x['LIMIT_BAL']
    x['max_delay'] = x[PAY].max(axis=1)
    x['mean_delay'] = x[PAY].mean(axis=1)
    x['mean_bill_to_limit'] = x[BILL].mean(axis=1) / x['LIMIT_BAL']
    x['mean_payment_to_limit'] = x[PAID].mean(axis=1) / x['LIMIT_BAL']
    return x
