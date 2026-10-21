"""Paired 45-vs-48 feature ablation on the FD001 engine holdout."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from stats_utils import binary_metrics, delong_test, mcnemar_test
from iot_revision import fit_model

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SEED = 42

d = pd.read_csv(OUT / 'iot_features_FD001.csv')
units = d.unit.unique()
tr_units, te_units = train_test_split(units, test_size=0.3, random_state=SEED)
m_tr = d.unit.isin(tr_units).values
m_te = ~m_tr
all_feats = [c for c in d.columns if c not in ('unit', 'cycle', 'RUL', 'fault')]
feats45 = [c for c in all_feats if not c.startswith('arima_')]
y = d.fault.values
pred = {}
for label, feats in [('45', feats45), ('48', all_feats)]:
    X = d[feats].values
    for name in ['Random Forest', 'XGBoost']:
        Xtr, ytr = X[m_tr], y[m_tr]
        mn = int(min(np.bincount(ytr)))
        Xr, yr = SMOTE(random_state=SEED, k_neighbors=min(5, mn - 1)).fit_resample(Xtr, ytr)
        model = fit_model(name, Xr, yr).fit(Xr, yr)
        pred[(label, name)] = model.predict_proba(X[m_te])[:, 1]

out = {'n_test': int(m_te.sum())}
for name in ['Random Forest', 'XGBoost']:
    p45, p48 = pred[('45', name)], pred[('48', name)]
    out[name] = {
        'metrics_45': binary_metrics(y[m_te], p45),
        'metrics_48': binary_metrics(y[m_te], p48),
        'delong_45_vs_48': delong_test(y[m_te], p45, p48),
        'mcnemar_45_vs_48': mcnemar_test(y[m_te], p45, p48),
    }
with open(OUT / 'iot_ablation_tests.json', 'w') as f:
    json.dump(out, f, indent=2)
pd.DataFrame({'y': y[m_te],
              'p45_rf': pred[('45', 'Random Forest')],
              'p48_rf': pred[('48', 'Random Forest')],
              'p45_xgb': pred[('45', 'XGBoost')],
              'p48_xgb': pred[('48', 'XGBoost')]}).to_csv(
    OUT / 'iot_ablation_predictions.csv', index=False)
print(json.dumps({k: v for k, v in out.items() if k != 'n_test'}, indent=1)[:1200])
