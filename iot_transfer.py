"""Strict FD001-to-FD002/3/4 transfer evaluation.

The FD001 models, ARIMA orders, feature list, and sensor list are frozen. Target
subsets are used only for feature construction and scoring; no target labels are
used for fitting, scaling, or threshold selection.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

from iot_revision import (CHANNELS, MODELS, OUT, SEED, apply_level_scaler,
                          causal_residuals, fit_level_scaler, load_fd)
from stats_utils import binary_metrics, bootstrap_metrics

TARGETS = ['FD002', 'FD003', 'FD004']
WORKERS = int(os.environ.get('IOT_WORKERS', '4'))


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / 'iot_transfer.log', 'a', encoding='utf-8') as f:
        f.write(line + "\n")


def residual_task(args):
    series, order = args
    return causal_residuals(series, order)


def build_transfer_features(df, sensors, orders, spec, fd):
    path = OUT / f"iot_transfer_features_{fd}.csv"
    if path.exists():
        log(f"{fd}: reusing cached transfer features")
        return pd.read_csv(path)
    raw = df.copy()
    for s in CHANNELS:
        col = f"arima_{s}"
        groups = [(u, g) for u, g in raw.groupby('unit', sort=False)]
        tasks = [(g[s].values.astype(float), tuple(orders[s])) for _, g in groups]
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            outs = list(ex.map(residual_task, tasks, chunksize=2))
        raw[col] = np.nan
        for (_, g), r in zip(groups, outs):
            raw.loc[g.index, col] = r
        log(f"{fd}: ARIMA residuals {CHANNELS[s]} done")
    scaled = apply_level_scaler(raw, sensors, spec)
    g = scaled.groupby('unit', sort=False)
    rm = g[sensors].rolling(5, min_periods=1).mean().reset_index(drop=True)
    rs = g[sensors].rolling(5, min_periods=1).std().reset_index(drop=True).fillna(0)
    feats = pd.concat([
        scaled[sensors].add_prefix('raw_'),
        rm.add_prefix('rm_'),
        rs.add_prefix('rs_'),
        scaled[[f"arima_{s}" for s in CHANNELS]].copy(),
    ], axis=1)
    out = pd.concat([scaled[['unit', 'cycle', 'RUL', 'fault']], feats], axis=1)
    out.to_csv(path, index=False)
    log(f"{fd}: feature matrix {feats.shape[1]} features, {len(out)} rows")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fds', nargs='+', default=TARGETS)
    args = ap.parse_args()

    with open(OUT / 'iot_fd001_results.json', encoding='utf-8') as f:
        fd001 = json.load(f)
    bundle = joblib.load(MODELS / 'iot_fd001.joblib')
    sensors = bundle['sensors']
    orders = bundle['orders']
    features = bundle['features']

    df, _ = load_fd('FD001')
    train_units = set(fd001['train_units'])
    train_mask = df.unit.isin(train_units).values
    spec = fit_level_scaler(df, sensors, train_mask, False)

    results = {
        'protocol': 'strict transfer; FD001-fitted preprocessing and models; '
                    'target labels unused',
        'source': 'FD001',
        'source_train_units': int(len(train_units)),
        'source_features': len(features),
        'source_sensors': sensors,
        'orders': {k: list(v) for k, v in orders.items()},
        'targets': {},
    }
    if (OUT / 'iot_transfer_results.json').exists():
        with open(OUT / 'iot_transfer_results.json', encoding='utf-8') as f:
            results['targets'] = json.load(f).get('targets', {})

    for fd in args.fds:
        log(f"=== strict transfer: FD001 -> {fd} ===")
        tdf, _ = load_fd(fd)
        tdf = tdf.sort_values(['unit', 'cycle']).reset_index(drop=True)
        feats = build_transfer_features(tdf, sensors, orders, spec, fd)
        units = tdf.unit.unique()
        _, te_units = train_test_split(units, test_size=0.3, random_state=SEED)
        mask = feats.unit.isin(te_units).values
        X = feats.loc[mask, features].values
        y = feats.loc[mask, 'fault'].values.astype(int)
        groups = feats.loc[mask, 'unit'].values

        entry = {
            'n_engines_total': int(tdf.unit.nunique()),
            'n_engines_eval': int(len(te_units)),
            'n_rows_eval': int(mask.sum()),
            'positive_rate_eval': float(y.mean()),
            'models': {},
        }
        pred_rows = []
        for key, label in [('rf', 'Random Forest'), ('xgb', 'XGBoost')]:
            p = bundle[key].predict_proba(X)[:, 1]
            m = binary_metrics(y, p)
            m['bootstrap'] = bootstrap_metrics(y, p, groups=groups, n_boot=1000)
            entry['models'][label] = m
            pred_rows.append(pd.DataFrame({
                'fd': fd, 'model': label, 'unit': groups,
                'cycle': feats.loc[mask, 'cycle'].values,
                'y': y, 'p': p,
            }))

        with open(OUT / f'iot_{fd}_results.json', encoding='utf-8') as f:
            within = json.load(f)
        for label in ['Random Forest', 'XGBoost']:
            m = entry['models'][label]
            w = within['holdout'][label]
            entry['models'][label]['within_subset'] = {
                'macro_f1': w['macro_f1'],
                'roc_auc': w['roc_auc'],
                'ap': w['ap'],
            }
            entry['models'][label]['gap_vs_within'] = {
                'macro_f1': float(w['macro_f1'] - m['macro_f1']),
                'roc_auc': float(w['roc_auc'] - m['roc_auc']),
                'ap': float(w['ap'] - m['ap']),
            }
        results['targets'][fd] = entry
        pd.concat(pred_rows).to_csv(OUT / f'iot_transfer_predictions_{fd}.csv', index=False)
        for label in ['Random Forest', 'XGBoost']:
            m = entry['models'][label]
            log(f"{fd} {label}: F1={m['macro_f1']:.4f} AUC={m['roc_auc']:.4f} "
                f"AP={m['ap']:.4f} within F1={m['within_subset']['macro_f1']:.4f}")
        with open(OUT / 'iot_transfer_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=float)

    with open(OUT / 'iot_transfer_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=float)
    log('saved iot_transfer_results.json')


if __name__ == '__main__':
    main()
