"""Parallel cross-subset replications for the IoT pipeline (FD002-FD004).

The per-engine ARIMA residual computation is independent across engines and is the
runtime bottleneck; this script distributes those tasks over worker processes while
keeping the modelling split and seeds identical to iot_revision.py.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import iot_revision
from iot_revision import (CHANNELS, SEED, causal_residuals, evaluate_holdout,
                          fit_level_scaler, apply_level_scaler, load_fd)
from stats_utils import calibration_bins

iot_revision.N_JOBS = 2

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
WORKERS = int(os.environ.get('IOT_WORKERS', '3'))
SENSORS = ['s02', 's03', 's04', 's06', 's07', 's08', 's09', 's11', 's12',
           's13', 's14', 's15', 's17', 's20', 's21']


def log(fd, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {fd}: {msg}"
    print(line, flush=True)
    with open(OUT / f'iot_external_{fd}.log', 'a', encoding='utf-8') as f:
        f.write(line + "\n")


def residual_task(args):
    series, order = args
    return causal_residuals(series, order)


def build_features_parallel(df, sensors, train_mask, orders, regime_aware, fd):
    raw = df.copy()
    for s in CHANNELS:
        col = f'arima_{s}'
        groups = [(u, g) for u, g in raw.groupby('unit', sort=False)]
        tasks = [(g[s].values.astype(float), tuple(orders[s])) for _, g in groups]
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            outs = list(ex.map(residual_task, tasks, chunksize=2))
        raw[col] = np.nan
        for (_, g), r in zip(groups, outs):
            raw.loc[g.index, col] = r
        log(fd, f'ARIMA residuals {CHANNELS[s]} done')
    spec = fit_level_scaler(raw, sensors, train_mask, regime_aware)
    scaled = apply_level_scaler(raw, sensors, spec)
    g = scaled.groupby('unit', sort=False)
    rm = g[sensors].rolling(5, min_periods=1).mean().reset_index(drop=True)
    rs = g[sensors].rolling(5, min_periods=1).std().reset_index(drop=True).fillna(0)
    feats = pd.concat([
        scaled[sensors].add_prefix('raw_'),
        rm.add_prefix('rm_'),
        rs.add_prefix('rs_'),
        scaled[[f'arima_{s}' for s in CHANNELS]].copy(),
    ], axis=1)
    out = pd.concat([scaled[['unit', 'cycle', 'RUL', 'fault']], feats], axis=1)
    out.to_csv(OUT / f'iot_features_{fd}.csv', index=False)
    log(fd, f'feature matrix {feats.shape[1]} features, {len(out)} rows')
    return out, list(feats.columns)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--fds', nargs='+', default=['FD002', 'FD003', 'FD004'])
    args = ap.parse_args()
    t0 = time.time()
    with open(OUT / 'iot_fd001_results.json', encoding='utf-8') as f:
        orders = json.load(f)['orders']
    for fd in args.fds:
        log(fd, f'=== external validation (workers={WORKERS}) ===')
        df, _ = load_fd(fd)
        sensors = SENSORS
        log(fd, 'using the fixed 15-sensor FD001 feature schema')
        units = df.unit.unique()
        tr_units, te_units = train_test_split(units, test_size=0.3, random_state=SEED)
        train_mask = df.unit.isin(tr_units).values
        feats, names = build_features_parallel(
            df, sensors, train_mask, orders, fd in ('FD002', 'FD004'), fd)
        y = feats.fault.values
        X = feats[names].values
        m_tr = feats.unit.isin(tr_units).values
        m_te = ~m_tr
        log(fd, 'training/evaluating RF and XGBoost')
        hold, models, preds = evaluate_holdout(
            X[m_tr], y[m_tr], X[m_te], y[m_te],
            feats.loc[m_te, 'unit'].values, names)
        result = {
            'fd': fd, 'n_cycles': int(len(df)), 'n_engines': int(df.unit.nunique()),
            'n_sensors': len(sensors), 'sensors': sensors, 'n_features': len(names),
            'regime_aware': bool(fd in ('FD002', 'FD004')),
            'orders': orders,
            'train_units': [int(u) for u in tr_units],
            'test_units': [int(u) for u in te_units],
            'train_counts': [int((y[m_tr] == 0).sum()), int((y[m_tr] == 1).sum())],
            'test_counts': [int((y[m_te] == 0).sum()), int((y[m_te] == 1).sum())],
            'holdout': hold,
        }
        with open(OUT / f'iot_{fd}_results.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=float)
        log(fd, f"RF F1={hold['Random Forest']['macro_f1']:.4f} "
                f"AUC={hold['Random Forest']['roc_auc']:.4f}; "
                f"XGB F1={hold['XGBoost']['macro_f1']:.4f} "
                f"AUC={hold['XGBoost']['roc_auc']:.4f}")
    log('all', f'external validation complete in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
