"""Revised IoT pipeline.

- causal, per-engine ARIMA residual features (no future leakage)
- 45 vs 48 feature ablation
- engine-grouped 5-fold CV + untouched engine holdout (FD001)
- external validation on FD002/FD003/FD004
- permutation importance, SHAP (if available), PR and calibration artefacts
"""
import warnings
warnings.filterwarnings('ignore')
import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_curve)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from imblearn.over_sampling import SMOTE
import xgboost as xgb

from stats_utils import binary_metrics, bootstrap_metrics, calibration_bins

ROOT = Path(__file__).resolve().parent
CMAP = ROOT / "CMaps"
OUT = ROOT / "results"
MODELS = OUT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
SEED = 42
N_JOBS = 4
CHANNELS = {'s03': 'Temperature', 's07': 'Compressor_Pressure', 's17': 'Coolant_Bleed'}
LOG_PATH = OUT / "iot_revision.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line + "\n")


def load_fd(fd):
    cols = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i:02d}' for i in range(1, 22)]
    df = pd.read_csv(CMAP / f"train_{fd}.txt", sep=r"\s+", header=None, names=cols)
    maxc = df.groupby('unit')['cycle'].max()
    df['RUL'] = maxc[df.unit].values - df['cycle']
    df['fault'] = (df['RUL'] <= 30).astype(int)
    sensors = [c for c in cols[5:] if df[c].std() > 1e-3]
    df = df.sort_values(['unit', 'cycle']).reset_index(drop=True)
    return df, sensors


def select_arima_orders(df, train_units, max_engines=12):
    """AIC-based order search on training engines only (diagnostic selection)."""
    rng = np.random.default_rng(SEED)
    engines = np.array(sorted(train_units))
    sample = rng.choice(engines, size=min(max_engines, len(engines)), replace=False)
    orders = {}
    table = []
    for s, name in CHANNELS.items():
        best = None
        for d in (0, 1):
            for p in (0, 1, 2):
                for q in (0, 1, 2):
                    aics = []
                    for u in sample:
                        y = df.loc[df.unit == u, s].values.astype(float)
                        try:
                            fit = ARIMA(y, order=(p, d, q), trend='c').fit()
                            if np.isfinite(fit.aic):
                                aics.append(float(fit.aic))
                        except Exception:
                            pass
                    if len(aics) >= max(3, len(sample) // 2):
                        med = float(np.median(aics))
                        table.append({'channel': name, 'sensor': s, 'p': p, 'd': d,
                                      'q': q, 'median_aic': med, 'n_fits': len(aics)})
                        if best is None or med < best[0]:
                            best = (med, (p, d, q))
        orders[s] = best[1] if best else (1, 0, 1)
        log(f"ARIMA order {name} ({s}): {orders[s]} (median AIC={best[0] if best else 'n/a'})")
    return orders, pd.DataFrame(table)


def causal_residuals(series, order, warmup=20, refit_every=10):
    """One-step-ahead residuals; each residual uses only observations up to t-1."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    out = np.zeros(n, dtype=float)
    if n <= warmup + 2:
        return out
    last = None
    for start in range(warmup, n, refit_every):
        try:
            last = ARIMA(series[:start], order=order, trend='c').fit()
        except Exception:
            pass
        if last is None:
            continue
        for t in range(start, min(start + refit_every, n)):
            if t > start:
                try:
                    last = last.append(series[t - 1:t], refit=False)
                except Exception:
                    pass
            try:
                fc = float(last.get_forecast(steps=1).predicted_mean[0])
                out[t] = series[t] - fc
            except Exception:
                out[t] = 0.0
    return out


def fit_level_scaler(df, sensors, train_mask, regime_aware):
    if not regime_aware:
        scaler = StandardScaler().fit(df.loc[train_mask, sensors])
        return {'kind': 'global', 'scaler': scaler}
    os_cols = ['os1', 'os2', 'os3']
    km = KMeans(n_clusters=6, random_state=SEED, n_init=10).fit(df.loc[train_mask, os_cols])
    labels = km.predict(df[os_cols])
    scalers = {}
    for c in range(6):
        mask = train_mask & (labels == c)
        if mask.sum() >= 50:
            scalers[c] = StandardScaler().fit(df.loc[mask, sensors])
        else:
            scalers[c] = StandardScaler().fit(df.loc[train_mask, sensors])
    return {'kind': 'regime', 'km': km, 'scalers': scalers, 'labels': labels}


def apply_level_scaler(df, sensors, spec):
    df = df.copy()
    if spec['kind'] == 'global':
        df[sensors] = spec['scaler'].transform(df[sensors])
    else:
        labels = spec['labels']
        for c in range(6):
            mask = labels == c
            if mask.any():
                df.loc[mask, sensors] = spec['scalers'][c].transform(df.loc[mask, sensors])
    return df


def build_features(df, sensors, train_mask, orders, regime_aware, fd, save=True):
    t0 = time.time()
    raw = df.copy()
    # ARIMA residuals on raw channels (causal, per engine)
    for s in CHANNELS:
        col = f"arima_{s}"
        raw[col] = np.nan
        for u, g in raw.groupby('unit', sort=False):
            idx = g.index.to_numpy()
            raw.loc[idx, col] = causal_residuals(g[s].values, orders[s])
        log(f"  ARIMA residuals {CHANNELS[s]} done ({time.time()-t0:.1f}s)")
    spec = fit_level_scaler(raw, sensors, train_mask, regime_aware)
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
    if save:
        out.to_csv(OUT / f"iot_features_{fd}.csv", index=False)
    log(f"  feature matrix {fd}: {feats.shape[1]} features, {len(out)} rows "
        f"({time.time()-t0:.1f}s)")
    return out, list(feats.columns), spec


def arima_diagnostics(df, orders, engines=None, fd='FD001'):
    rows = []
    units = engines if engines is not None else sorted(df.unit.unique())
    for u in units:
        sub = df[df.unit == u]
        for s, name in CHANNELS.items():
            y = sub[s].values.astype(float)
            try:
                fit = ARIMA(y, order=orders[s], trend='c').fit()
                lb = float(acorr_ljungbox(fit.resid, lags=[10], return_df=True)['lb_pvalue'].values[0])
                rows.append({'fd': fd, 'unit': int(u), 'sensor': s, 'channel': name,
                             'order': str(orders[s]), 'aic': float(fit.aic),
                             'rmse': float(np.sqrt(np.mean(fit.resid ** 2))),
                             'lb_p': lb})
            except Exception as e:
                rows.append({'fd': fd, 'unit': int(u), 'sensor': s, 'channel': name,
                             'order': str(orders[s]), 'aic': np.nan, 'rmse': np.nan,
                             'lb_p': np.nan, 'error': str(e)})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / f"iot_arima_diagnostics_{fd}.csv", index=False)
    summary = {}
    for name, g in d.groupby('channel'):
        summary[name] = {
            'n_engines': int(g.unit.nunique()),
            'median_aic': float(g.aic.median()),
            'median_rmse': float(g.rmse.median()),
            'ljung_box_pass_rate': float((g.lb_p > 0.05).mean()),
            'median_lb_p': float(g.lb_p.median()),
        }
    return summary, d


def fit_model(name, X, y):
    if name == 'Random Forest':
        return RandomForestClassifier(n_estimators=300, random_state=SEED,
                                      class_weight='balanced', n_jobs=N_JOBS)
    return xgb.XGBClassifier(n_estimators=300, random_state=SEED, eval_metric='logloss',
                             verbosity=0, n_jobs=N_JOBS)


def cv_experiment(X, y, groups, feature_names, label):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    oof = {}
    for model_name in ['Random Forest', 'XGBoost']:
        oof[model_name] = np.full(len(y), np.nan)
        for fold, (tr, va) in enumerate(cv.split(X, y, groups)):
            Xtr, Xva, ytr, yva = X[tr], X[va], y[tr], y[va]
            mn = int(min(np.bincount(ytr)))
            Xr, yr = SMOTE(random_state=SEED, k_neighbors=min(5, mn - 1)).fit_resample(Xtr, ytr)
            model = fit_model(model_name, Xr, yr)
            model.fit(Xr, yr)
            p = model.predict_proba(Xva)[:, 1]
            oof[model_name][va] = p
            m = binary_metrics(yva, p)
            m.update({'fold': fold, 'model': model_name, 'feature_set': label})
            rows.append(m)
    df = pd.DataFrame(rows)
    summary = {}
    for model_name, g in df.groupby('model'):
        summary[model_name] = {}
        for k in ['acc', 'macro_f1', 'roc_auc', 'ap', 'precision', 'recall']:
            summary[model_name][k] = {'mean': float(g[k].mean()),
                                      'std': float(g[k].std(ddof=1))}
        summary[model_name]['bootstrap'] = bootstrap_metrics(
            y, oof[model_name], groups=groups, n_boot=500)
    return summary, df, oof


def evaluate_holdout(Xtr, ytr, Xte, yte, units_te, feature_names):
    out = {}
    models = {}
    preds = {}
    for model_name in ['Random Forest', 'XGBoost']:
        mn = int(min(np.bincount(ytr)))
        Xr, yr = SMOTE(random_state=SEED, k_neighbors=min(5, mn - 1)).fit_resample(Xtr, ytr)
        model = fit_model(model_name, Xr, yr)
        model.fit(Xr, yr)
        p = model.predict_proba(Xte)[:, 1]
        models[model_name] = model
        preds[model_name] = p
        m = binary_metrics(yte, p)
        m['bootstrap'] = bootstrap_metrics(yte, p, groups=units_te, n_boot=1000)
        out[model_name] = m
    return out, models, preds


def permutation_table(model, X, y, feature_names, n_repeats=10):
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=SEED,
                               scoring='roc_auc', n_jobs=N_JOBS)
    d = pd.DataFrame({'feature': feature_names, 'mean': r.importances_mean,
                      'std': r.importances_std})
    return d.sort_values('mean', ascending=False).reset_index(drop=True)


def maybe_shap(model, X, feature_names, tag, max_rows=2000):
    try:
        import shap
    except Exception as e:
        log(f"SHAP unavailable: {e}")
        return None
    try:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X), size=min(max_rows, len(X)), replace=False)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X[idx])
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv)
        d = pd.DataFrame({'feature': feature_names,
                          'mean_abs_shap': np.abs(sv).mean(axis=0)})
        d = d.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        d.to_csv(OUT / f"iot_shap_{tag}.csv", index=False)
        log(f"  SHAP saved for {tag}")
        return d
    except Exception as e:
        log(f"SHAP failed for {tag}: {e}")
        return None


def run_fd001(orders):
    log("=== FD001: primary experiment ===")
    df, sensors = load_fd('FD001')
    log(f"FD001: {len(df)} cycles, {df.unit.nunique()} engines, {len(sensors)} sensors")
    units = df.unit.unique()
    tr_units, te_units = train_test_split(units, test_size=0.3, random_state=SEED)
    train_mask = df.unit.isin(tr_units).values
    feats, names, spec = build_features(df, sensors, train_mask, orders, False, 'FD001')
    arima_sum, arima_df = arima_diagnostics(df, orders, tr_units, 'FD001')
    X48 = feats[names].values
    y = feats['fault'].values
    groups = feats['unit'].values
    m_tr = feats.unit.isin(tr_units).values
    m_te = ~m_tr
    res45 = [c for c in names if not c.startswith('arima_')]
    X45 = feats[res45].values

    cv48, cv48_df, oof48 = cv_experiment(X48[m_tr], y[m_tr], groups[m_tr], names, '48')
    cv45, cv45_df, oof45 = cv_experiment(X45[m_tr], y[m_tr], groups[m_tr], res45, '45')
    cv48_df.to_csv(OUT / 'iot_cv_folds_48.csv', index=False)
    cv45_df.to_csv(OUT / 'iot_cv_folds_45.csv', index=False)
    np.save(OUT / 'iot_oof_48_rf.npy', oof48['Random Forest'])
    np.save(OUT / 'iot_oof_48_xgb.npy', oof48['XGBoost'])

    hold48, models48, preds48 = evaluate_holdout(
        X48[m_tr], y[m_tr], X48[m_te], y[m_te], groups[m_te], names)
    hold45, models45, preds45 = evaluate_holdout(
        X45[m_tr], y[m_tr], X45[m_te], y[m_te], groups[m_te], res45)

    pred = pd.DataFrame({'unit': groups[m_te], 'cycle': feats.loc[m_te, 'cycle'].values,
                         'y': y[m_te],
                         'p_rf': preds48['Random Forest'],
                         'p_xgb': preds48['XGBoost']})
    pred.to_csv(OUT / 'iot_holdout_predictions.csv', index=False)
    joblib.dump({'rf': models48['Random Forest'], 'xgb': models48['XGBoost'],
                 'features': names, 'sensors': sensors, 'orders': orders},
                MODELS / 'iot_fd001.joblib')

    perm = {}
    for model_name in ['Random Forest', 'XGBoost']:
        d = permutation_table(models48[model_name], X48[m_te], y[m_te], names)
        d.to_csv(OUT / f"iot_permutation_{model_name.replace(' ', '_').lower()}.csv", index=False)
        perm[model_name] = d.head(15).to_dict('records')
    shap = maybe_shap(models48['XGBoost'], X48[m_te], names, 'xgb')

    # PR / calibration arrays for figures
    y_te = y[m_te]
    pr = {}
    cal = {}
    for model_name in ['Random Forest', 'XGBoost']:
        p = preds48[model_name]
        prec, rec, thr = precision_recall_curve(y_te, p)
        fpr, tpr, _ = roc_curve(y_te, p)
        pr[model_name] = {'precision': prec.tolist(), 'recall': rec.tolist(),
                          'ap': float(average_precision_score(y_te, p))}
        cal[model_name] = calibration_bins(y_te, p)
        np.savez(OUT / f"iot_curve_{model_name.replace(' ', '_').lower()}.npz",
                 fpr=fpr, tpr=tpr, precision=prec, recall=rec)
    with open(OUT / 'iot_pr_calibration.json', 'w') as f:
        json.dump({'pr': pr, 'calibration': cal}, f)

    result = {
        'fd': 'FD001', 'n_cycles': int(len(df)), 'n_engines': int(df.unit.nunique()),
        'n_sensors': len(sensors), 'sensors': sensors, 'n_features': len(names),
        'orders': {k: list(v) for k, v in orders.items()},
        'train_units': [int(u) for u in tr_units], 'test_units': [int(u) for u in te_units],
        'train_counts': [int((y[m_tr] == 0).sum()), int((y[m_tr] == 1).sum())],
        'test_counts': [int((y_te == 0).sum()), int((y_te == 1).sum())],
        'cv_48': cv48, 'cv_45': cv45, 'holdout_48': hold48, 'holdout_45': hold45,
        'arima_diagnostics': arima_sum, 'permutation_top': perm,
        'versions': {'python': os.sys.version.split()[0],
                     'sklearn': __import__('sklearn').__version__,
                     'xgboost': xgb.__version__,
                     'statsmodels': __import__('statsmodels').__version__},
    }
    with open(OUT / 'iot_fd001_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=float)
    log(f"FD001 holdout 48f: RF F1={hold48['Random Forest']['macro_f1']:.4f} "
        f"AUC={hold48['Random Forest']['roc_auc']:.4f}; "
        f"XGB F1={hold48['XGBoost']['macro_f1']:.4f} AUC={hold48['XGBoost']['roc_auc']:.4f}")
    log(f"FD001 holdout 45f: RF F1={hold45['Random Forest']['macro_f1']:.4f}; "
        f"XGB F1={hold45['XGBoost']['macro_f1']:.4f}")
    return result


def run_external(fd, orders):
    log(f"=== External validation: {fd} ===")
    df, sensors = load_fd(fd)
    units = df.unit.unique()
    tr_units, te_units = train_test_split(units, test_size=0.3, random_state=SEED)
    train_mask = df.unit.isin(tr_units).values
    regime = fd in ('FD002', 'FD004')
    feats, names, spec = build_features(df, sensors, train_mask, orders, regime, fd)
    arima_sum, _ = arima_diagnostics(df, orders, tr_units, fd)
    X = feats[names].values
    y = feats['fault'].values
    m_tr = feats.unit.isin(tr_units).values
    m_te = ~m_tr
    hold, models, preds = evaluate_holdout(
        X[m_tr], y[m_tr], X[m_te], y[m_te], feats.loc[m_te, 'unit'].values, names)
    result = {
        'fd': fd, 'n_cycles': int(len(df)), 'n_engines': int(df.unit.nunique()),
        'n_sensors': len(sensors), 'sensors': sensors, 'n_features': len(names),
        'regime_aware': bool(regime),
        'orders': {k: list(v) for k, v in orders.items()},
        'train_units': [int(u) for u in tr_units], 'test_units': [int(u) for u in te_units],
        'train_counts': [int((y[m_tr] == 0).sum()), int((y[m_tr] == 1).sum())],
        'test_counts': [int((y[m_te] == 0).sum()), int((y[m_te] == 1).sum())],
        'holdout': hold, 'arima_diagnostics': arima_sum,
    }
    with open(OUT / f'iot_{fd}_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=float)
    log(f"{fd} holdout: RF F1={hold['Random Forest']['macro_f1']:.4f} "
        f"AUC={hold['Random Forest']['roc_auc']:.4f}; "
        f"XGB F1={hold['XGBoost']['macro_f1']:.4f} AUC={hold['XGBoost']['roc_auc']:.4f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='all', choices=['all', 'FD001', 'external'])
    args = ap.parse_args()
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('')
    df, _ = load_fd('FD001')
    units = df.unit.unique()
    tr_units, _ = train_test_split(units, test_size=0.3, random_state=SEED)
    orders, order_table = select_arima_orders(df, tr_units)
    order_table.to_csv(OUT / 'iot_arima_order_search.csv', index=False)
    results = {}
    if args.only in ('all', 'FD001'):
        results['FD001'] = run_fd001(orders)
    if args.only in ('all', 'external'):
        for fd in ['FD002', 'FD003', 'FD004']:
            results[fd] = run_external(fd, orders)
    with open(OUT / 'iot_revision_summary.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    log('IoT revision complete.')


if __name__ == '__main__':
    main()
