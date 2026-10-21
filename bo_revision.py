"""Revised backorder pipeline with a leakage-free evaluation protocol.

- 42 configurations screened on an internal validation split
- top-3 configurations + baselines re-evaluated with 5-fold CV
- official test split touched once, with bootstrap CIs, McNemar and DeLong tests
- PR/calibration, cost-sensitive thresholds, permutation importance, SHAP
- base-rate-shift analysis
"""
import warnings
warnings.filterwarnings('ignore')
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb

from stats_utils import (binary_metrics, bootstrap_metrics, calibration_bins,
                         choose_cost_threshold, cost_at_threshold, delong_test,
                         mcnemar_test)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
MODELS = OUT / "models"
MODELS.mkdir(exist_ok=True)
SEED = 42
N_JOBS = int(os.environ.get('BO_N_JOBS', '4'))
TARGET = 'went_on_backorder'
BOOL_FLAGS = ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk',
              'stop_auto_buy', 'rev_stop']
LOG_PATH = OUT / "bo_revision.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line + "\n")


def clean(df):
    df = df.copy().drop(columns=['sku'], errors='ignore')
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].map({'Yes': 1, 'No': 0})
    for c in BOOL_FLAGS:
        df[c] = df[c].map({'Yes': 1, 'No': 0})
    return df.reset_index(drop=True)


def fit_medians(df):
    return {c: float(df[c].median()) for c in df.columns if c != TARGET}


def impute(df, medians):
    df = df.copy()
    for c, v in medians.items():
        df[c] = df[c].fillna(v)
    return df


def resampler(name):
    if name == 'Baseline':
        return None
    return {
        'SMOTE': SMOTE(random_state=SEED),
        'ADASYN': ADASYN(random_state=SEED),
        'Borderline-SMOTE': BorderlineSMOTE(random_state=SEED),
        'SMOTE+Tomek': SMOTETomek(random_state=SEED),
        'RandUnder': RandomUnderSampler(random_state=SEED),
    }[name]


def model_for(name, ytr):
    if name == 'RF':
        return RandomForestClassifier(n_estimators=300, max_depth=15,
                                      random_state=SEED, n_jobs=N_JOBS)
    if name == 'RF-bal':
        return RandomForestClassifier(n_estimators=300, max_depth=15,
                                      random_state=SEED, n_jobs=N_JOBS,
                                      class_weight='balanced')
    if name == 'XGB':
        pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
        return xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 scale_pos_weight=pw, random_state=SEED,
                                 eval_metric='logloss', verbosity=0, n_jobs=N_JOBS)
    if name == 'LR-bal':
        return LogisticRegression(max_iter=1000, class_weight='balanced',
                                  random_state=SEED)
    if name == 'LR':
        return LogisticRegression(max_iter=1000, random_state=SEED)
    if name == 'NB':
        return GaussianNB()
    if name == 'DT':
        return DecisionTreeClassifier(max_depth=12, class_weight='balanced',
                                      random_state=SEED)
    raise ValueError(name)


def fit_one(rn, mn, Xtr, ytr, Xev):
    rs = resampler(rn)
    if rs is not None:
        Xr, yr = rs.fit_resample(Xtr, ytr)
    else:
        Xr, yr = Xtr, ytr
    model = model_for(mn, yr)
    model.fit(Xr, yr)
    return model, model.predict_proba(Xev)[:, 1]


def main():
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('')
    log('Loading backorder data')
    tr = clean(pd.read_csv(DATA / 'Training_BOP.csv', low_memory=False))
    te = clean(pd.read_csv(DATA / 'Testing_BOP.csv', low_memory=False))
    feat_cols = [c for c in tr.columns if c != TARGET]
    rng = np.random.default_rng(SEED)
    kept = train_test_split(np.arange(len(tr)), train_size=300000,
                            random_state=SEED, stratify=tr[TARGET].values)[0]
    gdf = tr.iloc[kept].reset_index(drop=True)
    y = gdf[TARGET].values
    X_raw = gdf[feat_cols].copy()
    log(f'Grid subset: {len(gdf)} rows, {y.sum()} positives ({y.mean()*100:.3f}%)')
    log(f'Official test: {len(te)} rows, {int(te[TARGET].sum())} positives '
        f'({te[TARGET].mean()*100:.3f}%)')

    tr_idx, va_idx = train_test_split(np.arange(len(gdf)), test_size=0.2,
                                      random_state=SEED, stratify=y)
    med = fit_medians(gdf.iloc[tr_idx])
    Xtr_raw = impute(gdf.iloc[tr_idx], med)[feat_cols].values
    Xva_raw = impute(gdf.iloc[va_idx], med)[feat_cols].values
    ytr, yva = y[tr_idx], y[va_idx]
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr, Xva = scaler.transform(Xtr_raw), scaler.transform(Xva_raw)
    log(f'Screening split: train={len(Xtr)} ({ytr.sum()} pos), '
        f'val={len(Xva)} ({yva.sum()} pos)')

    resamplers = ['Baseline', 'SMOTE', 'ADASYN', 'Borderline-SMOTE',
                  'SMOTE+Tomek', 'RandUnder']
    models = ['RF', 'RF-bal', 'XGB', 'LR-bal', 'LR', 'NB', 'DT']
    grid = []
    val_probs = {}
    grid_path = OUT / 'bo_validation_grid.csv'
    if grid_path.exists():
        gdf_res = pd.read_csv(grid_path).sort_values('macro_f1', ascending=False)
        grid = gdf_res.to_dict('records')
        top = gdf_res.head(3)[['resamp', 'model']].apply(tuple, axis=1).tolist()
        log(f'Reusing cached validation grid ({len(grid)} configs)')
    else:
        for rn in resamplers:
            for mn in models:
                t0 = time.time()
                try:
                    _, p = fit_one(rn, mn, Xtr, ytr, Xva)
                    m = binary_metrics(yva, p)
                    m.update({'resamp': rn, 'model': mn})
                    grid.append(m)
                    val_probs[(rn, mn)] = p
                    log(f'  val {rn:16s} {mn:7s} F1={m["macro_f1"]:.4f} '
                        f'AUC={m["roc_auc"]:.4f} AP={m["ap"]:.4f} '
                        f'P={m["precision"]:.4f} R={m["recall"]:.4f} '
                        f'[{time.time()-t0:.1f}s]')
                except Exception as e:
                    log(f'  val {rn} {mn}: FAIL {e}')
        gdf_res = pd.DataFrame(grid).sort_values('macro_f1', ascending=False)
        gdf_res.to_csv(grid_path, index=False)
        top = gdf_res.head(3)[['resamp', 'model']].apply(tuple, axis=1).tolist()
    log(f'Top-3 by validation macro-F1: {top}')

    final_set = list(dict.fromkeys(top + [('Baseline', 'RF'), ('Baseline', 'RF-bal'),
                                          ('Baseline', 'XGB')]))

    # 5-fold CV on the 300k subset (resampling inside folds only)
    log('Running 5-fold CV for top-3 and baselines')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    oof = {}
    for rn, mn in final_set:
        oof[(rn, mn)] = np.full(len(gdf), np.nan)
        for fold, (trn, val) in enumerate(cv.split(X_raw, y)):
            t0 = time.time()
            med_f = fit_medians(gdf.iloc[trn])
            Xf_tr = impute(gdf.iloc[trn], med_f)[feat_cols].values
            Xf_va = impute(gdf.iloc[val], med_f)[feat_cols].values
            sc = StandardScaler().fit(Xf_tr)
            Xf_tr, Xf_va = sc.transform(Xf_tr), sc.transform(Xf_va)
            _, p = fit_one(rn, mn, Xf_tr, y[trn], Xf_va)
            oof[(rn, mn)][val] = p
            m = binary_metrics(y[val], p)
            m.update({'resamp': rn, 'model': mn, 'fold': fold})
            cv_rows.append(m)
            log(f'  cv fold {fold} {rn:16s} {mn:7s} F1={m["macro_f1"]:.4f} '
                f'[{time.time()-t0:.1f}s]')
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(OUT / 'bo_cv_folds.csv', index=False)
    cv_summary = {}
    for (rn, mn), g in cv_df.groupby(['resamp', 'model']):
        key = f'{rn}|{mn}'
        cv_summary[key] = {}
        for k in ['acc', 'macro_f1', 'roc_auc', 'ap', 'precision', 'recall']:
            cv_summary[key][k] = {'mean': float(g[k].mean()),
                                  'std': float(g[k].std(ddof=1))}
        cv_summary[key]['oof'] = binary_metrics(y, oof[(rn, mn)])

    # Final refit on all 300k, single test evaluation
    log('Final refit on 300k and single official-test evaluation')
    med_full = fit_medians(gdf)
    Xfull_raw = impute(gdf, med_full)[feat_cols].values
    Xte_raw = impute(te, med_full)[feat_cols].values
    sc_full = StandardScaler().fit(Xfull_raw)
    Xfull = sc_full.transform(Xfull_raw)
    Xte = sc_full.transform(Xte_raw)
    yte = te[TARGET].values
    final = {}
    final_probs = {}
    final_models = {}
    for rn, mn in final_set:
        t0 = time.time()
        model, p = fit_one(rn, mn, Xfull, y, Xte)
        final_probs[(rn, mn)] = p
        final_models[(rn, mn)] = model
        m = binary_metrics(yte, p)
        m['bootstrap'] = bootstrap_metrics(yte, p, n_boot=1000)
        final[f'{rn}|{mn}'] = m
        log(f'  test {rn:16s} {mn:7s} F1={m["macro_f1"]:.4f} '
            f'AUC={m["roc_auc"]:.4f} AP={m["ap"]:.4f} P={m["precision"]:.4f} '
            f'R={m["recall"]:.4f} [{time.time()-t0:.1f}s]')

    champ_key = f'{top[0][0]}|{top[0][1]}'
    xgb_key = 'Baseline|XGB'
    rfbal_key = 'Baseline|RF-bal'
    rf_key = 'Baseline|RF'
    pch = final_probs[top[0]]
    comparisons = {}
    for other, label in [(('Baseline', 'XGB'), 'XGBoost-cost'),
                         (('Baseline', 'RF-bal'), 'RF-bal'),
                         (('Baseline', 'RF'), 'RF')]:
        po = final_probs[other]
        comparisons[label] = {
            'mcnemar': mcnemar_test(yte, pch, po),
            'delong': delong_test(yte, pch, po),
        }
        log(f'  {label}: McNemar p={comparisons[label]["mcnemar"]["pvalue"]:.4g}, '
            f'DeLong p={comparisons[label]["delong"]["pvalue"]:.4g}')

    # Threshold / cost analysis on out-of-fold probabilities
    log('Cost-sensitive threshold analysis on OOF predictions')
    thr_rows = []
    for label, key in [('champion', top[0]), ('XGBoost-cost', ('Baseline', 'XGB'))]:
        yo = y
        po = oof[key]
        for ratio in [1, 5, 10, 20, 50]:
            best = choose_cost_threshold(yo, po, fn_cost=ratio, fp_cost=1)
            test = cost_at_threshold(yte, final_probs[key], best['threshold'],
                                     fn_cost=ratio, fp_cost=1)
            thr_rows.append({'model': label, 'criterion': f'cost_FN{ratio}_FP1',
                             'threshold': best['threshold'], 'val_cost': best['cost'],
                             'val_precision': best['precision'], 'val_recall': best['recall'],
                             'test_cost': test['cost'], 'test_precision': test['precision'],
                             'test_recall': test['recall']})
        # macro-F1-optimal threshold
        best_t, best_f1 = 0.5, -1
        for t in np.linspace(0.01, 0.99, 99):
            f1 = binary_metrics(yo, po, t)['macro_f1']
            if f1 > best_f1:
                best_t, best_f1 = t, f1
        test = binary_metrics(yte, final_probs[key], best_t)
        thr_rows.append({'model': label, 'criterion': 'macro_f1',
                         'threshold': best_t, 'val_cost': np.nan,
                         'val_precision': np.nan, 'val_recall': np.nan,
                         'test_cost': np.nan, 'test_precision': test['precision'],
                         'test_recall': test['recall'],
                         'test_macro_f1': test['macro_f1']})
    pd.DataFrame(thr_rows).to_csv(OUT / 'bo_thresholds.csv', index=False)

    # Calibration (uncalibrated + Platt on OOF)
    log('Calibration analysis')
    cal = {}
    for label, key in [('champion', top[0]), ('XGBoost-cost', ('Baseline', 'XGB'))]:
        p = final_probs[key]
        po = oof[key]
        lr = LogisticRegression().fit(po.reshape(-1, 1), y)
        pc = lr.predict_proba(p.reshape(-1, 1))[:, 1]
        cal[label] = {
            'uncalibrated': {'brier': float(binary_metrics(yte, p)['brier']),
                             'bins': calibration_bins(yte, p)},
            'platt': {'brier': float(binary_metrics(yte, pc)['brier']),
                      'bins': calibration_bins(yte, pc)},
        }
    with open(OUT / 'bo_calibration.json', 'w') as f:
        json.dump(cal, f)

    # PR curves
    curves = {}
    for key, p in final_probs.items():
        prec, rec, _ = precision_recall_curve(yte, p)
        fpr, tpr, _ = roc_curve(yte, p)
        curves[f'{key[0]}|{key[1]}'] = {'precision': prec.tolist(), 'recall': rec.tolist(),
                                        'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
    with open(OUT / 'bo_curves.json', 'w') as f:
        json.dump(curves, f)

    # Permutation importance (50k stratified test sample)
    log('Permutation importance')
    idx = train_test_split(np.arange(len(yte)), train_size=50000,
                           random_state=SEED, stratify=yte)[0]
    perm = {}
    for label, key in [('champion', top[0]), ('XGBoost-cost', ('Baseline', 'XGB'))]:
        model = final_models[key]
        r = permutation_importance(model, Xte[idx], yte[idx], n_repeats=5,
                                   random_state=SEED, scoring='roc_auc', n_jobs=N_JOBS)
        d = pd.DataFrame({'feature': feat_cols, 'mean': r.importances_mean,
                          'std': r.importances_std}).sort_values('mean', ascending=False)
        d.to_csv(OUT / f'bo_permutation_{label}.csv', index=False)
        perm[label] = d.head(15).to_dict('records')

    # SHAP for XGBoost
    shap_top = None
    try:
        import shap
        model = final_models[('Baseline', 'XGB')]
        rng2 = np.random.default_rng(SEED)
        sidx = rng2.choice(len(Xte), size=min(5000, len(Xte)), replace=False)
        expl = shap.TreeExplainer(model)
        sv = expl.shap_values(Xte[sidx])
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        d = pd.DataFrame({'feature': feat_cols,
                          'mean_abs_shap': np.abs(sv).mean(axis=0)}).sort_values(
            'mean_abs_shap', ascending=False)
        d.to_csv(OUT / 'bo_shap_xgb.csv', index=False)
        shap_top = d.head(15).to_dict('records')
        log('  SHAP saved')
    except Exception as e:
        log(f'  SHAP failed: {e}')

    # Base-rate shift correction
    pi_tr, pi_te = float(y.mean()), float(yte.mean())
    odds = (pi_te / (1 - pi_te)) / (pi_tr / (1 - pi_tr))
    p = final_probs[top[0]]
    p_adj = (p * odds) / (p * odds + (1 - p))
    base = {'train_rate': pi_tr, 'test_rate': pi_te,
            'raw': binary_metrics(yte, p), 'prior_corrected': binary_metrics(yte, p_adj)}

    # Save models, predictions, and summary
    joblib.dump({'models': final_models, 'scaler': sc_full, 'medians': med_full,
                 'features': feat_cols, 'champion': top[0], 'val_grid': grid},
                MODELS / 'bo_final.joblib')
    pred_df = pd.DataFrame({'y': yte})
    for (rn, mn), p in final_probs.items():
        pred_df[f'{rn}|{mn}'] = p
    pred_df.to_csv(OUT / 'bo_final_predictions.csv', index=False)
    oof_df = pd.DataFrame({'y': y})
    for (rn, mn), p in oof.items():
        oof_df[f'{rn}|{mn}'] = p
    oof_df.to_csv(OUT / 'bo_oof_predictions.csv', index=False)

    result = {
        'n_train_subset': int(len(gdf)), 'n_train_split': int(len(tr_idx)),
        'n_val_split': int(len(va_idx)), 'n_test': int(len(te)),
        'train_rate': float(y.mean()), 'test_rate': float(yte.mean()),
        'top3': [list(x) for x in top],
        'validation_grid': grid,
        'cv': cv_summary,
        'final': final,
        'comparisons': comparisons,
        'calibration': cal,
        'permutation_top': perm, 'shap_top': shap_top,
        'base_rate': base,
        'versions': {'python': os.sys.version.split()[0],
                     'sklearn': __import__('sklearn').__version__,
                     'xgboost': xgb.__version__,
                     'imblearn': __import__('imblearn').__version__,
                     'statsmodels': __import__('statsmodels').__version__},
    }
    with open(OUT / 'bo_revision.json', 'w') as f:
        json.dump(result, f, indent=2, default=float)
    log('Backorder revision complete.')


if __name__ == '__main__':
    main()
