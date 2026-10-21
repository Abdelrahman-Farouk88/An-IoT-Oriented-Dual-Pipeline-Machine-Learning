"""Statistical utilities for the revision: bootstrap CIs, McNemar, DeLong, cost curves."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)


def binary_metrics(y, p, threshold=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'threshold': float(threshold),
        'acc': float(accuracy_score(y, pred)),
        'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
        'roc_auc': float(roc_auc_score(y, p)),
        'ap': float(average_precision_score(y, p)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'brier': float(brier_score_loss(y, p)),
        'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn),
    }


def _bootstrap_indices(y, rng, groups=None):
    y = np.asarray(y)
    if groups is None:
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        return np.concatenate([rng.choice(pos, len(pos), replace=True),
                               rng.choice(neg, len(neg), replace=True)])
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    pick = rng.choice(uniq, len(uniq), replace=True)
    return np.concatenate([np.flatnonzero(groups == g) for g in pick])


def bootstrap_metrics(y, p, threshold=0.5, n_boot=1000, seed=42, groups=None):
    """Stratified row bootstrap, or cluster bootstrap when groups are supplied."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    keys = ['acc', 'macro_f1', 'roc_auc', 'ap', 'precision', 'recall', 'brier']
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = _bootstrap_indices(y, rng, groups)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        m = binary_metrics(yy, p[idx], threshold)
        for k in keys:
            draws[k].append(m[k])
    out = {}
    for k in keys:
        arr = np.asarray(draws[k])
        out[k] = {
            'mean': float(arr.mean()),
            'ci_low': float(np.percentile(arr, 2.5)),
            'ci_high': float(np.percentile(arr, 97.5)),
        }
    return out


def mcnemar_test(y, p1, p2, threshold=0.5):
    from statsmodels.stats.contingency_tables import mcnemar
    y = np.asarray(y).astype(int)
    c1 = ((np.asarray(p1) >= threshold).astype(int) == y)
    c2 = ((np.asarray(p2) >= threshold).astype(int) == y)
    table = [[int(np.sum(c1 & c2)), int(np.sum(c1 & ~c2))],
             [int(np.sum(~c1 & c2)), int(np.sum(~c1 & ~c2))]]
    res = mcnemar(table, exact=False, correction=True)
    return {'statistic': float(res.statistic), 'pvalue': float(res.pvalue),
            'table': table}


def _compute_midrank(x):
    J = np.argsort(x)
    Z = np.asarray(x)[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds, label_1_count):
    """preds: (k, n) with positives in the first columns; returns AUCs and covariance."""
    m = label_1_count
    n = preds.shape[1] - m
    pos = preds[:, :m]
    neg = preds[:, m:]
    k = preds.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = np.atleast_2d(sx) / m + np.atleast_2d(sy) / n
    return aucs, cov


def delong_test(y, p1, p2):
    y = np.asarray(y).astype(int)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    idx = np.concatenate([np.flatnonzero(y == 1), np.flatnonzero(y == 0)])
    preds = np.vstack([p1[idx], p2[idx]])
    m = int(y.sum())
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-15))
    from scipy.stats import norm
    p = 2 * norm.sf(abs(z))
    return {'auc1': float(aucs[0]), 'auc2': float(aucs[1]),
            'z': float(z), 'pvalue': float(p)}


def calibration_bins(y, p, n_bins=10):
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy='quantile')
    return {'prob_true': frac_pos.tolist(), 'prob_pred': mean_pred.tolist()}


def cost_at_threshold(y, p, threshold, fn_cost=1.0, fp_cost=1.0):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    return {'threshold': float(threshold), 'FN': fn, 'FP': fp, 'TP': tp, 'TN': tn,
            'cost': float(fn * fn_cost + fp * fp_cost),
            'precision': float(tp / max(tp + fp, 1)),
            'recall': float(tp / max(tp + fn, 1))}


def choose_cost_threshold(y, p, fn_cost=1.0, fp_cost=1.0, grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    rows = [cost_at_threshold(y, p, t, fn_cost, fp_cost) for t in grid]
    best = min(rows, key=lambda r: (r['cost'], -r['recall']))
    return best
