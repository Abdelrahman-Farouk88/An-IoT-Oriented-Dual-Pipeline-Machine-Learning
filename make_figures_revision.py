"""Regenerate data-driven figures from saved revision artefacts only."""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent
CMAP = ROOT / "CMaps"
OUT = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.bbox': 'tight',
})
C = {'rf': '#0072B2', 'xgb': '#D55E00', 'grey': '#555555',
     'green': '#009E73', 'purple': '#CC79A7', 'yellow': '#E69F00'}


def save(fig, name):
    fig.savefig(FIG / name, dpi=300)
    plt.close(fig)
    print('saved', name)


def load_json(name):
    with open(OUT / name, encoding='utf-8') as f:
        return json.load(f)


def cmap_counts(m):
    return np.array([[m['TN'], m['FP']], [m['FN'], m['TP']]])


def fig_arima():
    cols = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i:02d}' for i in range(1, 22)]
    df = pd.read_csv(CMAP / 'train_FD001.txt', sep=r'\s+', header=None, names=cols)
    res = load_json('iot_fd001_results.json')
    orders = res['orders']
    ch = [('s03', 'Temperature (s03)', C['xgb']),
          ('s07', 'Compressor pressure (s07)', C['rf']),
          ('s17', 'Coolant bleed (s17)', C['green'])]
    sub = df[df.unit == 69].sort_values('cycle')
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.2), sharex=True)
    for j, (s, name, col) in enumerate(ch):
        y = sub[s].values.astype(float)
        fit = ARIMA(y, order=tuple(orders[s]), trend='c').fit()
        axes[0, j].plot(sub.cycle, y, color=col, lw=1.1)
        axes[0, j].axvspan(len(y) - 30, len(y), color='red', alpha=0.12)
        axes[0, j].set_title(name)
        axes[1, j].plot(sub.cycle, fit.resid, color=C['grey'], lw=0.8)
        axes[1, j].axhline(0, color='black', lw=0.6, ls='--')
        axes[1, j].axvspan(len(y) - 30, len(y), color='red', alpha=0.12)
        axes[1, j].set_xlabel('Cycle')
        axes[0, j].set_ylabel('Sensor value')
        axes[1, j].set_ylabel('Residual')
    fig.tight_layout()
    save(fig, 'fig_arima.pdf')


def fig_smote():
    res = load_json('iot_fd001_results.json')
    neg, pos = res['train_counts']
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    ax.bar([0, 1], [neg, pos], color=[C['rf'], C['xgb']], width=0.55)
    ax.bar([2.6, 3.6], [neg, neg], color=[C['rf'], C['xgb']], width=0.55, alpha=0.55)
    ax.set_xticks([0, 1, 2.6, 3.6])
    ax.set_xticklabels(['Normal', 'Fault', 'Normal\n(SMOTE)', 'Fault\n(SMOTE)'])
    ax.set_ylabel('Training cycles')
    ax.set_title('FD001 engine-wise training split')
    fig.tight_layout()
    save(fig, 'fig_smote_balance.pdf')


def fig_importance():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8))
    for ax, key, title, col in [(axes[0], 'random_forest', 'Random Forest', C['rf']),
                                (axes[1], 'xgboost', 'XGBoost', C['xgb'])]:
        d = pd.read_csv(OUT / f'iot_permutation_{key}.csv').head(15).iloc[::-1]
        ax.barh(d.feature, d['mean'], xerr=d['std'], color=col, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel('Mean ROC-AUC decrease')
        ax.tick_params(axis='y', labelsize=7)
    fig.tight_layout()
    save(fig, 'fig_feature_importance.pdf')


def fig_confusion():
    res = load_json('iot_fd001_results.json')
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.7))
    for ax, key, title in [(axes[0], 'Random Forest', 'Random Forest'),
                           (axes[1], 'XGBoost', 'XGBoost')]:
        m = res['holdout_48'][key]
        cm = cmap_counts(m)
        im = ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Normal', 'Fault'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['Normal', 'Fault'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.set_title(title)
    fig.tight_layout()
    save(fig, 'fig_confusion.pdf')


def fig_roc():
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    for key, label, col in [('random_forest', 'Random Forest', C['rf']),
                            ('xgboost', 'XGBoost', C['xgb'])]:
        d = np.load(OUT / f'iot_curve_{key}.npz')
        ax.plot(d['fpr'], d['tpr'], color=col, lw=1.4, label=label)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.7)
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
    ax.legend(loc='lower right'); ax.set_title('FD001 holdout ROC')
    fig.tight_layout()
    save(fig, 'fig_roc.pdf')


def fig_pca():
    d = pd.read_csv(OUT / 'iot_features_FD001.csv')
    feat = [c for c in d.columns if c not in ('unit', 'cycle', 'RUL', 'fault')]
    X = StandardScaler().fit_transform(d[feat].values)
    Z = PCA(n_components=2, random_state=42).fit_transform(X)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(d), size=min(6000, len(d)), replace=False)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for lab, col, name in [(0, C['rf'], 'Normal'), (1, C['xgb'], 'Fault (RUL<=30)')]:
        m = d.fault.values[idx] == lab
        ax.scatter(Z[idx][m, 0], Z[idx][m, 1], s=4, alpha=0.35, color=col, label=name)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(markerscale=3)
    ax.set_title('FD001 48-feature PCA')
    fig.tight_layout()
    save(fig, 'fig_pca.pdf')


def fig_iot_external():
    fds = ['FD001', 'FD002', 'FD003', 'FD004']
    data = [load_json('iot_fd001_results.json')] + [load_json(f'iot_{fd}_results.json') for fd in fds[1:]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    x = np.arange(len(fds)); w = 0.36
    for ax, metric, title in [(axes[0], 'macro_f1', 'Macro F1'),
                              (axes[1], 'roc_auc', 'ROC-AUC')]:
        rf = [d['holdout_48']['Random Forest'][metric] for d in data[:1]] + \
             [d['holdout']['Random Forest'][metric] for d in data[1:]]
        xg = [d['holdout_48']['XGBoost'][metric] for d in data[:1]] + \
             [d['holdout']['XGBoost'][metric] for d in data[1:]]
        ax.bar(x - w / 2, rf, w, color=C['rf'], label='Random Forest')
        ax.bar(x + w / 2, xg, w, color=C['xgb'], label='XGBoost')
        ax.set_xticks(x); ax.set_xticklabels(fds); ax.set_title(title)
        ax.set_ylim(0.5, 1.0)
        axes[0].set_ylabel('Score')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, 'fig_iot_external.pdf')


def fig_bo_class():
    d = load_json('bo_revision.json')
    n = d['n_train_subset']; pos = int(round(n * d['train_rate']))
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    ax.bar([0, 1], [n - pos, pos], color=[C['rf'], C['xgb']], width=0.55)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['No backorder', 'Backorder'])
    ax.set_ylabel('Rows'); ax.set_title(f'300K stratified subset ({pos} positives)')
    fig.tight_layout()
    save(fig, 'fig_bo_classbalance.pdf')


def fig_bo_roc():
    d = load_json('bo_revision.json')
    curves = load_json('bo_curves.json')
    top = d['top3'][0]
    keys = [f'{top[0]}|{top[1]}', 'Baseline|XGB', 'Baseline|RF-bal', 'Baseline|RF']
    labels = [f'{top[0]} + {top[1]} (champion)', 'XGBoost cost-weighted',
              'RF balanced', 'RF baseline']
    cols = [C['xgb'], C['rf'], C['green'], C['grey']]
    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    for k, lab, col in zip(keys, labels, cols):
        if k not in curves:
            continue
        ax.plot(curves[k]['fpr'], curves[k]['tpr'], lw=1.4, color=col, label=lab)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.7)
    ax.set_xlabel('False positive rate'); ax.set_ylabel('True positive rate')
    ax.legend(loc='lower right', fontsize=7); ax.set_title('Official test ROC')
    fig.tight_layout()
    save(fig, 'fig_bo_roc.pdf')


def fig_bo_pr_cal():
    d = load_json('bo_revision.json')
    curves = load_json('bo_curves.json')
    cal = load_json('bo_calibration.json')
    top = d['top3'][0]
    keys = [(f'{top[0]}|{top[1]}', 'Champion', C['xgb']),
            ('Baseline|XGB', 'XGBoost', C['rf'])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for k, lab, col in keys:
        c = curves[k]
        axes[0].plot(c['recall'], c['precision'], color=col, lw=1.4,
                     label=f"{lab} (AP={d['final'][k]['ap']:.3f})")
    axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision')
    axes[0].legend(fontsize=7); axes[0].set_title('Precision-recall')
    for k, key, lab, col in [('champion', f'{top[0]}|{top[1]}', 'Champion', C['xgb']),
                             ('XGBoost-cost', 'Baseline|XGB', 'XGBoost', C['rf'])]:
        b = cal[k]['uncalibrated']['bins']
        axes[1].plot(b['prob_pred'], b['prob_true'], 'o-', ms=3, color=col,
                     label=f"{lab} (Brier={d['final'][key]['brier']:.3f})")
    axes[1].plot([0, 1], [0, 1], 'k--', lw=0.7)
    axes[1].set_xlabel('Mean predicted probability'); axes[1].set_ylabel('Observed fraction')
    axes[1].legend(fontsize=7); axes[1].set_title('Calibration')
    fig.tight_layout()
    save(fig, 'fig_bo_pr_calibration.pdf')


def fig_bo_cv():
    d = pd.read_csv(OUT / 'bo_cv_folds.csv')
    order = [('ADASYN', 'XGB'), ('Borderline-SMOTE', 'XGB'), ('SMOTE', 'XGB'),
             ('Baseline', 'XGB'), ('Baseline', 'RF-bal'), ('Baseline', 'RF')]
    data, labels = [], []
    for rn, mn in order:
        g = d[(d.resamp == rn) & (d.model == mn)]
        if len(g):
            data.append(g.macro_f1.values)
            labels.append(f'{rn}\n{mn}')
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    bp = ax.boxplot(data, patch_artist=True, widths=0.6)
    for patch in bp['boxes']:
        patch.set_facecolor(C['rf']); patch.set_alpha(0.35)
    ax.set_xticklabels(labels, fontsize=6.5, rotation=25, ha='right')
    ax.set_ylabel('Fold macro F1'); ax.set_title('Five-fold cross-validation')
    fig.tight_layout()
    save(fig, 'fig_bo_cv.pdf')


def fig_bo_top10():
    d = pd.read_csv(OUT / 'bo_validation_grid.csv').head(10).iloc[::-1]
    labels = [f'{r.model} + {r.resamp}' for r in d.itertuples()]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.barh(labels, d.macro_f1, color=C['rf'], alpha=0.85)
    ax.set_xlabel('Validation macro F1')
    ax.set_title('Top-10 configurations (validation)')
    ax.tick_params(axis='y', labelsize=7)
    fig.tight_layout()
    save(fig, 'fig_bo_top10.pdf')


def fig_bo_cost():
    oof = pd.read_csv(OUT / 'bo_oof_predictions.csv')
    d = load_json('bo_revision.json')
    top = d['top3'][0]
    keys = [(f'{top[0]}|{top[1]}', 'Champion', C['xgb']),
            ('Baseline|XGB', 'XGBoost', C['rf'])]
    thr = pd.read_csv(OUT / 'bo_thresholds.csv')
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    grid = np.linspace(0.01, 0.99, 99)
    for k, lab, col in keys:
        p = oof[k].values
        y = oof['y'].values
        costs = []
        for t in grid:
            pred = (p >= t).astype(int)
            fn = int(((y == 1) & (pred == 0)).sum())
            fp = int(((y == 0) & (pred == 1)).sum())
            costs.append((10 * fn + fp) / len(y))
        ax.plot(grid, costs, color=col, lw=1.4, label=lab)
        row = thr[(thr.model == ('champion' if lab == 'Champion' else 'XGBoost-cost')) &
                  (thr.criterion == 'cost_FN10_FP1')]
        if len(row):
            ax.axvline(float(row.threshold.iloc[0]), color=col, ls='--', lw=0.8)
    ax.set_xlabel('Threshold'); ax.set_ylabel('Cost per record (FN:FP=10:1)')
    ax.legend(); ax.set_title('Out-of-fold cost sensitivity')
    fig.tight_layout()
    save(fig, 'fig_bo_cost.pdf')


if __name__ == '__main__':
    fig_arima()
    fig_smote()
    fig_importance()
    fig_confusion()
    fig_roc()
    fig_pca()
    fig_iot_external()
    fig_bo_class()
    fig_bo_roc()
    fig_bo_pr_cal()
    fig_bo_cv()
    fig_bo_top10()
    fig_bo_cost()
