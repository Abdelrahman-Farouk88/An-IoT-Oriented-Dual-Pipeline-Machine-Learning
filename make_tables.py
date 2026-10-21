"""Generate LaTeX tables directly from the revision result artefacts."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
TAB = ROOT / "tables"
TAB.mkdir(exist_ok=True)


def j(name):
    with open(OUT / name, encoding='utf-8') as f:
        return json.load(f)


def f4(x):
    return f"{float(x):.4f}"


def f3(x):
    return f"{float(x):.3f}"


def ci(m, key):
    b = m['bootstrap'][key]
    return f"{f4(m[key])} [{f3(b['ci_low'])}, {f3(b['ci_high'])}]"


def nice(rn, mn):
    r = {'Baseline': 'None', 'Borderline-SMOTE': 'Borderline-SMOTE'}.get(rn, rn)
    m = {'RF': 'RF', 'RF-bal': 'RF-balanced', 'XGB': 'XGBoost',
         'LR-bal': 'LR-balanced', 'LR': 'LR', 'NB': 'NB', 'DT': 'DT'}[mn]
    return f"{m} + {r}" if r != 'None' else m


def write(name, text):
    (TAB / name).write_text(text, encoding='utf-8')
    print('wrote', name)


iot = j('iot_fd001_results.json')
abl = j('iot_ablation_tests.json')
ext = {fd: j(f'iot_{fd}_results.json') for fd in ['FD002', 'FD003', 'FD004']}
bo = j('bo_revision.json')
top = bo['top3']
champ = f"{top[0][0]}|{top[0][1]}"
final_keys = list(dict.fromkeys([f"{r}|{m}" for r, m in top] +
                                ['Baseline|RF', 'Baseline|RF-bal', 'Baseline|XGB']))

# ARIMA diagnostics
rows = []
for ch, d in iot['arima_diagnostics'].items():
    s = {'Temperature': 's03', 'Compressor_Pressure': 's07',
         'Coolant_Bleed': 's17'}[ch]
    order = tuple(iot['orders'][s])
    rows.append(f"{ch.replace('_', ' ')} ({s}) & {order} & "
                f"{d['median_aic']:.1f} & {d['median_rmse']:.4f} & "
                f"{100*d['ljung_box_pass_rate']:.1f}\\% \\\\")
write('tab_iot_arima.tex', r"""\begin{table}[H]
\centering
\caption{ARIMA orders and residual diagnostics on the 70 FD001 training engines.
Orders were selected by median AIC on 12 training engines; diagnostics are reported as
medians across the training engines.}
\label{tab:arima}
\small
\begin{tabular}{lcccc}
\toprule
Channel & Order & Median AIC & Median RMSE & Ljung--Box pass \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# 45 vs 48 ablation
lines = []
for setname, cvkey in [('45 (rolling only)', 'cv_45'), ('48 (+ ARIMA)', 'cv_48')]:
    for model in ['Random Forest', 'XGBoost']:
        cv = iot[cvkey][model]
        h = iot['holdout_45' if setname.startswith('45') else 'holdout_48'][model]
        p = abl[model]
        if setname.startswith('45'):
            dp = f3(p['delong_45_vs_48']['pvalue'])
            mp = f3(p['mcnemar_45_vs_48']['pvalue'])
        else:
            dp, mp = '--', '--'
        lines.append(f"{setname} & {model} & {f4(cv['macro_f1']['mean'])} $\\pm$ "
                     f"{f3(cv['macro_f1']['std'])} & {f4(h['macro_f1'])} & "
                     f"{f4(h['roc_auc'])} & {dp} & {mp} \\\\")
write('tab_iot_ablation.tex', r"""\begin{table}[H]
\centering
\caption{Engine-grouped ablation of the three causal ARIMA residual features. The
45-feature matrix contains only the raw channels and their rolling statistics; the
48-feature matrix adds the residuals. \(p\)-values are paired tests between the two
feature sets for the same model.}
\label{tab:iot-ablation}
\small
\begin{tabular}{lcccccc}
\toprule
Features & Model & CV macro F1 & Holdout F1 & Holdout AUC & DeLong \(p\) & McNemar \(p\) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# IoT holdout
lines = []
for model in ['Random Forest', 'XGBoost']:
    m = iot['holdout_48'][model]
    lines.append(f"{model} & {f4(m['acc'])} & {f4(m['precision'])} & {f4(m['recall'])} & "
                 f"{ci(m, 'macro_f1')} & {f4(m['ap'])} & {ci(m, 'roc_auc')} & {f4(m['brier'])} \\\\")
write('tab_iot_holdout.tex', r"""\begin{table}[H]
\centering
\caption{IoT early-warning performance on the untouched FD001 engine holdout
(48 features, SMOTE inside training folds). Brackets are 95\% engine-cluster bootstrap
confidence intervals.}
\label{tab:iot-models}
\small
\begin{tabular}{lccccccc}
\toprule
Model & Acc. & Prec. & Rec. & Macro F1 [95\% CI] & AP & ROC-AUC [95\% CI] & Brier \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# External
lines = []
def add_ext(fd, label, model, m, feats):
    lines.append(f"{label} & {model} & {feats} & {f4(m['macro_f1'])} & {f4(m['roc_auc'])} & {f4(m['ap'])} \\\\")
for model in ['Random Forest', 'XGBoost']:
    add_ext('FD001', 'FD001 (holdout)', model, iot['holdout_48'][model], iot['n_features'])
    for fd in ['FD002', 'FD003', 'FD004']:
        r = ext[fd]
        add_ext(fd, f'{fd} ({r["n_engines"]} engines)', model, r['holdout'][model], r['n_features'])
write('tab_iot_external.tex', r"""\begin{table}[H]
\centering
\caption{External domain-shift stress tests. The same 15 FD001 sensor channels are
used in every dataset; FD002 and FD004 are scaled within six operating-regime clusters.
FD001 is the untouched 30-engine holdout; FD002--FD004 are 30\% engine holdouts.}
\label{tab:iot-external}
\small
\begin{tabular}{llcccc}
\toprule
Dataset & Model & Features & Macro F1 & ROC-AUC & AP \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# Full 42 validation table
d = pd.read_csv(OUT / 'bo_validation_grid.csv')
d = d.sort_values('macro_f1', ascending=False).reset_index(drop=True)
lines = []
for i, r in d.iterrows():
    mark = '\\champ{' if (r['resamp'], r['model']) == tuple(top[0]) else '{'
    close = '}' if mark != '{' else ''
    lines.append(f"{i+1} & {nice(r['resamp'], r['model'])} & {f4(r['macro_f1'])} & "
                 f"{f4(r['ap'])} & {f4(r['roc_auc'])} & {f4(r['precision'])} & {f4(r['recall'])} \\\\")
write('tab_bo_validation.tex', r"""\begin{center}
\begin{longtable}{clccccc}
\caption{Validation performance of all 42 backorder configurations. Configurations are
ranked by validation macro F1; the official test split was not used for this ranking.}
\label{tab:bo-full}\\
\toprule
Rank & Configuration & Macro F1 & AP & ROC-AUC & Prec. & Rec. \\
\midrule
\endfirsthead
\multicolumn{7}{c}{{\bfseries \tablename\ \thetable\ -- continued}} \\
\toprule
Rank & Configuration & Macro F1 & AP & ROC-AUC & Prec. & Rec. \\
\midrule
\endhead
\bottomrule
\endlastfoot
""" + "\n".join(lines) + r"""
\end{longtable}
\end{center}
""")

# CV table
lines = []
for key in final_keys:
    rn, mn = key.split('|')
    cv = bo['cv'][key]
    oof = cv['oof']
    mark = '\\champ{' if key == champ else '{'
    close = '}' if mark != '{' else ''
    lines.append(f"{nice(rn, mn)} & {f4(cv['macro_f1']['mean'])} $\\pm$ "
                 f"{f3(cv['macro_f1']['std'])} & {f4(cv['ap']['mean'])} & "
                 f"{f4(cv['roc_auc']['mean'])} & {f4(oof['macro_f1'])} \\\\")
write('tab_bo_cv.tex', r"""\begin{table}[H]
\centering
\caption{Five-fold cross-validation on the 300{,}000-row backorder subset for the
validation-selected top three configurations and the prespecified baselines. Resampling
is applied inside each training fold only. The final column is the macro F1 of the
pooled out-of-fold predictions.}
\label{tab:bo-cv}
\small
\begin{tabular}{lcccc}
\toprule
Configuration & CV macro F1 & CV AP & CV ROC-AUC & OOF macro F1 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# Final test table
lines = []
for key in final_keys:
    rn, mn = key.split('|')
    m = bo['final'][key]
    mark = '\\champ{' if key == champ else '{'
    close = '}' if mark != '{' else ''
    lines.append(f"{nice(rn, mn)} & {ci(m, 'macro_f1')} & {f4(m['ap'])} & "
                 f"{ci(m, 'roc_auc')} & {f4(m['precision'])} & {f4(m['recall'])} & "
                 f"{f4(m['brier'])} \\\\")
write('tab_bo_final.tex', r"""\begin{table}[H]
\centering
\caption{One-time evaluation on the official 242{,}075-row backorder test split.
The champion is the validation-selected configuration; the other rows are the remaining
two top-three configurations and the three prespecified baselines. Brackets are 95\% stratified bootstrap
confidence intervals.}
\label{tab:bo-champion}
\footnotesize
\begin{tabular}{lcccccc}
\toprule
Configuration & Macro F1 [95\% CI] & AP & ROC-AUC [95\% CI] & Prec. & Rec. & Brier \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# Threshold table
thr = pd.read_csv(OUT / 'bo_thresholds.csv')
lines = []
for _, r in thr.iterrows():
    label = 'Champion' if r['model'] == 'champion' else 'XGBoost'
    crit = r['criterion'].replace('cost_FN', 'FN:FP = ').replace('_FP1', ':1')
    if crit == 'macro_f1':
        crit = 'Macro-F1 optimum'
    cost = r['test_cost'] / bo['n_test'] * 1000 if pd.notna(r['test_cost']) else float('nan')
    cost_s = f"{cost:.2f}" if pd.notna(cost) else '--'
    lines.append(f"{label} & {crit} & {f3(r['threshold'])} & {f4(r['test_precision'])} & "
                 f"{f4(r['test_recall'])} & {cost_s} \\\\")
write('tab_bo_threshold.tex', r"""\begin{table}[H]
\centering
\caption{Thresholds selected on out-of-fold predictions and evaluated on the official
test split. Cost is reported per 1{,}000 records for a false-negative:false-positive
cost ratio of the indicated value. The macro-F1 row selects the threshold that maximises
out-of-fold macro F1.}
\label{tab:bo-threshold}
\small
\begin{tabular}{llcccc}
\toprule
Model & Criterion & Threshold & Test precision & Test recall & Cost / 1k \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# Base-rate table
b = bo['base_rate']
lines = []
for label, m in [('Unadjusted', b['raw']), ('Prior-corrected', b['prior_corrected'])]:
    lines.append(f"{label} & {f4(m['macro_f1'])} & {f4(m['ap'])} & {f4(m['roc_auc'])} & "
                 f"{f4(m['precision'])} & {f4(m['recall'])} & {f4(m['brier'])} \\\\")
write('tab_bo_base.tex', r"""\begin{table}[H]
\centering
\caption{Base-rate shift for the validation-selected champion. The training prevalence
is 0.669\% and the official-test prevalence is 1.110\%. Prior correction rescales the
predicted probabilities to the test prevalence; it is reported as a sensitivity analysis,
not as a tuned result.}
\label{tab:bo-base}
\small
\begin{tabular}{lcccccc}
\toprule
Probabilities & Macro F1 & AP & ROC-AUC & Prec. & Rec. & Brier \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
