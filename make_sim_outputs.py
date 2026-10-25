import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
TAB = ROOT / "tables"
FIG = ROOT / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})
C = {"reactive": "#D55E00", "calendar": "#E69F00",
     "risk": "#0072B2", "oracle": "#009E73"}
LABEL = {"reactive": "Reactive", "calendar": "Calendar",
         "risk": "Risk-triggered", "oracle": "Oracle"}
ORDER = ["reactive", "calendar", "risk", "oracle"]


def fmt(x):
    return f"{x:,.0f}"


def ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    return float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


def load():
    with open(RES / "sim_test_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    runs = pd.read_csv(RES / "sim_test_runs.csv")
    sens = pd.read_csv(RES / "sim_sensitivity_runs.csv")
    tune = pd.read_csv(RES / "sim_tune_risk.csv")
    return summary, runs, sens, tune


def table_policies(summary, runs):
    rows = []
    for pol in ORDER:
        s = summary[pol]
        c = s["config"]
        config = f"$\\tau={c['tau']:.2f}$, $s={c['s']}$, $Q={c['q']}$" if pol in ("risk",) else \
                 f"age={c['calendar_age']}, $s={c['s']}$, $Q={c['q']}$" if pol == "calendar" else \
                 f"$s={c['s']}$, $Q={c['q']}$"
        sub = runs[runs.policy == pol]
        half = ci95(sub.total_cost.values)
        inv = s["holding_cost"] + s["backorder_cost"] + s["purchasing_cost"]
        rows.append((LABEL[pol], config, f"{fmt(s['total_cost'])} ({fmt(half)})",
                     fmt(inv), f"{s['failures']:.2f}", f"{s['planned']:.1f}",
                     f"{100*s['immediate_fill_rate']:.1f}", f"{100*s['uptime']:.2f}"))
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Simulation-based policy comparison on 20 held-out seeds (100 assets, 600 cycles, five SKUs). Costs are mean values; the total-cost column reports the 95\% confidence-interval half-width in parentheses. Inventory/order combines holding, backorder, and purchasing costs. The cost decomposition is shown in Fig.~\ref{fig:sim-policies}.}",
        r"\label{tab:sim-policies}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Policy & Configuration & Total cost & Inventory/order & Failures & Planned & Fill (\%) & Uptime (\%) \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TAB / "tab_sim_policies.tex").write_text("\n".join(lines), encoding="utf-8")


def table_sensitivity(sens):
    s = sens.groupby(["scenario", "policy"]).total_cost.mean().unstack()
    s = s[ORDER]
    rows = []
    names = {"base": "Base", "low_failure": "Failure cost $750$",
             "high_failure": "Failure cost $3{,}000$",
             "short_lead": "Lead time $\\times 0.75$",
             "long_lead": "Lead time $\\times 1.5$"}
    for sc in ["base", "low_failure", "high_failure", "short_lead", "long_lead"]:
        v = s.loc[sc]
        red = 100 * (1 - v["risk"] / v["reactive"])
        rows.append((names[sc], fmt(v["reactive"]), fmt(v["calendar"]),
                     fmt(v["risk"]), fmt(v["oracle"]), f"{red:.1f}"))
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Sensitivity of the tuned policies to the failure-cost and lead-time assumptions. Values are mean total simulation cost over ten held-out seeds. The policies are not retuned within each scenario.}",
        r"\label{tab:sim-sensitivity}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Scenario & Reactive & Calendar & Risk-triggered & Oracle & Risk vs.\ reactive (\%) \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TAB / "tab_sim_sensitivity.tex").write_text("\n".join(lines), encoding="utf-8")


def table_b_controller():
    with open(RES / "b_inventory_summary.json", encoding="utf-8") as f:
        d = json.load(f)
    order = ["a_only_risk_s0", "static_best", "b_adjusted", "b_shuffled", "oracle"]
    labels = {"a_only_risk_s0": "A-only risk ($s=0$)",
              "static_best": "Static A-only (best)",
              "b_adjusted": "B-adjusted",
              "b_shuffled": "B-shuffled control",
              "oracle": "Oracle"}
    rows = []
    for key in order:
        s = d["summary"][key]
        c = s["config"]
        inv = s["holding_cost"] + s["backorder_cost"] + s["purchasing_cost"]
        rows.append((labels[key], f"{fmt(s['total_cost'])}", f"{s['failures']:.2f}",
                     fmt(inv), f"{100*s['immediate_fill_rate']:.1f}",
                     f"{100*s['uptime']:.2f}", f"{s['b_mean']:.4f}",
                     f"{100*s['b_adjust_rate']:.1f}"))
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Pipeline B-informed inventory controller on 20 held-out seeds. B probabilities are queried every ten cycles from a state-to-feature mapping and add a discrete safety buffer when the predicted backorder risk exceeds the tuned threshold. The shuffled control uses the same buffer and adjustment frequency but randomly permutes B probabilities across SKUs. Values are means; the B-adjusted configuration was selected on ten training seeds.}",
        r"\label{tab:sim-b}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Policy & Total cost & Failures & Inventory/order & Fill (\%) & Uptime (\%) & Mean $p_B$ & Adjusted (\%) \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TAB / "tab_sim_b.tex").write_text("\n".join(lines), encoding="utf-8")


def fig_policies(summary, runs):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    ax = axes[0]
    x = np.arange(len(ORDER))
    bottoms = np.zeros(len(ORDER))
    comps = [("failure_cost", "Failure", "#D55E00"),
             ("planned_cost", "Planned", "#0072B2"),
             ("downtime_cost", "Downtime", "#E69F00"),
             ("holding_cost", "Holding", "#009E73"),
             ("backorder_cost", "Backorder", "#CC79A7"),
             ("purchasing_cost", "Ordering", "#999999")]
    for key, lab, col in comps:
        vals = np.array([summary[p][key] for p in ORDER]) / 1000.0
        ax.bar(x, vals, bottom=bottoms, color=col, label=lab, width=0.65)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[p] for p in ORDER], rotation=15, ha="right")
    ax.set_ylabel(r"Cost ($\times 10^3$)")
    ax.set_title("Cost decomposition")
    ax.legend(ncol=2, fontsize=7, frameon=False)
    ax = axes[1]
    fails = [summary[p]["failures"] for p in ORDER]
    planned = [summary[p]["planned"] for p in ORDER]
    ax.bar(x - 0.18, fails, 0.36, color="#D55E00", label="Failures")
    ax.bar(x + 0.18, planned, 0.36, color="#0072B2", label="Planned")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[p] for p in ORDER], rotation=15, ha="right")
    ax.set_ylabel("Mean count per run")
    ax.set_title("Maintenance events")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_sim_policies.pdf", dpi=300)
    plt.close(fig)


def fig_sensitivity(sens):
    s = sens.groupby(["scenario", "policy"]).total_cost.mean().unstack()[ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=False)
    ax = axes[0]
    x = np.arange(3)
    labels = ["$0.75\\times$", "Base", "$1.5\\times$"]
    for pol in ORDER:
        vals = [s.loc["short_lead", pol], s.loc["base", pol], s.loc["long_lead", pol]]
        ax.plot(x, np.array(vals) / 1000, marker="o", color=C[pol], label=LABEL[pol])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Lead-time multiplier")
    ax.set_ylabel(r"Cost ($\times 10^3$)")
    ax.set_title("Lead-time sensitivity")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    x = np.arange(3)
    labels = ["$750$", "Base", "$3{,}000$"]
    for pol in ORDER:
        vals = [s.loc["low_failure", pol], s.loc["base", pol], s.loc["high_failure", pol]]
        ax.plot(x, np.array(vals) / 1000, marker="o", color=C[pol], label=LABEL[pol])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Failure cost")
    ax.set_ylabel(r"Cost ($\times 10^3$)")
    ax.set_title("Failure-cost sensitivity")
    fig.tight_layout()
    fig.savefig(FIG / "fig_sim_sensitivity.pdf", dpi=300)
    plt.close(fig)


def fig_threshold(tune):
    best = tune.groupby("tau").mean_cost.min().sort_index()
    fig, ax = plt.subplots(figsize=(4.1, 2.9))
    ax.plot(best.index, best.values / 1000, marker="o", color=C["risk"])
    ax.set_xlabel(r"Risk threshold $\tau$")
    ax.set_ylabel(r"Best mean cost ($\times 10^3$)")
    ax.set_title("Risk-threshold tuning")
    fig.tight_layout()
    fig.savefig(FIG / "fig_sim_threshold.pdf", dpi=300)
    plt.close(fig)


def main():
    summary, runs, sens, tune = load()
    table_policies(summary, runs)
    table_sensitivity(sens)
    table_b_controller()
    fig_policies(summary, runs)
    fig_sensitivity(sens)
    fig_threshold(tune)
    print("simulation outputs written")


if __name__ == "__main__":
    main()
