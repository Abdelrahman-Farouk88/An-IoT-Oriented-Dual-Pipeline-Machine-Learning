"""Pipeline B-informed inventory simulation.

Pipeline B's champion backorder model is used as a closed-loop controller. At each
cycle, each SKU's inventory state is mapped to the 21-feature backorder schema; the
predicted probability adjusts the reorder point by a discrete safety buffer. The
controller is compared with a static A-only policy, a distribution-matched shuffled
controller, and the original A-only risk policy.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from integrated_simulation import (TEST_SEEDS, TRAIN_SEEDS, mean_metrics,
                                   simulate)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def run(cfg, seeds):
    rows = []
    for seed in seeds:
        rows.append(simulate(
            seed, cfg["policy"], cfg["s"], cfg["q"], cfg["tau"],
            cfg.get("calendar_age", 150),
            b_controller=cfg.get("b_controller"),
            b_buffer=cfg.get("b_buffer", 0.0),
            b_threshold=cfg.get("b_threshold"),
            b_every=cfg.get("b_every", 1),
            b_seed=cfg.get("b_seed", 0),
        ))
    return pd.DataFrame(rows)


def summarize(df, cfg):
    out = mean_metrics(df.to_dict("records"))
    out["config"] = cfg
    return out


def main():
    static_cfgs = [
        {"label": f"static_s{s}", "policy": "risk", "tau": 0.95,
         "calendar_age": 150, "s": s, "q": 4}
        for s in (0, 2, 4, 6, 8)
    ]
    b_cfgs = []
    for s in (0, 2):
        for threshold in (0.02, 0.05, 0.10):
            for buffer in (2, 4, 8):
                b_cfgs.append({
                    "label": f"b_s{s}_t{threshold}_b{buffer}",
                    "policy": "risk", "tau": 0.95, "calendar_age": 150,
                    "s": s, "q": 4, "b_controller": "b_adjust",
                    "b_buffer": buffer, "b_threshold": threshold, "b_every": 10,
                })

    rows = []
    for cfg in static_cfgs:
        df = run(cfg, TRAIN_SEEDS)
        rows.append(summarize(df, cfg))
        print(f"{cfg['label']}: {rows[-1]['total_cost']:.1f}", flush=True)
    for cfg in b_cfgs:
        df = run(cfg, TRAIN_SEEDS)
        rows.append(summarize(df, cfg))
        print(f"{cfg['label']}: {rows[-1]['total_cost']:.1f} "
              f"p={rows[-1]['b_mean']:.4f} adj={rows[-1]['b_adjust_rate']:.3f}",
              flush=True)

    tune = pd.DataFrame(rows).sort_values("total_cost").reset_index(drop=True)
    tune.to_csv(OUT / "b_inventory_tuning.csv", index=False)

    static_best = tune[tune["config"].apply(lambda c: c["label"].startswith("static"))].iloc[0]
    b_best = tune[tune["config"].apply(lambda c: c["label"].startswith("b_"))].iloc[0]
    static_cfg = static_best["config"]
    b_cfg = b_best["config"]
    shuffle_cfg = dict(b_cfg, label="b_shuffled", b_controller="b_shuffle", b_seed=999)

    test_cfgs = [
        {"label": "a_only_risk_s0", "policy": "risk", "tau": 0.95,
         "calendar_age": 150, "s": 0, "q": 4},
        dict(static_cfg, label="static_best"),
        dict(b_cfg, label="b_adjusted"),
        shuffle_cfg,
        {"label": "oracle", "policy": "oracle", "tau": 0.9,
         "calendar_age": 150, "s": 0, "q": 4},
    ]
    test_rows = []
    summaries = {}
    for cfg in test_cfgs:
        df = run(cfg, TEST_SEEDS)
        test_rows.append(df)
        summaries[cfg["label"]] = summarize(df, cfg)
        print(f"{cfg['label']}: {summaries[cfg['label']]['total_cost']:.1f}", flush=True)
    pd.concat(test_rows).to_csv(OUT / "b_inventory_test_runs.csv", index=False)

    a_cost = summaries["a_only_risk_s0"]["total_cost"]
    s_cost = summaries["static_best"]["total_cost"]
    b_cost = summaries["b_adjusted"]["total_cost"]
    sh_cost = summaries["b_shuffled"]["total_cost"]
    comparisons = {
        "b_vs_static_pct": float(100 * (b_cost - s_cost) / s_cost),
        "b_vs_shuffled_pct": float(100 * (b_cost - sh_cost) / sh_cost),
        "b_vs_a_only_pct": float(100 * (b_cost - a_cost) / a_cost),
    }
    result = {
        "protocol": "B probabilities are computed from a state-to-feature mapping; "
                    "controller and buffers are tuned on training seeds only",
        "mapping": {
            "national_inv": "on_hand + on_order - backorders",
            "lead_time": "SKU mean lead time",
            "in_transit_qty": "units on order",
            "forecast_3/6/9_month": "demand in previous 3/6/9 cycles",
            "sales_1/3/6/9_month": "demand in previous 1/3/6/9 cycles",
            "min_bank": "nominal reorder point s",
            "potential_issue": "any backorder in queue",
            "pieces_past_due": "number of backordered assets",
            "perf_6/12_month_avg": "fill rate over previous 6/12 cycles",
            "local_bo_qty": "number of backordered assets",
            "deck_risk": "inventory position below zero",
            "oe_constraint": "0",
            "ppap_risk": "0",
            "stop_auto_buy": "0",
            "rev_stop": "0",
        },
        "static_best": static_cfg,
        "b_best": b_cfg,
        "shuffle_config": shuffle_cfg,
        "comparisons": comparisons,
        "summary": summaries,
    }
    with open(OUT / "b_inventory_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
