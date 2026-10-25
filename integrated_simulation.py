import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"
RESULTS.mkdir(exist_ok=True)
RISK_FILE = RESULTS / "sim_risk_curves.csv"
B_BUNDLE = None


@dataclass(frozen=True)
class SimParams:
    n_assets: int = 100
    n_skus: int = 5
    horizon: int = 600
    holding_cost: float = 1.0
    order_cost: float = 40.0
    planned_cost: float = 150.0
    failure_cost: float = 1500.0
    downtime_cost: float = 25.0
    backorder_cost: float = 15.0
    planned_repair: int = 1
    failure_repair: int = 3
    min_age: int = 10
    max_replacements: int = 120
    lead_scale: float = 1.0


DEFAULT = SimParams()
LEAD_MEANS = [2, 3, 4, 5, 6]
TRAIN_SEEDS = list(range(100, 110))
TEST_SEEDS = list(range(200, 220))


def load_risk():
    d = pd.read_csv(RISK_FILE)
    units = sorted(d["unit"].unique())
    max_life = int(d.groupby("unit")["cycle"].max().max())
    risk = np.zeros((len(units), max_life), dtype=float)
    lives = np.zeros(len(units), dtype=int)
    for j, u in enumerate(units):
        g = d.loc[d["unit"] == u].sort_values("cycle")
        life = int(g["cycle"].max())
        risk[j, :life] = g["p"].to_numpy(dtype=float)
        lives[j] = life
    return risk, lives, units


RISK, LIVES, UNITS = load_risk()


def load_b_bundle():
    global B_BUNDLE
    if B_BUNDLE is None:
        B_BUNDLE = joblib.load(MODELS / "bo_final.joblib")
        model = B_BUNDLE["models"][B_BUNDLE["champion"]]
        if hasattr(model, "set_params"):
            model.set_params(n_jobs=1)
    return B_BUNDLE


def b_probabilities(rows, shuffle_rng=None):
    bundle = load_b_bundle()
    cols = bundle["features"]
    df = pd.DataFrame(rows, columns=cols)
    for c in cols:
        df[c] = df[c].fillna(bundle["medians"][c])
    x = bundle["scaler"].transform(df[cols].values)
    model = bundle["models"][bundle["champion"]]
    p = model.predict_proba(x)[:, 1]
    if shuffle_rng is not None:
        p = shuffle_rng.permutation(p)
    return p


def simulate(seed, policy, s, q, tau=0.9, calendar_age=150, oracle_margin=10,
             params=DEFAULT, b_controller=None, b_buffer=0.0, b_threshold=None,
             b_every=1, b_seed=0):
    rng = np.random.default_rng(seed)
    n = params.n_assets
    n_skus = params.n_skus
    horizon = params.horizon

    sku = np.repeat(np.arange(n_skus), n // n_skus)
    rng.shuffle(sku)

    engine_draws = rng.integers(0, len(UNITS), size=(n, params.max_replacements + 2))
    lead_draws = {}
    lead_count = np.zeros(n_skus, dtype=int)
    for k, mu in enumerate(LEAD_MEANS[:n_skus]):
        lead_draws[k] = np.clip(np.rint(rng.normal(mu * params.lead_scale, 0.75, 5000)), 1, 15).astype(int)

    on_hand = np.full(n_skus, s + q, dtype=int)
    pipeline = [[] for _ in range(n_skus)]
    backorders = [[] for _ in range(n_skus)]
    demand_hist = np.zeros((n_skus, horizon + 2), dtype=int)
    fill_hist = np.zeros((n_skus, horizon + 2), dtype=int)
    b_rng = np.random.default_rng(seed + b_seed)
    b_sum = 0.0
    b_max = 0.0
    b_adjust_sum = 0.0
    b_adjust_count = 0
    b_steps = 0
    s_eff = np.full(n_skus, s, dtype=int)

    age = np.zeros(n, dtype=int)
    life = np.zeros(n, dtype=int)
    engine = np.zeros(n, dtype=int)
    state = np.zeros(n, dtype=int)
    pending_type = np.zeros(n, dtype=int)
    repair_end = np.full(n, -1, dtype=int)
    replace_count = np.zeros(n, dtype=int)

    engine[:] = engine_draws[:, 0]
    life[:] = LIVES[engine]

    total_demands = 0
    immediate_fills = 0
    planned_count = 0
    failure_count = 0
    converted_failures = 0
    order_count = 0
    backorder_events = 0

    failure_cost = 0.0
    planned_cost = 0.0
    downtime_cost = 0.0
    holding_cost = 0.0
    backorder_cost = 0.0
    purchasing_cost = 0.0

    def request(asset, kind):
        nonlocal total_demands, immediate_fills, planned_count, failure_count
        nonlocal backorder_events, failure_cost, planned_cost
        total_demands += 1
        pending_type[asset] = kind
        if kind == 0:
            planned_count += 1
            planned_cost += params.planned_cost
        else:
            failure_count += 1
            failure_cost += params.failure_cost
        sk = sku[asset]
        demand_hist[sk, t] += 1
        if on_hand[sk] > 0:
            on_hand[sk] -= 1
            immediate_fills += 1
            fill_hist[sk, t] += 1
            state[asset] = 3 if kind == 0 else 4
            repair_end[asset] = t + (params.planned_repair if kind == 0 else params.failure_repair)
        else:
            state[asset] = 1 if kind == 0 else 2
            backorders[sk].append(asset)
            backorder_events += 1

    for t in range(1, horizon + 1):
        for sk in range(n_skus):
            kept = []
            for arrival, qty in pipeline[sk]:
                if arrival <= t:
                    on_hand[sk] += qty
                else:
                    kept.append((arrival, qty))
            pipeline[sk] = kept
            while on_hand[sk] > 0 and backorders[sk]:
                asset = backorders[sk].pop(0)
                on_hand[sk] -= 1
                kind = pending_type[asset]
                state[asset] = 3 if kind == 0 else 4
                repair_end[asset] = t + (params.planned_repair if kind == 0 else params.failure_repair)

        if b_controller in ("b_adjust", "b_shuffle") and t % b_every == 0:
            rows = []
            end = t - 1
            for sk in range(n_skus):
                pos = int(on_hand[sk] + sum(qty for _, qty in pipeline[sk])
                          - len(backorders[sk]))
                in_transit = int(sum(qty for _, qty in pipeline[sk]))
                lo1, lo3 = max(1, end), max(1, end - 2)
                lo6, lo9 = max(1, end - 5), max(1, end - 8)
                d1 = int(demand_hist[sk, lo1:end + 1].sum()) if end >= 1 else 0
                d3 = int(demand_hist[sk, lo3:end + 1].sum()) if end >= 1 else 0
                d6 = int(demand_hist[sk, lo6:end + 1].sum()) if end >= 1 else 0
                d9 = int(demand_hist[sk, lo9:end + 1].sum()) if end >= 1 else 0
                f6 = int(fill_hist[sk, lo6:end + 1].sum()) if end >= 1 else 0
                f12 = int(fill_hist[sk, max(1, end - 11):end + 1].sum()) if end >= 1 else 0
                d12 = int(demand_hist[sk, max(1, end - 11):end + 1].sum()) if end >= 1 else 0
                rows.append({
                    "national_inv": pos,
                    "lead_time": float(LEAD_MEANS[sk] * params.lead_scale),
                    "in_transit_qty": in_transit,
                    "forecast_3_month": d3,
                    "forecast_6_month": d6,
                    "forecast_9_month": d9,
                    "sales_1_month": d1,
                    "sales_3_month": d3,
                    "sales_6_month": d6,
                    "sales_9_month": d9,
                    "min_bank": int(s),
                    "potential_issue": int(len(backorders[sk]) > 0),
                    "pieces_past_due": int(len(backorders[sk])),
                    "perf_6_month_avg": float(f6 / d6) if d6 else 1.0,
                    "perf_12_month_avg": float(f12 / d12) if d12 else 1.0,
                    "local_bo_qty": int(len(backorders[sk])),
                    "deck_risk": int(pos < 0),
                    "oe_constraint": 0,
                    "ppap_risk": 0,
                    "stop_auto_buy": 0,
                    "rev_stop": 0,
                })
            p_b = b_probabilities(rows, b_rng if b_controller == "b_shuffle" else None)
            if b_threshold is None:
                adjust = np.clip(np.rint(b_buffer * p_b).astype(int), 0, 50)
            else:
                adjust = np.where(p_b >= b_threshold, int(b_buffer), 0)
            s_eff = s + adjust
            b_sum += float(p_b.sum())
            b_max = max(b_max, float(p_b.max()))
            b_adjust_sum += float(adjust.sum())
            b_adjust_count += int((adjust > 0).sum())
            b_steps += n_skus

        for asset in range(n):
            if state[asset] != 0:
                continue
            age[asset] += 1
            if age[asset] >= life[asset]:
                request(asset, 1)
            elif policy == "calendar" and age[asset] >= calendar_age:
                request(asset, 0)
            elif policy == "risk" and age[asset] >= params.min_age:
                p = RISK[engine[asset], min(age[asset] - 1, life[asset] - 1)]
                if p >= tau:
                    request(asset, 0)
            elif policy == "oracle" and age[asset] >= life[asset] - oracle_margin:
                request(asset, 0)

        for asset in range(n):
            if state[asset] == 1:
                age[asset] += 1
                if age[asset] >= life[asset]:
                    converted_failures += 1
                    failure_count += 1
                    failure_cost += params.failure_cost
                    state[asset] = 2
                    pending_type[asset] = 1

        for asset in range(n):
            if state[asset] in (3, 4) and t >= repair_end[asset]:
                age[asset] = 0
                replace_count[asset] += 1
                slot = min(replace_count[asset], params.max_replacements)
                engine[asset] = engine_draws[asset, slot]
                life[asset] = LIVES[engine[asset]]
                state[asset] = 0
                repair_end[asset] = -1

        downtime_cost += params.downtime_cost * float(np.sum(state != 0))
        for sk in range(n_skus):
            holding_cost += params.holding_cost * on_hand[sk]
            backorder_cost += params.backorder_cost * len(backorders[sk])

        for sk in range(n_skus):
            position = on_hand[sk] + sum(qty for _, qty in pipeline[sk]) - len(backorders[sk])
            if position <= s_eff[sk]:
                lead = int(lead_draws[sk][lead_count[sk] % len(lead_draws[sk])])
                lead_count[sk] += 1
                pipeline[sk].append((t + lead, q))
                order_count += 1
                purchasing_cost += params.order_cost

    total_cost = (failure_cost + planned_cost + downtime_cost + holding_cost
                  + backorder_cost + purchasing_cost)
    replacements = int(np.sum(replace_count))
    return {
        "seed": int(seed),
        "policy": policy,
        "tau": float(tau),
        "calendar_age": int(calendar_age),
        "s": int(s),
        "q": int(q),
        "total_cost": float(total_cost),
        "failure_cost": float(failure_cost),
        "planned_cost": float(planned_cost),
        "downtime_cost": float(downtime_cost),
        "holding_cost": float(holding_cost),
        "backorder_cost": float(backorder_cost),
        "purchasing_cost": float(purchasing_cost),
        "failures": int(failure_count),
        "planned": int(planned_count),
        "converted_failures": int(converted_failures),
        "replacements": replacements,
        "orders": int(order_count),
        "demands": int(total_demands),
        "immediate_fill_rate": float(immediate_fills / total_demands) if total_demands else 1.0,
        "backorder_events": int(backorder_events),
        "uptime": float(1.0 - downtime_cost / (params.downtime_cost * n * horizon)),
        "b_controller": b_controller or "none",
        "b_buffer": float(b_buffer),
        "b_threshold": float(b_threshold) if b_threshold is not None else -1.0,
        "b_every": int(b_every),
        "b_mean": float(b_sum / b_steps) if b_steps else 0.0,
        "b_max": float(b_max),
        "b_adjust_mean": float(b_adjust_sum / b_steps) if b_steps else 0.0,
        "b_adjust_rate": float(b_adjust_count / b_steps) if b_steps else 0.0,
    }


def mean_metrics(rows):
    d = pd.DataFrame(rows)
    out = {"n": int(len(d))}
    for col in ["total_cost", "failure_cost", "planned_cost", "downtime_cost",
                "holding_cost", "backorder_cost", "purchasing_cost", "failures",
                "planned", "converted_failures", "replacements", "orders", "demands",
                "immediate_fill_rate", "backorder_events", "uptime", "b_mean",
                "b_max", "b_adjust_mean", "b_adjust_rate"]:
        out[col] = float(d[col].mean())
        out[col + "_sd"] = float(d[col].std(ddof=1))
    return out


def config_rows(policy):
    rows = []
    if policy in ("reactive", "oracle"):
        for s in (0, 4, 8, 12, 16):
            for q in (4, 8, 12, 16, 20):
                rows.append({"policy": policy, "tau": 0.9, "calendar_age": 150,
                             "s": s, "q": q})
    elif policy == "calendar":
        for cal in (80, 100, 120, 140, 160, 180, 200):
            for s in (0, 4, 8, 12, 16):
                for q in (4, 8, 12, 16, 20):
                    rows.append({"policy": policy, "tau": 0.9, "calendar_age": cal,
                                 "s": s, "q": q})
    else:
        for tau in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            for s in (0, 4, 8, 12, 16):
                for q in (4, 8, 12, 16, 20):
                    rows.append({"policy": policy, "tau": tau, "calendar_age": 150,
                                 "s": s, "q": q})
    return rows


def tune_one(cfg, seeds):
    vals = [simulate(seed, cfg["policy"], cfg["s"], cfg["q"], cfg["tau"],
                     cfg["calendar_age"])["total_cost"] for seed in seeds]
    return {**cfg, "mean_cost": float(np.mean(vals)), "sd_cost": float(np.std(vals, ddof=1))}


def tune(policy, seeds=TRAIN_SEEDS, n_jobs=4):
    rows = config_rows(policy)
    out = Parallel(n_jobs=n_jobs)(delayed(tune_one)(c, seeds) for c in rows)
    return pd.DataFrame(out).sort_values("mean_cost").reset_index(drop=True)


def evaluate_configs(configs, seeds):
    rows = []
    for cfg in configs:
        for seed in seeds:
            rows.append(simulate(seed, cfg["policy"], cfg["s"], cfg["q"], cfg["tau"],
                                 cfg["calendar_age"]))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quick", "tune", "final", "sensitivity"],
                    default="quick")
    ap.add_argument("--n_jobs", type=int, default=4)
    args = ap.parse_args()

    if args.mode == "quick":
        for pol in ("reactive", "calendar", "risk", "oracle"):
            r = simulate(200, pol, 8, 8, 0.9, 150)
            print(pol, {k: round(r[k], 3) if isinstance(r[k], float) else r[k]
                        for k in ("total_cost", "failures", "planned", "immediate_fill_rate",
                                  "uptime", "replacements")})
        return

    if args.mode == "tune":
        all_best = []
        for pol in ("reactive", "calendar", "risk", "oracle"):
            d = tune(pol, TRAIN_SEEDS, args.n_jobs)
            d.to_csv(RESULTS / f"sim_tune_{pol}.csv", index=False)
            best = d.iloc[0].to_dict()
            best["policy"] = pol
            all_best.append(best)
            print(pol, best, flush=True)
        with open(RESULTS / "sim_best_configs.json", "w") as f:
            json.dump(all_best, f, indent=2)
        return

    if args.mode == "final":
        with open(RESULTS / "sim_best_configs.json") as f:
            cfgs = json.load(f)
        rows = evaluate_configs(cfgs, TEST_SEEDS)
        rows.to_csv(RESULTS / "sim_test_runs.csv", index=False)
        summary = {}
        for pol, g in rows.groupby("policy"):
            summary[pol] = mean_metrics(g.to_dict("records"))
            summary[pol]["config"] = {k: cfgs[[c["policy"] for c in cfgs].index(pol)][k]
                                      for k in ("policy", "tau", "calendar_age", "s", "q")}
        with open(RESULTS / "sim_test_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        return

    if args.mode == "sensitivity":
        with open(RESULTS / "sim_best_configs.json") as f:
            cfgs = json.load(f)
        rows = []
        for label, params in {
            "base": DEFAULT,
            "high_failure": SimParams(failure_cost=3000.0),
            "low_failure": SimParams(failure_cost=750.0),
            "long_lead": SimParams(lead_scale=1.5),
            "short_lead": SimParams(lead_scale=0.75),
        }.items():
            for cfg in cfgs:
                for seed in TEST_SEEDS[:10]:
                    r = simulate(seed, cfg["policy"], cfg["s"], cfg["q"], cfg["tau"],
                                 cfg["calendar_age"], 10, params)
                    r["scenario"] = label
                    rows.append(r)
        d = pd.DataFrame(rows)
        d.to_csv(RESULTS / "sim_sensitivity_runs.csv", index=False)
        print(d.groupby(["scenario", "policy"])["total_cost"].mean().unstack().round(2).to_string())
        return


if __name__ == "__main__":
    main()
