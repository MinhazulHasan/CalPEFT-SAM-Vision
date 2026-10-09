import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from calpeft.conformal import random_splits, resplit
from calpeft.models import MODELS

MODEL_ORDER = ["unet", "unet-r34", "dinov2s", "sam1", "medsam", "sam2", "medsam2"]
SEEDS = (42, 43, 44)
REF = "sam2"
INK, INK2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
COLORS = {"kvasir": "#2a78d6", "busi": "#eb6834"}


def load_runs(out_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "runs", "*", "*", "metrics.json"))):
        with open(path) as f:
            m = json.load(f)
        m["run_dir"] = os.path.dirname(path)
        rows.append(m)
    return pd.DataFrame(rows)


def per_image_dice(run_dir):
    return pd.read_csv(os.path.join(run_dir, "per_image_dice.csv"))["dice"].to_numpy()


def holm(p):
    p = np.asarray(p, float)
    adjusted, running = np.empty_like(p), 0.0
    for k, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - k) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def crc_validity(runs, alpha, n_splits):
    rows, dists, perms = [], {}, {}
    for _, r in runs.iterrows():
        path = os.path.join(r.run_dir, "crc_curves.npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        key = (r.dataset, len(z["fnr"]))
        if key not in perms:
            perms[key] = random_splits(len(z["fnr"]), n_splits, seed=42)
        res = resplit(z["fnr"], z["reg"], int(z["n_cal"]), alpha, perms[key])
        dists[(r.dataset, r.run)] = res
        rows.append({"dataset": r.dataset, "run": r.run, "fnr_mean": round(res[:, 0].mean(), 4),
                     "fnr_sd": round(res[:, 0].std(), 4), "fnr_p95": round(np.percentile(res[:, 0], 95), 4),
                     "violation_rate": round((res[:, 0] > alpha).mean(), 3), "region_mean": round(res[:, 1].mean(), 4),
                     "region_sd": round(res[:, 1].std(), 4), "fnr_single_split": r.test_fnr})
    columns = ["dataset", "run", "fnr_mean", "fnr_sd", "fnr_p95", "violation_rate", "region_mean", "region_sd",
               "fnr_single_split"]
    return pd.DataFrame(rows, columns=columns), dists


def model_comparison(runs, crc):
    rows = []
    for ds in runs.dataset.unique():
        base = runs[(runs.dataset == ds) & (runs.run == runs.model) & runs.model.isin(MODEL_ORDER)]
        if REF not in set(base.model):
            continue
        ref = per_image_dice(base[base.model == REF].run_dir.item())
        part = []
        for m in [m for m in MODEL_ORDER if m in set(base.model)]:
            r = base[base.model == m].iloc[0]
            d = per_image_dice(r.run_dir)
            part.append({"dataset": ds, "model": m, "name": MODELS[m]["name"], "params_M": r.params_M,
                         "trainable_M": r.trainable_M, "dice": r.dice, "iou": r.iou, "ece": r.ece, "ace": r.ace,
                         "roi_ece": r.get("roi_ece", np.nan), "delta_dice_vs_sam2": round((d - ref).mean(), 4),
                         "p": 1.0 if m == REF or not np.any(d != ref) else wilcoxon(d, ref).pvalue, "p_holm": np.nan})
        others = [x for x in part if x["model"] != REF]
        for x, a in zip(others, holm([x["p"] for x in others])):
            x["p_holm"] = a
        rows += part
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).merge(crc.rename(columns={"run": "model"}), on=["dataset", "model"], how="left")


def main_comparison(runs, crc):
    summary, stats = [], []
    for ds in runs.dataset.unique():
        seeds = {}
        for m in ("sam2", "unet-r34"):
            names = [m if s == 42 else f"{m}-s{s}" for s in SEEDS]
            g = runs[(runs.dataset == ds) & runs.run.isin(names)].merge(
                crc[crc.dataset == ds][["run", "fnr_mean", "region_mean"]], on="run", how="left")
            if g.empty:
                continue
            seeds[m] = g.set_index("seed")
            row = {"dataset": ds, "model": m, "name": MODELS[m]["name"], "n_seeds": len(g)}
            for col, name in [("dice", "dice"), ("iou", "iou"), ("ece", "ece"), ("ace", "ace"), ("roi_ece", "roi_ece"),
                              ("fnr_mean", "crc_fnr"), ("region_mean", "crc_region")]:
                if col in g.columns and g[col].notna().all():
                    row[f"{name}_mean"], row[f"{name}_std"] = g[col].mean(), g[col].std(ddof=1)
            summary.append(row)
        if len(seeds) < 2:
            continue
        common = sorted(set(seeds["sam2"].index) & set(seeds["unet-r34"].index))
        diffs = [seeds["sam2"].loc[s, "dice"] - seeds["unet-r34"].loc[s, "dice"] for s in common]
        a = np.mean([per_image_dice(seeds["sam2"].loc[s, "run_dir"]) for s in common], axis=0)
        b = np.mean([per_image_dice(seeds["unet-r34"].loc[s, "run_dir"]) for s in common], axis=0)
        d = a - b
        rng = np.random.default_rng(0)
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
        ci = np.percentile(boot, [2.5, 97.5])
        stats.append({"dataset": ds, "seeds": len(common), "dice_diff_per_seed": " ".join(f"{x:+.4f}" for x in diffs),
                      "seeds_sam2_better": sum(x > 0 for x in diffs), "test_images": len(d),
                      "images_sam2_better": int(np.sum(d > 1e-9)), "images_sam2_worse": int(np.sum(d < -1e-9)),
                      "images_equal": int(np.sum(np.abs(d) <= 1e-9)), "mean_diff": d.mean(), "median_diff": np.median(d),
                      "wilcoxon_p": wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0, "paired_t_p": ttest_rel(a, b).pvalue,
                      "boot_ci_low": ci[0], "boot_ci_high": ci[1]})
    return pd.DataFrame(summary), pd.DataFrame(stats)


def ablation(runs, crc):
    names = {"sam2-nocal": "lambda_cal 0.0, rank 8", "sam2": "lambda_cal 0.1, rank 8", "sam2-cal05": "lambda_cal 0.5, rank 8",
             "sam2-cal10": "lambda_cal 1.0, rank 8", "sam2-r2": "lambda_cal 0.1, rank 2", "sam2-r4": "lambda_cal 0.1, rank 4",
             "sam2-r16": "lambda_cal 0.1, rank 16"}
    g = runs[(runs.dataset == "kvasir") & runs.run.isin(list(names))].merge(
        crc[crc.dataset == "kvasir"][["run", "fnr_mean", "region_mean"]], on="run", how="left")
    g = g.sort_values("run", key=lambda col: col.map(list(names).index))
    g.insert(1, "setting", g.run.map(names))
    cols = ["run", "setting", "trainable_M", "dice", "iou", "ece", "ace", "roi_ece", "fnr_mean", "region_mean"]
    return g[[c for c in cols if c in g.columns]]


def risk_efficiency(runs, n_splits, alphas=(0.05, 0.10, 0.15, 0.20)):
    rows, perms = [], {}
    for _, r in runs[runs.seed == 42].iterrows():
        path = os.path.join(r.run_dir, "crc_curves.npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        key = (r.dataset, len(z["fnr"]))
        if key not in perms:
            perms[key] = random_splits(len(z["fnr"]), n_splits, seed=42)
        for a in alphas:
            res = resplit(z["fnr"], z["reg"], int(z["n_cal"]), a, perms[key])
            rows.append({"dataset": r.dataset, "run": r.run, "alpha": a, "fnr_mean": round(res[:, 0].mean(), 4),
                         "region_mean": round(res[:, 1].mean(), 4)})
    return pd.DataFrame(rows)


def label(model):
    return MODELS[model]["name"] + (" (ours)" if model == REF else "")


def _style(ax, grid_axis="both"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)


def _model_ticks(ax, models, positions):
    ax.set_yticks(positions, [label(m) for m in models])
    for text, m in zip(ax.get_yticklabels(), models):
        text.set_color(INK)
        text.set_fontweight("bold" if m == REF else "normal")
    ax.tick_params(axis="y", length=0)


def crc_figure(dists, alpha, path):
    datasets = [ds for ds in COLORS if any(k[0] == ds for k in dists)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.6 * len(datasets), 4.8), facecolor=SURFACE, squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        models = [m for m in MODEL_ORDER if (ds, m) in dists]
        data = [dists[(ds, m)][:, 0] for m in models]
        style = dict(whis=(5, 95), showfliers=False, widths=0.55, patch_artist=True,
                     boxprops=dict(facecolor=COLORS[ds], edgecolor=SURFACE, linewidth=1.5),
                     medianprops=dict(color=SURFACE, linewidth=1.5), whiskerprops=dict(color=AXIS), capprops=dict(color=AXIS))
        try:
            ax.boxplot(data, orientation="horizontal", **style)
        except TypeError:                                   # matplotlib < 3.10
            ax.boxplot(data, vert=False, **style)
        _style(ax, "x")
        ax.scatter([d.mean() for d in data], np.arange(1, len(data) + 1), marker="D", s=28, color=INK, zorder=3)
        ax.axvline(alpha, color=INK2, linestyle="--", linewidth=1)
        ax.text(alpha, 0.5, f" target {alpha:.0%}", color=INK2, fontsize=8.5, va="center")
        _model_ticks(ax, models, np.arange(1, len(data) + 1))
        ax.set_ylim(0.3, len(data) + 0.5)
        ax.invert_yaxis()
        ax.set_title(f"{ds}: test FNR over {len(data[0])} random re-splits", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("test FNR   (box 25-75%, whiskers 5-95%, diamond = mean)", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def comparison_figure(table, alpha, path):
    panels = [("dice", "Dice (higher is better)"), ("ece", "ECE (lower is better)"),
              ("region_mean", f"CRC region size at FNR ≤ {alpha:.0%} (lower is tighter)")]
    models = [m for m in MODEL_ORDER if m in set(table.model)]
    datasets = [ds for ds in COLORS if ds in set(table.dataset)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.5 * len(models) + 1.6), sharey=True, facecolor=SURFACE)
    height = 0.8 / len(datasets)
    for ax, (col, title) in zip(axes, panels):
        _style(ax, "x")
        for j, ds in enumerate(datasets):
            g = table[table.dataset == ds].set_index("model")
            ys = np.arange(len(models)) + (j - (len(datasets) - 1) / 2) * height
            ax.barh(ys, [g[col].get(m, np.nan) for m in models], height=height, color=COLORS[ds],
                    edgecolor=SURFACE, linewidth=1.5, label=ds)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.tick_params(axis="y", length=0)
    _model_ticks(axes[0], models, np.arange(len(models)))
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=len(datasets), frameon=False,
               fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Build every table and figure of the paper from outputs/runs.")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--splits", type=int, default=1000)
    args = parser.parse_args()

    runs = load_runs(args.out_dir)
    if runs.empty:
        raise SystemExit(f"no finished runs in {args.out_dir}/runs - train something first")
    tables = os.path.join(args.out_dir, "tables")
    figures = os.path.join(args.out_dir, "figures")
    os.makedirs(tables, exist_ok=True)
    os.makedirs(figures, exist_ok=True)

    crc, dists = crc_validity(runs, args.alpha, args.splits)
    results = {"all_runs": runs.drop(columns="run_dir"), "crc_validity": crc,
               "model_comparison": model_comparison(runs, crc), "ablation": ablation(runs, crc),
               "risk_efficiency": risk_efficiency(runs, args.splits)}
    results["main_comparison"], results["main_statistics"] = main_comparison(runs, crc)
    for name, table in results.items():
        table.to_csv(os.path.join(tables, f"{name}.csv"), index=False)
        if not table.empty and name not in ("all_runs", "risk_efficiency"):
            print(f"\n=== {name} ===\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    seed42 = {k: v for k, v in dists.items() if k[1] in MODEL_ORDER}
    if seed42:
        crc_figure(seed42, args.alpha, os.path.join(figures, "crc_resplit_fnr.png"))
    if not results["model_comparison"].empty:
        comparison_figure(results["model_comparison"], args.alpha, os.path.join(figures, "model_comparison.png"))
    print(f"\ntables -> {tables}\nfigures -> {figures}")


if __name__ == "__main__":
    main()
