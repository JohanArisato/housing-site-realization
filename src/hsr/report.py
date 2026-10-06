"""Figures and planner-facing tables."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import warnings

warnings.filterwarnings("ignore", message=".*TreeExplainer shap values output has changed.*")
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

# Validated categorical palette (fixed order) + text tokens
C = {"capacity_baseline": "#8a8984", "logistic": "#eb6834", "gbm": "#2a78d6"}
LABEL = {"capacity_baseline": "Capacity baseline (city assumption)",
         "logistic": "Logistic regression", "gbm": "Gradient boosting (calibrated)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_roc_pr(y, oof, path, subtitle=""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for name in ["capacity_baseline", "logistic", "gbm"]:
        s = oof[name].astype(float)
        fpr, tpr, _ = roc_curve(y, s)
        axes[0].plot(fpr, tpr, color=C[name], lw=2,
                     label=f"{LABEL[name]} (AUC {roc_auc_score(y, s):.2f})")
        p, r, _ = precision_recall_curve(y, s)
        axes[1].plot(r, p, color=C[name], lw=2,
                     label=f"{LABEL[name]} (AP {average_precision_score(y, s):.2f})")
    axes[0].plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    axes[1].axhline(np.mean(y), color=GRID, lw=1, ls="--")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision–recall")
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "upper right", fontsize=8)
    fig.suptitle(f"Held-out community plan areas{subtitle}", color=INK2, fontsize=9, y=1.0)
    _save(fig, path)


def plot_calibration(y, oof, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--", label="Perfect calibration")
    for name in ["logistic", "gbm"]:
        pt, pp = calibration_curve(y, oof[name].astype(float), n_bins=10, strategy="quantile")
        ax.plot(pp, pt, marker="o", ms=5, lw=2, color=C[name], label=LABEL[name])
    lim = max(0.05, float(np.nanmax(oof[["logistic", "gbm"]].astype(float).quantile(0.99))) * 1.1)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Predicted probability",
           ylabel="Observed share realized", title="Calibration")
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, path)


def plot_spatial_vs_random(scores_spatial, scores_random, path):
    names = ["capacity_baseline", "logistic", "gbm"]
    sp = scores_spatial.set_index("model").loc[names, "roc_auc"]
    rd = scores_random.set_index("model").loc[names, "roc_auc"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    yy = np.arange(len(names))
    for i, n in enumerate(names):
        ax.plot([sp[n], rd[n]], [i, i], color=GRID, lw=3, zorder=1)
    ax.scatter(sp, yy, s=60, color="#2a78d6", zorder=2, label="Spatial CV (unseen communities)")
    ax.scatter(rd, yy, s=60, color="#eb6834", zorder=2, label="Random CV")
    ax.set_yticks(yy, [LABEL[n] for n in names])
    ax.set(xlabel="ROC-AUC", title="Spatial vs. random cross-validation")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, path)


def plot_shap(model, X, path, max_display=14):
    sample = X.sample(min(len(X), 2000), random_state=0)
    sv = shap.TreeExplainer(model).shap_values(sample)
    if isinstance(sv, list):
        sv = sv[1]
    plt.figure()
    shap.summary_plot(sv, sample, max_display=max_display, show=False, plot_size=(7.5, 5.2))
    plt.title("What drives predicted realization (SHAP)", loc="left")
    _save(plt.gcf(), path)
    imp = pd.Series(np.abs(sv).mean(0), index=sample.columns).sort_values(ascending=False)
    return imp


def realistic_capacity(sites, y, p, groups) -> pd.DataFrame:
    """Claimed capacity vs model-expected units, by community plan area."""
    df = pd.DataFrame({"cpa": groups.values, "net_units": sites["NetUnits"].clip(lower=0).values,
                       "p": p.values, "realized": y.values})
    df["expected_units"] = df.p * df.net_units
    t = df.groupby("cpa").agg(sites=("p", "size"), claimed_units=("net_units", "sum"),
                              expected_units=("expected_units", "sum"),
                              mean_prob=("p", "mean"), observed_rate=("realized", "mean"))
    t["realization_ratio"] = t.expected_units / t.claimed_units
    return t.sort_values("claimed_units", ascending=False)


def plot_realistic_capacity(table, path, top=15):
    t = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(t) + 1.2))
    yy = np.arange(len(t))
    ax.barh(yy, t.claimed_units, height=0.62, color="#cde2fb", label="Claimed capacity (inventory)")
    ax.barh(yy, t.expected_units, height=0.62, color="#2a78d6",
            label="Model-expected units (Σ p × capacity)")
    for i, (c, e) in enumerate(zip(t.claimed_units, t.expected_units)):
        ax.text(c, i, f"  {e / c:.0%}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(yy, t.index)
    ax.set(xlabel="Net new housing units", title="Claimed vs. realistic capacity by community")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, path)


def equity_table(y, p, X) -> pd.DataFrame:
    names = {4: "Highest", 3: "High", 2: "Moderate", 1: "Low", 0: "High seg. & poverty"}
    df = pd.DataFrame({"opp": X["opportunity"].map(names).fillna("Unknown"),
                       "y": y.values, "p": p.values})
    rows = []
    for g, d in df.groupby("opp"):
        auc = roc_auc_score(d.y, d.p) if d.y.nunique() == 2 and len(d) > 30 else np.nan
        rows.append({"opportunity_area": g, "sites": len(d), "observed_rate": d.y.mean(),
                     "mean_predicted": d.p.mean(), "roc_auc": auc})
    order = ["Highest", "High", "Moderate", "Low", "High seg. & poverty", "Unknown"]
    return pd.DataFrame(rows).set_index("opportunity_area").reindex(
        [o for o in order if o in set(df.opp)])


def plot_equity(t, path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(t))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, t.observed_rate, w, color="#eb6834", label="Observed realization rate")
    ax.bar(x + w / 2 + 0.01, t.mean_predicted, w, color="#2a78d6", label="Mean predicted probability")
    for i, a in enumerate(t.roc_auc):
        if not np.isnan(a):
            ax.text(i, max(t.observed_rate.iloc[i], t.mean_predicted.iloc[i]) * 1.04,
                    f"AUC {a:.2f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, t.index, fontsize=8)
    ax.set_ylim(0, float(t[["observed_rate", "mean_predicted"]].max().max()) * 1.25)
    ax.set(ylabel="Share of sites", title="Is the model equally right across opportunity areas?")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, loc="upper right")
    _save(fig, path)
