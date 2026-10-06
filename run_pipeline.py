"""Run the full study end to end.

    python run_pipeline.py --data sample   # offline, synthetic data (default)
    python run_pipeline.py --data real     # after `python -m hsr.download`
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd  # noqa: E402

from hsr import config, models, report, webmap  # noqa: E402
from hsr.build_dataset import build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["sample", "real"], default="sample")
    args = ap.parse_args()
    data_dir = config.DATA_SAMPLE if args.data == "sample" else config.DATA_RAW
    if not (data_dir / "sites.geojson").exists():
        hint = "python -m hsr.make_sample_data" if args.data == "sample" else "python -m hsr.download"
        sys.exit(f"No data in {data_dir}. Run `{hint}` first (with PYTHONPATH=src).")
    tag = " — SYNTHETIC SAMPLE" if args.data == "sample" else ""
    out, fig = config.OUTPUTS / args.data, config.FIGURES / args.data
    out.mkdir(parents=True, exist_ok=True)

    print("1/6 Building dataset ...")
    sites, X, y, groups = build(data_dir)
    print(f"    {len(X)} sites, {X.shape[1]} features, {y.mean():.1%} realized, "
          f"{groups.nunique()} community plan areas")

    print("2/6 Spatial cross-validation (held-out community plan areas) ...")
    oof = models.cross_validate(X, y, groups, scheme="spatial")
    scores = models.score_table(y, oof)
    print("3/6 Random cross-validation (for comparison) ...")
    oof_rand = models.cross_validate(X, y, groups, scheme="random")
    scores_rand = models.score_table(y, oof_rand, ci=False)

    print("4/6 Explaining the model (SHAP) ...")
    final = models.fit_final_gbm(X, y)
    importance = report.plot_shap(final, X, fig / "shap_summary.png")

    print("5/6 Planner outputs ...")
    p = oof["gbm"].astype(float)
    cap = report.realistic_capacity(sites, y, p, groups)
    eq = report.equity_table(y, p, X)
    report.plot_roc_pr(y, oof, fig / "roc_pr.png", subtitle=tag)
    report.plot_calibration(y, oof, fig / "calibration.png")
    report.plot_spatial_vs_random(scores, scores_rand, fig / "spatial_vs_random_cv.png")
    report.plot_realistic_capacity(cap, fig / "realistic_capacity.png")
    report.plot_equity(eq, fig / "equity_by_opportunity.png")

    print("6/6 Writing tables and map ...")
    scores.to_csv(out / "metrics_spatial_cv.csv", index=False)
    scores_rand.to_csv(out / "metrics_random_cv.csv", index=False)
    cap.to_csv(out / "realistic_capacity_by_cpa.csv")
    eq.to_csv(out / "equity_by_opportunity.csv")
    importance.to_csv(out / "feature_importance_shap.csv", header=["mean_abs_shap"])
    preds = sites[["SITE_ID", "APN_8", "CPNAME", "NetUnits", "geometry"]].copy()
    preds["p_realized"] = p.values
    preds["expected_units"] = (p * sites["NetUnits"].clip(lower=0)).values
    preds["realized"] = y.values
    preds.to_file(out / "site_predictions.geojson", driver="GeoJSON")
    webmap.make_map(sites, p, y, out / "site_realization_map.html",
                    title=f"San Diego Housing Element sites: predicted realization{tag}")

    total = cap[["claimed_units", "expected_units"]].sum()
    summary = {
        "data": args.data, "n_sites": int(len(X)), "base_rate": float(y.mean()),
        "n_communities": int(groups.nunique()),
        "spatial_cv": scores.to_dict(orient="records"),
        "random_cv": scores_rand.to_dict(orient="records"),
        "claimed_units": float(total.claimed_units),
        "expected_units": float(total.expected_units),
        "top_features": importance.head(8).round(4).to_dict(),
    }
    if "_p_true" in sites:  # synthetic only: how close are we to the truth?
        from sklearn.metrics import roc_auc_score
        summary["oracle_auc_true_p"] = float(roc_auc_score(y, sites["_p_true"]))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    pd.set_option("display.width", 120)
    print("\nSpatial CV:\n", scores.round(3).to_string(index=False))
    print("\nRandom CV:\n", scores_rand.round(3).to_string(index=False))
    print(f"\nClaimed capacity {total.claimed_units:,.0f} units -> "
          f"model-expected {total.expected_units:,.0f} "
          f"({total.expected_units / total.claimed_units:.0%})")
    print(f"Outputs: {out}  Figures: {fig}")


if __name__ == "__main__":
    main()
