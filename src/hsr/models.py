"""Models and spatially honest cross-validation.

Three scorers are compared on identical folds:

* capacity_baseline - the implicit Housing Element assumption: sites with more
  zoned capacity are the likeliest to be built. Score = log net units.
* logistic          - regularized logistic regression (interpretable benchmark).
* gbm               - LightGBM gradient boosting with isotonic calibration, so
                      outputs can be read as probabilities and summed into
                      "expected units".

Cross-validation holds out whole community plan areas (GroupKFold). A model
must predict neighborhoods it has never seen, which is what a planner needs.
Random K-fold is also run so the optimism from spatial leakage can be shown.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")


def gbm_params() -> dict:
    return dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=config.RANDOM_STATE, verbose=-1)


def make_model(name: str):
    if name == "logistic":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.5, max_iter=2000))
    if name == "gbm":
        return CalibratedClassifierCV(lgb.LGBMClassifier(**gbm_params()),
                                      method="isotonic", cv=3)
    raise ValueError(name)


def cross_validate(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                   scheme: str = "spatial") -> pd.DataFrame:
    """Return out-of-fold predictions for every scorer."""
    if scheme == "spatial":
        splits = GroupKFold(n_splits=config.N_FOLDS).split(X, y, groups)
    else:
        splits = StratifiedKFold(config.N_FOLDS, shuffle=True,
                                 random_state=config.RANDOM_STATE).split(X, y)
    oof = pd.DataFrame(index=X.index, columns=["fold", "capacity_baseline", "logistic", "gbm"],
                       dtype=float)
    for k, (tr, te) in enumerate(splits):
        oof.iloc[te, 0] = k
        oof.iloc[te, 1] = X["log_net_units"].iloc[te].values
        for name in ["logistic", "gbm"]:
            m = make_model(name).fit(X.iloc[tr], y.iloc[tr])
            oof.loc[X.index[te], name] = m.predict_proba(X.iloc[te])[:, 1]
    return oof


def _bootstrap_ci(y, s, fn, n=500, seed=config.RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(fn(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


def score_table(y: pd.Series, oof: pd.DataFrame, ci: bool = True) -> pd.DataFrame:
    rows = []
    for name in ["capacity_baseline", "logistic", "gbm"]:
        s = oof[name].astype(float)
        row = {"model": name,
               "roc_auc": roc_auc_score(y, s),
               "pr_auc": average_precision_score(y, s),
               "brier": brier_score_loss(y, s) if name != "capacity_baseline" else np.nan}
        if ci:
            row["roc_auc_95ci"] = "[{:.3f}, {:.3f}]".format(*_bootstrap_ci(y, s, roc_auc_score))
            row["pr_auc_95ci"] = "[{:.3f}, {:.3f}]".format(
                *_bootstrap_ci(y, s, average_precision_score))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["base_rate"] = float(np.mean(y))
    return out


def fit_final_gbm(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    """Uncalibrated booster on all data, used for SHAP explanations."""
    return lgb.LGBMClassifier(**gbm_params()).fit(X, y)
