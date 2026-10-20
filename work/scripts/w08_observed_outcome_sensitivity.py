"""Aggregate sensitivity check for pages with observed March and April totals.

Run after ``w05_model.ipynb`` has produced its ignored feature-vector cache.
Only aggregate metrics are written; no page or client rows leave the cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "work/outputs/model_feature_vector.csv"
OUTPUT = ROOT / "work/outputs/w08_observed_outcome_sensitivity.json"


def precision_at_50(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(labels[np.argsort(-scores)[:50]].mean())


def main() -> None:
    if not CACHE.exists():
        raise FileNotFoundError("Run work/notebooks/w05_model.ipynb to create its ignored cache first")

    frame = pd.read_csv(CACHE)
    original_rows = len(frame)
    observed = frame.loc[
        frame["gsc_impressions_last_30d"].notna()
        & frame["gsc_impressions_next_30d"].notna()
    ].copy()
    observed["main_intent"] = observed["main_intent"].fillna("unknown")
    observed["recent_share"] = (
        observed["gsc_impressions_last_30d"] / observed["gsc_impressions_90d"]
    ).clip(0, 1).fillna(0)
    observed["log_gsc_impressions_90d"] = np.log1p(observed["gsc_impressions_90d"])
    observed["log_gsc_impressions_last_30d"] = np.log1p(observed["gsc_impressions_last_30d"])
    observed["momentum_x_volume"] = (
        observed["recent_share"] * observed["log_gsc_impressions_90d"]
    )

    numeric = [
        "gsc_impressions_90d", "gsc_avg_position_90d", "gsc_clicks_90d",
        "gsc_ctr_90d", "sessions_organic_90d", "days_with_impressions_90d",
        "gsc_impressions_last_30d", "search_volume", "competition", "cpc",
        "word_count", "content_age_days", "recent_share",
        "log_gsc_impressions_90d", "log_gsc_impressions_last_30d",
        "momentum_x_volume",
    ]
    encoded = pd.get_dummies(
        observed, columns=["content_type", "main_intent"],
        prefix=["content_type", "main_intent"], dtype=int,
    )
    features = numeric + [
        name for name in encoded.columns
        if name.startswith("content_type_") or name.startswith("main_intent_")
    ]
    x = encoded[features].fillna(0).replace([np.inf, -np.inf], 0)
    y = observed["growth_label"].to_numpy(dtype=int)
    groups = observed["client_hash_id"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    folds = []
    for number, (train, test) in enumerate(cv.split(x, y, groups), 1):
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
        )
        model.fit(x.iloc[train], y[train], sample_weight=compute_sample_weight("balanced", y[train]))
        model_scores = model.predict_proba(x.iloc[test])[:, 1]
        held_out = observed.iloc[test]
        rule_scores = np.where(
            held_out["gsc_impressions_90d"].to_numpy() >= 500,
            held_out["recent_share"].to_numpy()
            * np.log1p(held_out["gsc_impressions_90d"].to_numpy()),
            0.0,
        )
        folds.append({
            "fold": number,
            "gradient_boosting_precision_at_50": precision_at_50(y[test], model_scores),
            "baseline_rule_precision_at_50": precision_at_50(y[test], rule_scores),
        })

    receipt = {
        "analysis": "post_hoc_observed_march_and_april_only",
        "original_rows": original_rows,
        "observed_rows": len(observed),
        "excluded_rows": original_rows - len(observed),
        "client_groups": int(observed["client_hash_id"].nunique()),
        "base_growth_rate": float(y.mean()),
        "split": "5-fold StratifiedGroupKFold by client_hash_id, random_state=42",
        "metric": "mean held-out precision at 50 per fold",
        "gradient_boosting_mean_precision_at_50": float(np.mean([
            f["gradient_boosting_precision_at_50"] for f in folds
        ])),
        "baseline_rule_mean_precision_at_50": float(np.mean([
            f["baseline_rule_precision_at_50"] for f in folds
        ])),
        "folds": folds,
        "note": "Sensitivity to absent monthly aggregates; post hoc, not a future-time holdout.",
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2) + "\n")
    print("Aggregate observed-outcome sensitivity receipt written.")
    print(f"Observed pages: {len(observed):,}; excluded: {original_rows - len(observed):,}")
    print("Mean precision@50: model {:.3f}; rule {:.3f}".format(
        receipt["gradient_boosting_mean_precision_at_50"],
        receipt["baseline_rule_mean_precision_at_50"],
    ))


if __name__ == "__main__":
    main()
