"""TV3 - Random Forest bằng scikit-learn, không dùng Spark.

Cùng FEATURE_COLS như LR để việc so sánh chỉ thay đổi thuật toán.
Chạy: python -m src.model_rf --features-dir /duong/dan/features_v1 --output-root .
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .tv3_common import (
    SEED, FEATURE_COLS, TARGET, append_csv, assert_disjoint,
    calculate_metrics, choose_best_threshold, make_preprocessor,
    read_split, save_artifacts,
)

LOG_COLUMNS = [
    "model", "n_estimators", "max_depth", "min_samples_leaf", "max_features",
    "roc_auc", "pr_auc", "f1", "threshold", "fit_seconds", "n_train", "n_val", "seed",
]


def build_rf_pipeline(n_estimators: int = 100, max_depth: int = 12,
                      min_samples_leaf: int = 10, max_features: str = "sqrt") -> Pipeline:
    return Pipeline([
        ("preprocessor", make_preprocessor(for_logistic=False)),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf, max_features=max_features,
            random_state=SEED, n_jobs=-1,
        )),
    ])


def save_rf_importances(model: Pipeline, output_root: str | Path) -> None:
    """Feature importance (impurity) chỉ để tham khảo, không phải quan hệ nhân quả."""
    names = model.named_steps["preprocessor"].get_feature_names_out()
    scores = model.named_steps["classifier"].feature_importances_
    report = pd.DataFrame({"transformed_feature": names, "importance": scores})
    report.sort_values("importance", ascending=False).to_csv(
        Path(output_root) / "results" / "rf_feature_importances.csv", index=False
    )


def train_rf(features_dir: str | Path, output_root: str | Path = ".",
             check_counts: bool = True,
             search_grid: list[dict] | None = None) -> dict:
    root = Path(output_root)
    train = read_split(features_dir, "train", check_counts)
    val = read_split(features_dir, "val", check_counts)
    assert_disjoint(train, val)
    X_train, y_train = train[FEATURE_COLS], train[TARGET].to_numpy()
    X_val, y_val = val[FEATURE_COLS], val[TARGET].to_numpy()

    if search_grid is None:
        # Lưới gọn để chạy được trên Colab CPU: có thể tăng n_estimators sau.
        search_grid = [
            {"n_estimators": 80, "max_depth": 10, "min_samples_leaf": 20},
            {"n_estimators": 80, "max_depth": 16, "min_samples_leaf": 20},
            {"n_estimators": 120, "max_depth": 12, "min_samples_leaf": 10},
            {"n_estimators": 120, "max_depth": 16, "min_samples_leaf": 10},
        ]

    best = None
    log_path = root / "results" / "rf_experiments.csv"
    for i, p in enumerate(search_grid, start=1):
        print(f"[RF] Thử {i}/{len(search_grid)}: {p}")
        model = build_rf_pipeline(**p)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        val_score = model.predict_proba(X_val)[:, 1]
        thr, _ = choose_best_threshold(y_val, val_score)
        metrics = calculate_metrics(y_val, val_score, thr)
        row = {
            "model": "rf", "n_estimators": p["n_estimators"],
            "max_depth": p["max_depth"], "min_samples_leaf": p["min_samples_leaf"],
            "max_features": "sqrt", "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"], "f1": metrics["f1"],
            "threshold": thr, "fit_seconds": round(elapsed, 3),
            "n_train": len(train), "n_val": len(val), "seed": SEED,
        }
        append_csv(log_path, row, LOG_COLUMNS)
        print(f"  ROC-AUC={metrics['roc_auc']:.5f}, PR-AUC={metrics['pr_auc']:.5f}, "
              f"F1={metrics['f1']:.5f}, threshold={thr:.4f}, fit={elapsed:.1f}s")
        rank = (metrics["roc_auc"], metrics["pr_auc"], metrics["f1"])
        if best is None or rank > best[0]:
            best = (rank, model, val_score, thr, metrics,
                    {**p, "max_features": "sqrt"})

    if best is None:
        raise RuntimeError("Chưa có cấu hình RF nào được huấn luyện")
    _, model, score, threshold, metrics, params = best
    save_artifacts(model, root, "rf", val, score, threshold, metrics, params)
    save_rf_importances(model, root)
    print("[RF] Đã lưu models/rf/, results/ và reports/tv3/. Chưa chạm Test.")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Random Forest TV3")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--output-root", default=".")
    args = parser.parse_args()
    train_rf(args.features_dir, args.output_root)


if __name__ == "__main__":
    main()
