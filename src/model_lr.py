"""TV3 - Logistic Regression bằng scikit-learn, không dùng Spark.

Chạy: python -m src.model_lr --features-dir /duong/dan/features_v1 --output-root .
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .tv3_common import (
    SEED, FEATURE_COLS, TARGET, append_csv, assert_disjoint, calculate_metrics,
    choose_best_threshold, evaluate_baselines, make_preprocessor, read_split,
    save_artifacts,
)

LOG_COLUMNS = [
    "model", "C", "l1_ratio", "max_iter", "roc_auc", "pr_auc", "f1",
    "threshold", "fit_seconds", "n_train", "n_val", "seed",
]


def build_lr_pipeline(C: float = 1.0, l1_ratio: float = 0.0) -> Pipeline:
    """LR: one-hot Airline/DayOfWeek/hour_cat, impute + scale biến số."""
    if not 0 <= l1_ratio <= 1:
        raise ValueError("l1_ratio phải từ 0 đến 1")
    # scikit-learn 1.8 ngừng khuyến nghị tham số penalty; hỗ trợ cả bản cũ hơn.
    version_parts = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    kwargs = {"solver": "saga", "C": C, "max_iter": 500,
              "tol": 1e-3, "random_state": SEED}
    if version_parts >= (1, 8):
        kwargs["l1_ratio"] = l1_ratio
    else:
        if l1_ratio == 0:
            kwargs.update({"penalty": "l2"})
        elif l1_ratio == 1:
            kwargs.update({"penalty": "l1"})
        else:
            kwargs.update({"penalty": "elasticnet", "l1_ratio": l1_ratio})
    estimator = LogisticRegression(**kwargs)
    return Pipeline([
        ("preprocessor", make_preprocessor(for_logistic=True)),
        ("classifier", estimator),
    ])


def save_lr_coefficients(model: Pipeline, output_root: str | Path) -> None:
    """Xuất odds ratio: số đã chuẩn hóa là mỗi +1 độ lệch chuẩn."""
    try:
        names = model.named_steps["preprocessor"].get_feature_names_out()
        coef = model.named_steps["classifier"].coef_.ravel()
        report = pd.DataFrame({
            "transformed_feature": names,
            "coefficient": coef,
            "odds_ratio": np.exp(np.clip(coef, -30, 30)),
        }).sort_values("coefficient", key=np.abs, ascending=False)
        out = Path(output_root) / "results" / "lr_coefficients.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out, index=False)
    except Exception as exc:
        print(f"[WARN] Chưa thể xuất hệ số: {exc}")


def train_lr(features_dir: str | Path, output_root: str | Path = ".",
             check_counts: bool = True,
             search_grid: list[tuple[float, float]] | None = None) -> dict:
    root = Path(output_root)
    train = read_split(features_dir, "train", check_counts)
    val = read_split(features_dir, "val", check_counts)
    assert_disjoint(train, val)

    (root / "results").mkdir(parents=True, exist_ok=True)
    baseline_path = root / "results" / "tv3_baselines.csv"
    evaluate_baselines(train, val).to_csv(baseline_path, index=False)
    print("=== 2 BASELINE (validation) ===")
    print(pd.read_csv(baseline_path)[["baseline", "accuracy", "f1", "roc_auc", "pr_auc"]].to_string(index=False))

    X_train, y_train = train[FEATURE_COLS], train[TARGET].to_numpy()
    X_val, y_val = val[FEATURE_COLS], val[TARGET].to_numpy()
    if search_grid is None:
        # C lớn hơn = chính quy hóa yếu hơn. l1_ratio=0 là L2; 0.5 là Elastic Net; 1 là L1.
        search_grid = [(0.1, 0.0), (1.0, 0.0), (10.0, 0.0),
                       (0.1, 0.5), (1.0, 0.5), (1.0, 1.0)]

    best = None
    log_path = root / "results" / "lr_experiments.csv"
    for i, (C, ratio) in enumerate(search_grid, start=1):
        print(f"[LR] Thử {i}/{len(search_grid)}: C={C}, l1_ratio={ratio}")
        model = build_lr_pipeline(C, ratio)
        start = time.perf_counter()
        model.fit(X_train, y_train)  # Mọi transformer được fit CHỈ trên train.
        elapsed = time.perf_counter() - start
        val_score = model.predict_proba(X_val)[:, 1]
        thr, _ = choose_best_threshold(y_val, val_score)
        metrics = calculate_metrics(y_val, val_score, thr)
        row = {
            "model": "lr", "C": C, "l1_ratio": ratio, "max_iter": 500,
            "roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"],
            "f1": metrics["f1"], "threshold": thr, "fit_seconds": round(elapsed, 3),
            "n_train": len(train), "n_val": len(val), "seed": SEED,
        }
        append_csv(log_path, row, LOG_COLUMNS)
        print(f"  ROC-AUC={metrics['roc_auc']:.5f}, PR-AUC={metrics['pr_auc']:.5f}, "
              f"F1={metrics['f1']:.5f}, threshold={thr:.4f}, fit={elapsed:.1f}s")
        rank = (metrics["roc_auc"], metrics["pr_auc"], metrics["f1"])
        if best is None or rank > best[0]:
            best = (rank, model, val_score, thr, metrics,
                    {"C": C, "l1_ratio": ratio, "solver": "saga", "max_iter": 500})

    if best is None:
        raise RuntimeError("Chưa có cấu hình LR nào được huấn luyện")
    _, model, score, threshold, metrics, params = best
    save_artifacts(model, root, "lr", val, score, threshold, metrics, params)
    save_lr_coefficients(model, root)
    print("[LR] Đã lưu models/lr/, results/ và reports/tv3/. Chưa chạm Test.")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Logistic Regression TV3")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--output-root", default=".")
    args = parser.parse_args()
    train_lr(args.features_dir, args.output_root)


if __name__ == "__main__":
    main()
