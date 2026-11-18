"""TV4 Phase 4: Spark MLlib GBT training and validation workflow.

Consumes frozen Phase-3 train/val feature tables only. Test data is deliberately
not accepted here. Reuses TV2's make_pipeline_gbt; feature engineering stays in
src/features.py.
"""
from __future__ import annotations

import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.storagelevel import StorageLevel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.features import SEED, make_pipeline_gbt


CATEGORICAL_COLUMNS = ["Airline", "DayOfWeek", "AirportFrom_g", "AirportTo_g"]
REQUIRED_TRAIN = {
    "row_id", "Delay", "fold", "Airline", "DayOfWeek", "AirportFrom_g",
    "AirportTo_g", "hour", "Length", "Length_invalid", "te_route",
    "te_flight", "log_freq_route", "density_from", "density_to",
}
REQUIRED_VAL = REQUIRED_TRAIN - {"fold"}
LOG_COLUMNS = [
    "run_id", "timestamp_utc", "stage", "status", "sample_fraction",
    "train_rows", "val_rows", "max_bins", "max_iter", "max_depth",
    "step_size", "subsampling_rate", "seed", "fit_seconds", "train_roc_auc",
    "val_roc_auc", "val_pr_auc", "selected_num_trees", "notes",
]


def _validate_inputs(train: DataFrame, val: DataFrame, max_bins: int) -> dict:
    for name, frame, required in (
        ("train", train, REQUIRED_TRAIN), ("val", val, REQUIRED_VAL)
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                f"{name} thiếu cột {missing}; cần dùng đúng bảng features_v1 đã freeze."
            )
        labels = {int(r[0]) for r in frame.select("Delay").distinct().collect()}
        if labels != {0, 1}:
            raise ValueError(f"{name}.Delay phải có đủ hai lớp 0/1; nhận được {labels}.")
    cardinalities = {}
    for col in CATEGORICAL_COLUMNS:
        # StringIndexer(handleInvalid='keep') reserves one additional unseen bucket.
        cardinalities[col] = int(train.select(col).distinct().count()) + 1
        if cardinalities[col] > max_bins:
            raise ValueError(
                f"maxBins={max_bins} < số category yêu cầu của {col}: "
                f"{cardinalities[col]} (đã cộng unseen bucket)."
            )
    return cardinalities


def _auc(frame: DataFrame) -> dict:
    result = {}
    for metric, key in (("areaUnderROC", "roc_auc"), ("areaUnderPR", "pr_auc")):
        evaluator = BinaryClassificationEvaluator(
            labelCol="Delay", rawPredictionCol="rawPrediction", metricName=metric
        )
        result[key] = float(evaluator.evaluate(frame))
    return result


def _threshold_metrics(frame: DataFrame) -> tuple[float, dict]:
    pdf = frame.select(
        F.col("Delay").cast("int").alias("label"),
        vector_to_array("probability")[1].cast("double").alias("score"),
    ).toPandas()
    y = pdf["label"].to_numpy(dtype=np.int32)
    score = pdf["score"].to_numpy(dtype=np.float64)
    precision, recall, thresholds = precision_recall_curve(y, score)
    if len(thresholds) == 0:
        threshold = 0.5
    else:
        denominator = precision[1:] + recall[1:]
        f1 = np.divide(
            2 * precision[1:] * recall[1:], denominator,
            out=np.zeros_like(denominator), where=denominator > 0,
        )
        threshold = float(thresholds[int(np.nanargmax(f1))])
    pred = (score >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    p = float(tp / (tp + fp)) if tp + fp else 0.0
    r = float(tp / (tp + fn)) if tp + fn else 0.0
    f = float(2 * p * r / (p + r)) if p + r else 0.0
    return threshold, {
        "accuracy": float((tp + tn) / len(y)),
        "precision": p,
        "recall": r,
        "f1": f,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _append_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{c: row.get(c) for c in LOG_COLUMNS}]).to_csv(
        path, mode="a", header=not path.exists(), index=False
    )


def _log_row(run_id: str, stage: str, fraction: float, n_train: int,
             n_val: int, max_bins: int, max_iter: int, depth: int,
             step: float, subsampling: float, seed: int) -> dict:
    return {
        "run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage, "status": "ok", "sample_fraction": fraction,
        "train_rows": n_train, "val_rows": n_val, "max_bins": max_bins,
        "max_iter": max_iter, "max_depth": depth, "step_size": step,
        "subsampling_rate": subsampling, "seed": seed, "fit_seconds": None,
        "train_roc_auc": None, "val_roc_auc": None, "val_pr_auc": None,
        "selected_num_trees": None, "notes": "",
    }


def _pipeline(max_bins: int, max_iter: int, depth: int, step: float,
              subsampling: float, seed: int):
    # Keep StringIndexer inside TV2's pipeline so nominal metadata reaches GBT.
    return make_pipeline_gbt(
        max_bins=max_bins, max_iter=max_iter, max_depth=depth,
        step_size=step, subsampling_rate=subsampling, seed=seed,
        include_length_invalid=True, include_estimator=True,
    )


def train_gbt(
    train_df: DataFrame,
    val_df: DataFrame,
    model_dir: str | Path,
    results_dir: str | Path,
    *,
    sample_fraction: float = 0.25,
    max_bins: int = 300,
    max_iter: int = 100,
    seed: int = SEED,
    top_k: int = 3,
    max_depth_values: Sequence[int] = (3, 5, 7),
    step_size_values: Sequence[float] = (0.05, 0.1, 0.2),
    subsampling_values: Sequence[float] = (0.7, 1.0),
) -> dict:
    """Run coarse tuning on a train sample, then refine top candidates on full train.

    Selects the number of trees using minimum Validation loss from
    GBTClassificationModel.evaluateEachIteration when that API is available.
    Selects the classification threshold by maximum Validation F1.
    """
    if not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction phải thuộc (0, 1].")
    if top_k < 1:
        raise ValueError("top_k phải >= 1.")

    model_dir, results_dir = Path(model_dir), Path(results_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "gbt_experiments.csv"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cardinalities = _validate_inputs(train_df, val_df, max_bins)
    n_train, n_val = int(train_df.count()), int(val_df.count())
    if not n_train or not n_val:
        raise ValueError("Train và Validation không được rỗng.")

    val = val_df.persist(StorageLevel.MEMORY_AND_DISK)
    val.count()
    sample = train_df.sample(False, sample_fraction, seed).persist(
        StorageLevel.MEMORY_AND_DISK
    )
    n_sample = int(sample.count())
    if not n_sample:
        sample.unpersist()
        val.unpersist()
        raise ValueError("Mẫu Train rỗng; tăng sample_fraction.")
    if {int(r[0]) for r in sample.select("Delay").distinct().collect()} != {0, 1}:
        sample.unpersist()
        val.unpersist()
        raise ValueError("Mẫu Train phải chứa cả hai lớp; tăng sample_fraction.")

    coarse = []
    grid = list(itertools.product(max_depth_values, step_size_values, subsampling_values))
    print(f"[TV4] Coarse search: {len(grid)} configs, {n_sample:,} sample rows.")
    try:
        for i, (depth, step, subsampling) in enumerate(grid, 1):
            row = _log_row(
                run_id, "coarse_sample", sample_fraction, n_sample, n_val,
                max_bins, max_iter, int(depth), float(step),
                float(subsampling), seed,
            )
            started = time.perf_counter()
            try:
                model = _pipeline(
                    max_bins, max_iter, int(depth), float(step),
                    float(subsampling), seed,
                ).fit(sample)
                fit_seconds = time.perf_counter() - started
                train_metrics = _auc(model.transform(sample))
                val_metrics = _auc(model.transform(val))
                row.update({
                    "fit_seconds": round(fit_seconds, 3),
                    "train_roc_auc": train_metrics["roc_auc"],
                    "val_roc_auc": val_metrics["roc_auc"],
                    "val_pr_auc": val_metrics["pr_auc"],
                    "notes": f"coarse {i}/{len(grid)}",
                })
                coarse.append({
                    "max_depth": int(depth), "step_size": float(step),
                    "subsampling_rate": float(subsampling),
                    "val_roc_auc": val_metrics["roc_auc"],
                    "val_pr_auc": val_metrics["pr_auc"],
                })
                print(
                    f"[TV4] {i}/{len(grid)} depth={depth}, step={step}, "
                    f"subsample={subsampling}: ROC-AUC={val_metrics['roc_auc']:.5f}, "
                    f"PR-AUC={val_metrics['pr_auc']:.5f}, fit={fit_seconds:.1f}s"
                )
                del model
            except Exception as exc:
                row["status"] = "failed"
                row["fit_seconds"] = round(time.perf_counter() - started, 3)
                row["notes"] = f"{type(exc).__name__}: {str(exc)[:500]}"
                print("[TV4][WARN]", row["notes"])
            _append_log(log_path, row)
    finally:
        sample.unpersist()

    if not coarse:
        val.unpersist()
        raise RuntimeError("Tất cả cấu hình coarse đều lỗi; xem gbt_experiments.csv.")

    ranked = sorted(
        coarse, key=lambda x: (x["val_roc_auc"], x["val_pr_auc"]), reverse=True
    )[:min(top_k, len(coarse))]
    full_train = train_df.persist(StorageLevel.MEMORY_AND_DISK)
    full_train.count()
    refined, loss_rows = [], []

    try:
        for i, candidate in enumerate(ranked, 1):
            depth = candidate["max_depth"]
            step = candidate["step_size"]
            subsampling = candidate["subsampling_rate"]
            row = _log_row(
                run_id, "full_train_refine", 1.0, n_train, n_val,
                max_bins, max_iter, depth, step, subsampling, seed,
            )
            started = time.perf_counter()
            try:
                fitted = _pipeline(
                    max_bins, max_iter, depth, step, subsampling, seed
                ).fit(full_train)
                fit_seconds = time.perf_counter() - started
                val_transformed = fitted.transform(val)
                val_metrics = _auc(val_transformed)
                losses = None
                loss_note = ""
                try:
                    prep = PipelineModel(stages=fitted.stages[:-1])
                    val_prepared = prep.transform(val)
                    losses = [
                        float(x) for x in fitted.stages[-1].evaluateEachIteration(val_prepared)
                    ]
                    if not losses or not np.isfinite(losses).all():
                        raise ValueError("evaluateEachIteration returned invalid values")
                    selected_trees = int(np.argmin(losses)) + 1
                    min_loss = float(min(losses))
                    for iteration, loss in enumerate(losses, 1):
                        loss_rows.append({
                            "candidate": i, "max_depth": depth, "step_size": step,
                            "subsampling_rate": subsampling, "iteration": iteration,
                            "validation_loss": loss,
                            "selected_iteration": iteration == selected_trees,
                        })
                except Exception as exc:
                    selected_trees, min_loss = max_iter, None
                    loss_note = (
                        "evaluateEachIteration failed; fallback max_iter. "
                        f"{type(exc).__name__}: {str(exc)[:250]}"
                    )
                row.update({
                    "fit_seconds": round(fit_seconds, 3),
                    "val_roc_auc": val_metrics["roc_auc"],
                    "val_pr_auc": val_metrics["pr_auc"],
                    "selected_num_trees": selected_trees,
                    "notes": loss_note or f"full-train candidate {i}",
                })
                refined.append({
                    **candidate, "selected_num_trees": selected_trees,
                    "validation_loss_min": min_loss,
                    "val_roc_auc": val_metrics["roc_auc"],
                    "val_pr_auc": val_metrics["pr_auc"],
                    "loss_selection_available": losses is not None,
                })
                print(
                    f"[TV4] Refine {i}/{len(ranked)}: depth={depth}, step={step}, "
                    f"subsample={subsampling}, trees={selected_trees}, loss={min_loss}"
                )
                del fitted, val_transformed
            except Exception as exc:
                row["status"] = "failed"
                row["fit_seconds"] = round(time.perf_counter() - started, 3)
                row["notes"] = f"{type(exc).__name__}: {str(exc)[:500]}"
                print("[TV4][WARN]", row["notes"])
            _append_log(log_path, row)

        if loss_rows:
            pd.DataFrame(loss_rows).to_csv(
                results_dir / "gbt_iteration_loss.csv", index=False
            )
        if not refined:
            raise RuntimeError("Tất cả cấu hình full-train refinement đều lỗi.")

        with_loss = [x for x in refined if x["loss_selection_available"]]
        if with_loss:
            winner = min(
                with_loss,
                key=lambda x: (x["validation_loss_min"], -x["val_roc_auc"]),
            )
        else:
            winner = max(refined, key=lambda x: (x["val_roc_auc"], x["val_pr_auc"]))

        params = {
            "max_depth": winner["max_depth"],
            "step_size": winner["step_size"],
            "subsampling_rate": winner["subsampling_rate"],
            "max_iter": winner["selected_num_trees"],
        }
        final_started = time.perf_counter()
        final_model = _pipeline(
            max_bins, params["max_iter"], params["max_depth"],
            params["step_size"], params["subsampling_rate"], seed,
        ).fit(full_train)
        final_fit_seconds = time.perf_counter() - final_started

        train_pred = final_model.transform(full_train).persist(
            StorageLevel.MEMORY_AND_DISK
        )
        val_pred = final_model.transform(val).persist(StorageLevel.MEMORY_AND_DISK)
        train_metrics, val_metrics = _auc(train_pred), _auc(val_pred)
        threshold, threshold_metrics = _threshold_metrics(val_pred)
        gap = train_metrics["roc_auc"] - val_metrics["roc_auc"]

        predictions_path = results_dir / "predictions_val_gbt.parquet"
        val_pred.select(
            F.col("row_id").cast("long"),
            F.col("Delay").cast("int").alias("label"),
            vector_to_array("probability")[1].cast("double").alias("score"),
        ).write.mode("overwrite").parquet(str(predictions_path))

        model_path = model_dir / "best_model"
        final_model.write().overwrite().save(str(model_path))
        config = {
            "model_name": "Spark MLlib GBTClassifier",
            "phase": 4, "seed": int(seed), "max_bins": int(max_bins),
            "categorical_cardinality_plus_unseen": cardinalities,
            "selected_params": params, "threshold": float(threshold),
            "validation_metrics": {**val_metrics, **threshold_metrics},
            "train_metrics": train_metrics,
            "train_minus_validation_roc_auc": float(gap),
            "overfit_warning": bool(gap > 0.05),
            "validation_loss_min": winner["validation_loss_min"],
            "validation_loss_method": (
                "minimum evaluateEachIteration validation loss"
                if winner["loss_selection_available"]
                else "fallback: maximum validation ROC-AUC; evaluateEachIteration unavailable"
            ),
            "feature_columns": [
                "Airline", "DayOfWeek", "AirportFrom_g", "AirportTo_g", "hour",
                "Length", "te_route", "te_flight", "log_freq_route", "density_from",
                "density_to", "Length_invalid",
            ],
            "test_used": False, "run_id": run_id,
            "model_path": str(model_path),
            "predictions_val_path": str(predictions_path),
            "final_fit_seconds": round(final_fit_seconds, 3),
        }
        config_path = model_dir / "config.json"
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        final_row = _log_row(
            run_id, "final_full_train", 1.0, n_train, n_val, max_bins,
            params["max_iter"], params["max_depth"], params["step_size"],
            params["subsampling_rate"], seed,
        )
        final_row.update({
            "fit_seconds": round(final_fit_seconds, 3),
            "train_roc_auc": train_metrics["roc_auc"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "selected_num_trees": params["max_iter"],
            "notes": f"threshold={threshold:.8f}; train_val_auc_gap={gap:.6f}",
        })
        _append_log(log_path, final_row)
        print(
            f"[TV4] Final validation ROC-AUC={val_metrics['roc_auc']:.5f}; "
            f"PR-AUC={val_metrics['pr_auc']:.5f}; F1={threshold_metrics['f1']:.5f}; "
            f"threshold={threshold:.5f}; train-val gap={gap:.5f}"
        )
        if gap > 0.05:
            print("[TV4][WARN] Train AUC cao hơn Validation trên 0.05; cần thảo luận overfitting.")
        train_pred.unpersist()
        val_pred.unpersist()
        return {
            "model": final_model, "config": config,
            "train_metrics": train_metrics,
            "validation_metrics": {**val_metrics, **threshold_metrics},
            "threshold": threshold, "overfit_gap": gap,
            "experiment_log": str(log_path),
            "iteration_loss_log": str(results_dir / "gbt_iteration_loss.csv"),
            "model_path": str(model_path), "config_path": str(config_path),
            "predictions_path": str(predictions_path),
        }
    finally:
        full_train.unpersist()
        val.unpersist()


def permutation_importance_val(
    model, val_df: DataFrame, results_dir: str | Path,
    *, repeats: int = 3, seed: int = SEED,
) -> pd.DataFrame:
    """Grouped permutation importance: decrease in Validation ROC-AUC."""
    if repeats < 1:
        raise ValueError("repeats phải >= 1.")
    missing = sorted(REQUIRED_VAL - set(val_df.columns))
    if missing:
        raise ValueError(f"Validation thiếu cột: {missing}")

    groups = {
        "schedule": ["DayOfWeek", "hour"],
        "airline": ["Airline"],
        "airports": ["AirportFrom_g", "AirportTo_g"],
        "route_history": ["te_route", "log_freq_route"],
        "flight_history": ["te_flight"],
        "airport_density": ["density_from", "density_to"],
        "flight_length": ["Length", "Length_invalid"],
    }
    baseline = _auc(model.transform(val_df))["roc_auc"]
    original = val_df.toPandas()
    rng, records = np.random.default_rng(seed), []
    for group, columns in groups.items():
        for repeat in range(1, repeats + 1):
            order = rng.permutation(len(original))
            shuffled = original.copy(deep=True)
            shuffled.loc[:, columns] = original.iloc[order][columns].to_numpy()
            permuted = val_df.sparkSession.createDataFrame(shuffled, schema=val_df.schema)
            score = _auc(model.transform(permuted))["roc_auc"]
            records.append({
                "feature_group": group, "columns": ",".join(columns),
                "repeat": repeat, "baseline_val_roc_auc": baseline,
                "permuted_val_roc_auc": score,
                "roc_auc_decrease": baseline - score, "seed": seed,
            })
            print(
                f"[TV4] permutation {group} {repeat}/{repeats}: "
                f"AUC decrease={baseline - score:+.6f}"
            )
            del permuted

    result = pd.DataFrame(records)
    summary = result.groupby(
        ["feature_group", "columns"], as_index=False
    ).agg(
        mean_auc_decrease=("roc_auc_decrease", "mean"),
        std_auc_decrease=("roc_auc_decrease", "std"),
        baseline_val_roc_auc=("baseline_val_roc_auc", "mean"),
    ).sort_values("mean_auc_decrease", ascending=False)
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "gbt_permutation_importance_repeats.csv", index=False)
    summary.to_csv(out / "gbt_permutation_importance.csv", index=False)
    return summary
