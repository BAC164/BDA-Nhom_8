import os
import pandas as pd
from pyspark.sql import DataFrame, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array


def run_ablation_study_gbt(
    train_fe: DataFrame,
    val_fe: DataFrame,
    max_bins: int = 300,
    max_iter: int = 50,
    max_depth: int = 5,
    step_size: float = 0.1,
    seed: int = 42,
    output_csv: str = "outputs/ablation.csv"
) -> pd.DataFrame:
    """
    Chạy 5 cấu hình Ablation trên tập Validation cho mô hình GBT theo đúng mục 5C.
    """
    cat_cols = ["Airline", "DayOfWeek", "AirportFrom_g", "AirportTo_g"]
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_i", handleInvalid="keep") for c in cat_cols]
    cat_idx_cols = [f"{c}_i" for c in cat_cols]

    configs = [
        (
            "0. Full Model (Đầy đủ đặc trưng v1)",
            cat_idx_cols + ["hour", "Length", "te_route", "te_flight", "log_freq_route", "density_from", "density_to"]
        ),
        (
            "1. Bỏ Target Encoding (-te_route, -te_flight)",
            cat_idx_cols + ["hour", "Length", "log_freq_route", "density_from", "density_to"]
        ),
        (
            "2. Thay Target Encoding bằng Frequency Encoding",
            cat_idx_cols + ["hour", "Length", "log_freq_from", "log_freq_to", "log_freq_route", "log_freq_flight", "density_from", "density_to"]
        ),
        (
            "3. Bỏ Mật độ chuyến bay (-density_from, -density_to)",
            cat_idx_cols + ["hour", "Length", "te_route", "te_flight", "log_freq_route"]
        ),
        (
            "4. Bỏ các đặc trưng Route (-te_route, -log_freq_route)",
            cat_idx_cols + ["hour", "Length", "te_flight", "density_from", "density_to"]
        ),
    ]

    roc_eval = BinaryClassificationEvaluator(labelCol="Delay", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    pr_eval = BinaryClassificationEvaluator(labelCol="Delay", rawPredictionCol="rawPrediction", metricName="areaUnderPR")

    rows = []
    full_roc = None

    for name, feature_list in configs:
        asm = VectorAssembler(inputCols=feature_list, outputCol="features")
        gbt = GBTClassifier(
            labelCol="Delay", featuresCol="features",
            maxBins=max_bins, maxIter=max_iter, maxDepth=max_depth, stepSize=step_size, seed=seed
        )
        pipe = Pipeline(stages=indexers + [asm, gbt])
        model = pipe.fit(train_fe)
        preds = model.transform(val_fe)

        roc_auc = float(roc_eval.evaluate(preds))
        pr_auc = float(pr_eval.evaluate(preds))
        if full_roc is None:
            full_roc = roc_auc

        rows.append({
            "Ablation_Variant": name,
            "Num_Features": len(feature_list),
            "Val_ROC_AUC": round(roc_auc, 4),
            "Val_PR_AUC": round(pr_auc, 4),
            "Delta_ROC_AUC": round(roc_auc - full_roc, 4)
        })
        print(f"[ABLATION] {name:<52} | ROC-AUC: {roc_auc:.4f} (Delta: {roc_auc - full_roc:+.4f})")

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    return df_out


def run_slice_error_analysis(predictions_val: DataFrame, slice_col: str, min_count: int = 500) -> pd.DataFrame:
  
    pred_with_prob = predictions_val.withColumn("prob_delay", vector_to_array(F.col("probability"))[1])
    slice_df = (
        pred_with_prob.groupBy(slice_col)
        .agg(
            F.count("*").alias("total_flights"),
            F.avg(F.col("Delay").cast("double")).alias("actual_delay_rate"),
            F.sum(F.when((F.col("Delay") == 1) & (F.col("prediction") == 1.0), 1).otherwise(0)).alias("TP"),
            F.sum(F.when((F.col("Delay") == 1) & (F.col("prediction") == 0.0), 1).otherwise(0)).alias("FN"),
            F.sum(F.when((F.col("Delay") == 0) & (F.col("prediction") == 1.0), 1).otherwise(0)).alias("FP"),
            F.sum(F.when((F.col("Delay") == 0) & (F.col("prediction") == 0.0), 1).otherwise(0)).alias("TN"),
        )
        .filter(F.col("total_flights") >= min_count)
        .withColumn("recall", F.col("TP") / (F.col("TP") + F.col("FN")))
        .withColumn("precision", F.col("TP") / (F.col("TP") + F.col("FP")))
        .withColumn("accuracy", (F.col("TP") + F.col("TN")) / F.col("total_flights"))
        .orderBy(F.asc("recall"))
        .toPandas()
    )
    return slice_df
