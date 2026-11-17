"""
src/features.py — Module Kỹ thuật Đặc trưng & Pipeline (Chủ sở hữu: TV2 - Phase 3)
Đề tài BDA_20 (Nhóm 8): Dự báo chuyến bay bị trễ (OpenML ID 1169 - Airlines)

Đầu vào (từ Phase 2 - TV1 / bản bàn giao tv1_v1):
  - splits/{train,val,test} (Parquet)
    Các cột: row_id (long), Airline (string), Flight (string), AirportFrom (string),
             AirportTo (string), DayOfWeek (string), Time (double), Length (double),
             Delay (int: 0/1), Length_invalid (int: 0/1), và cột fold (int: 0..4, chỉ có trong train).

Đầu ra (Phase 3 - Data Freeze v1):
  1. Bảng đặc trưng Parquet: data/features/v1/{train,val,test}/
  2. Bảng tra cứu (Lookup Maps) cho Demo (Phase 6): data/features/v1/lookup_maps/
  3. Hai hàm tạo Pipeline chuẩn cho TV3 & TV4:
     - make_pipeline_lr(...)
     - make_pipeline_gbt(max_bins=300, ...)
  4. Hàm kiểm thử chống rò rỉ nhãn (Leakage Test): verify_no_label_leakage(...)
"""

import os
import json
from typing import Tuple, Dict, Optional, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# ==============================================================================
# 1. CẤU HÌNH MẶC ĐỊNH PHASE 3 (DATA FREEZE V1)
# ==============================================================================
SEED = 42
NUM_FOLDS = 5
SMOOTHING_M = 30.0           # Hệ số làm mượt m-estimate cho Target Encoding (20..50)
MIN_COUNT_AIRPORT = 100      # Ngưỡng gộp nhóm hiếm OTHER cho sân bay (từ đường cong độ phủ EDA)
MIN_COUNT_ROUTE = 50         # Ngưỡng gộp nhóm hiếm OTHER cho tuyến bay (route)


# ==============================================================================
# 2. CÁC HÀM BIẾN ĐỔI ĐẶC TRƯNG THÀNH PHẦN
# ==============================================================================
FORBIDDEN_PREDICTORS = {"Delay", "row_id", "fold"}


def ensure_schema_and_fold(df: DataFrame, is_train: bool = False, seed: int = SEED) -> DataFrame:
    """
    Đảm bảo kiểu dữ liệu chuẩn khớp bản bàn giao tv1_v1 từ Phase 2:
      - Delay -> int
      - Length_invalid -> int (giữ nguyên cờ 4 dòng Length <= 0 từ TV1; nếu chưa có sẽ tự tạo)
      - DayOfWeek, Flight, Airline, AirportFrom, AirportTo -> string
      - Time, Length -> double
      - row_id -> long
      - Nếu là tập train và chưa có cột 'fold', tự động gán fold theo SHA256('fold:42:' + row_id) % 5.
    """
    out = (
        df
        .withColumn("Airline", F.col("Airline").cast(StringType()))
        .withColumn("Flight", F.col("Flight").cast("int").cast(StringType()))
        .withColumn("AirportFrom", F.col("AirportFrom").cast(StringType()))
        .withColumn("AirportTo", F.col("AirportTo").cast(StringType()))
        .withColumn("DayOfWeek", F.col("DayOfWeek").cast("int").cast(StringType()))
        .withColumn("Time", F.col("Time").cast(DoubleType()))
        .withColumn("Length", F.col("Length").cast(DoubleType()))
        .withColumn("Delay", F.col("Delay").cast(IntegerType()))
    )
    if "Length_invalid" in out.columns:
        out = out.withColumn("Length_invalid", F.col("Length_invalid").cast(IntegerType()))
    else:
        out = out.withColumn(
            "Length_invalid",
            F.when(F.col("Length").isNull() | (F.col("Length") <= 0), F.lit(1)).otherwise(F.lit(0)).cast(IntegerType())
        )
    if "row_id" not in out.columns:
        out = out.withColumn("row_id", F.monotonically_increasing_id())
    else:
        out = out.withColumn("row_id", F.col("row_id").cast("long"))
    if is_train and "fold" not in out.columns:
        fold_hash = F.sha2(F.concat(F.lit(f"fold:{seed}:"), F.col("row_id").cast("string")), 256)
        out = out.withColumn(
            "fold",
            F.pmod(F.conv(F.substring(fold_hash, 1, 8), 16, 10).cast("long"), F.lit(NUM_FOLDS)).cast(IntegerType())
        )
    return out


def add_basic_features(df: DataFrame) -> DataFrame:
    """
    Nhiệm vụ 1: Tạo các đặc trưng cơ bản:
      - hour (double): floor(Time / 60) -> dùng cho GBT
      - hour_cat (string): chuỗi "0".."23" -> dùng cho One-Hot Encoding trong LR
      - time_bucket (string): morning (5-11), afternoon (12-16), evening (17-20), night (21-4)
      - is_weekend (double): 1.0 nếu DayOfWeek thuộc {"6", "7"}, ngược lại 0.0
      - len_bin (string): short (<=90), medium (91-180), long (181-300), ultra_long (>300)
      - route (string): AirportFrom + "_" + AirportTo
      - airline_flight (string): Airline + "_" + Flight
    """
    hour_col = F.floor(F.col("Time") / 60.0)
    return (
        df
        .withColumn("hour", hour_col.cast(DoubleType()))
        .withColumn("hour_cat", hour_col.cast(IntegerType()).cast(StringType()))
        .withColumn(
            "time_bucket",
            F.when((hour_col >= 5) & (hour_col <= 11), F.lit("morning"))
             .when((hour_col >= 12) & (hour_col <= 16), F.lit("afternoon"))
             .when((hour_col >= 17) & (hour_col <= 20), F.lit("evening"))
             .otherwise(F.lit("night"))
        )
        .withColumn(
            "is_weekend",
            F.when(F.col("DayOfWeek").isin("6", "7"), F.lit(1.0)).otherwise(F.lit(0.0)).cast(DoubleType())
        )
        .withColumn(
            "len_bin",
            F.when(F.col("Length") <= 90, F.lit("short"))
             .when(F.col("Length") <= 180, F.lit("medium"))
             .when(F.col("Length") <= 300, F.lit("long"))
             .otherwise(F.lit("ultra_long"))
        )
        .withColumn("route", F.concat_ws("_", F.col("AirportFrom"), F.col("AirportTo")))
        .withColumn("airline_flight", F.concat_ws("_", F.col("Airline"), F.col("Flight")))
    )


def compute_flight_density(
    train_df: DataFrame,
    val_df: DataFrame,
    test_df: DataFrame,
    fit_on_train_only: bool = True
) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Nhiệm vụ 2: Mật độ chuyến bay (Flight Density):
      - density_from: số chuyến bay theo (AirportFrom, DayOfWeek, hour)
      - density_to  : số chuyến bay theo (AirportTo, DayOfWeek, hour)
    Mặc định fit_on_train_only=True để tuân thủ nghiêm ngặt quy tắc của TV1 (02_Read_Shared_Data.ipynb):
    "Fit các bước học từ dữ liệu chỉ trên Train".
    Cắt đứt lineage bằng localCheckpoint + broadcast join để tránh phình LogicalPlan.
    """
    cols_needed = ["AirportFrom", "AirportTo", "DayOfWeek", "hour"]
    if fit_on_train_only:
        schedule_source = train_df.select(cols_needed)
    else:
        schedule_source = (
            train_df.select(cols_needed)
            .unionByName(val_df.select(cols_needed))
            .unionByName(test_df.select(cols_needed))
        )

    dens_from_map = (
        schedule_source
        .groupBy("AirportFrom", "DayOfWeek", "hour")
        .agg(F.count("*").cast(DoubleType()).alias("density_from"))
        .localCheckpoint(eager=True)
    )
    dens_to_map = (
        schedule_source
        .groupBy("AirportTo", "DayOfWeek", "hour")
        .agg(F.count("*").cast(DoubleType()).alias("density_to"))
        .localCheckpoint(eager=True)
    )

    def attach_density(d: DataFrame) -> DataFrame:
        return (
            d
            .join(F.broadcast(dens_from_map), on=["AirportFrom", "DayOfWeek", "hour"], how="left")
            .join(F.broadcast(dens_to_map), on=["AirportTo", "DayOfWeek", "hour"], how="left")
            .fillna({"density_from": 0.0, "density_to": 0.0})
        )

    return attach_density(train_df), attach_density(val_df), attach_density(test_df), dens_from_map, dens_to_map


def fit_and_apply_rare_grouping(
    train_df: DataFrame,
    val_df: DataFrame,
    test_df: DataFrame,
    min_count_airport: int = MIN_COUNT_AIRPORT,
    min_count_route: int = MIN_COUNT_ROUTE
) -> Tuple[DataFrame, DataFrame, DataFrame, Dict[str, List[str]]]:
    """
    Nhiệm vụ 3: Gộp nhóm hiếm (Rare Category Grouping):
      - Các category xuất hiện ít hơn N dòng trong tập TRAIN được gộp thành "OTHER".
      - Tạo các cột: AirportFrom_g, AirportTo_g, route_g.
      - Dùng broadcast join thay cho .isin(2431 strings) để cây Catalyst luôn gọn nhẹ.
    """
    valid_from_df = (
        train_df.groupBy("AirportFrom")
        .count()
        .filter(F.col("count") >= min_count_airport)
        .select("AirportFrom", F.col("AirportFrom").alias("AirportFrom_g"))
        .localCheckpoint(eager=True)
    )
    valid_to_df = (
        train_df.groupBy("AirportTo")
        .count()
        .filter(F.col("count") >= min_count_airport)
        .select("AirportTo", F.col("AirportTo").alias("AirportTo_g"))
        .localCheckpoint(eager=True)
    )
    valid_route_df = (
        train_df.groupBy("route")
        .count()
        .filter(F.col("count") >= min_count_route)
        .select("route", F.col("route").alias("route_g"))
        .localCheckpoint(eager=True)
    )

    valid_from = [r[0] for r in valid_from_df.select("AirportFrom").collect() if r[0] is not None]
    valid_to = [r[0] for r in valid_to_df.select("AirportTo").collect() if r[0] is not None]
    valid_route = [r[0] for r in valid_route_df.select("route").collect() if r[0] is not None]

    def apply_grouping(d: DataFrame) -> DataFrame:
        return (
            d
            .join(F.broadcast(valid_from_df), on="AirportFrom", how="left")
            .join(F.broadcast(valid_to_df), on="AirportTo", how="left")
            .join(F.broadcast(valid_route_df), on="route", how="left")
            .fillna({"AirportFrom_g": "OTHER", "AirportTo_g": "OTHER", "route_g": "OTHER"})
        )

    rare_maps = {
        "min_count_airport": min_count_airport,
        "min_count_route": min_count_route,
        "valid_AirportFrom": valid_from,
        "valid_AirportTo": valid_to,
        "valid_route": valid_route
    }
    return apply_grouping(train_df), apply_grouping(val_df), apply_grouping(test_df), rare_maps


def fit_and_apply_frequency_encoding(
    train_df: DataFrame,
    val_df: DataFrame,
    test_df: DataFrame
) -> Tuple[DataFrame, DataFrame, DataFrame, Dict[str, DataFrame]]:
    """
    Nhiệm vụ 4: Frequency Encoding:
      - Tính log1p(count trong train) = ln(1 + count_train) cho:
        + AirportFrom    -> log_freq_from
        + AirportTo      -> log_freq_to
        + route          -> log_freq_route
        + airline_flight -> log_freq_flight
      - Category chưa từng thấy (unseen) trong val/test nhận giá trị 0.0.
    """
    specs = [
        ("AirportFrom", "log_freq_from"),
        ("AirportTo", "log_freq_to"),
        ("route", "log_freq_route"),
        ("airline_flight", "log_freq_flight"),
    ]
    freq_tables: Dict[str, DataFrame] = {}
    source_train = train_df

    for col_name, out_col in specs:
        freq_df = (
            source_train.groupBy(col_name)
            .agg(F.log1p(F.count("*").cast(DoubleType())).alias(out_col))
            .localCheckpoint(eager=True)
        )
        freq_tables[col_name] = freq_df
        train_df = train_df.join(F.broadcast(freq_df), on=col_name, how="left").fillna({out_col: 0.0})
        val_df = val_df.join(F.broadcast(freq_df), on=col_name, how="left").fillna({out_col: 0.0})
        test_df = test_df.join(F.broadcast(freq_df), on=col_name, how="left").fillna({out_col: 0.0})

    return train_df, val_df, test_df, freq_tables


def fit_and_apply_oof_target_encoding(
    train_df: DataFrame,
    val_df: DataFrame,
    test_df: DataFrame,
    m: float = SMOOTHING_M
) -> Tuple[DataFrame, DataFrame, DataFrame, float, Dict[str, DataFrame]]:
    """
    Nhiệm vụ 5: Target Encoding Out-of-Fold (OOF) với m-estimate smoothing (kể cả OOF prior):
      - Trên tập TRAIN: cả thống kê nhóm lẫn prior toàn cục (oof_prior) của fold f
        đều được tính CHỈ từ 4 folds còn lại.
      - Trên tập VAL và TEST: dùng tổng và prior của toàn bộ tập TRAIN.
      - Sử dụng localCheckpoint(eager=True) + broadcast join cho từng bảng map để cây Catalyst
        giữ độ sâu tuyến tính O(1), không bao giờ bị tràn Heap Java.
    """
    source_train = train_df
    global_agg = source_train.agg(
        F.sum(F.col("Delay").cast(DoubleType())).alias("total_S"),
        F.count("*").cast(DoubleType()).alias("total_N")
    ).first()
    total_S = float(global_agg["total_S"])
    total_N = float(global_agg["total_N"])
    prior = total_S / total_N

    # Tính oof_prior riêng cho từng fold từ 4 folds còn lại
    fold_totals = (
        source_train.groupBy("fold")
        .agg(
            F.sum(F.col("Delay").cast(DoubleType())).alias("fold_S"),
            F.count("*").cast(DoubleType()).alias("fold_N")
        )
    )
    oof_priors = (
        fold_totals
        .withColumn(
            "oof_prior",
            ((F.lit(total_S) - F.col("fold_S")) / (F.lit(total_N) - F.col("fold_N"))).cast(DoubleType())
        )
        .select("fold", "oof_prior")
        .localCheckpoint(eager=True)
    )

    te_specs = [
        ("route", "te_route"),
        ("airline_flight", "te_flight"),
        ("AirportFrom_g", "te_from"),
        ("AirportTo_g", "te_to"),
    ]
    full_te_maps: Dict[str, DataFrame] = {}

    for col_name, te_col in te_specs:
        # Thống kê theo (category, fold) trên train gốc (không dùng train_df đang bị join dần)
        g = (
            source_train.groupBy(col_name, "fold")
            .agg(
                F.sum(F.col("Delay").cast(DoubleType())).alias("sum_fold"),
                F.count("*").cast(DoubleType()).alias("cnt_fold")
            )
        )
        tot = (
            g.groupBy(col_name)
            .agg(
                F.sum("sum_fold").alias("sum_total"),
                F.sum("cnt_fold").alias("cnt_total")
            )
        )
        oof = (
            g.join(tot, on=col_name, how="inner")
            .join(F.broadcast(oof_priors), on="fold", how="inner")
            .withColumn(
                te_col,
                (
                    (F.col("sum_total") - F.col("sum_fold") + F.lit(m) * F.col("oof_prior"))
                    / (F.col("cnt_total") - F.col("cnt_fold") + F.lit(m))
                ).cast(DoubleType())
            )
            .select(col_name, "fold", te_col)
            .localCheckpoint(eager=True)
        )
        train_df = (
            train_df
            .join(F.broadcast(oof), on=[col_name, "fold"], how="left")
            .join(F.broadcast(oof_priors), on="fold", how="left")
            .withColumn(te_col, F.coalesce(F.col(te_col), F.col("oof_prior")))
            .drop("oof_prior")
        )

        full_map = (
            tot.withColumn(
                te_col,
                ((F.col("sum_total") + F.lit(m * prior)) / (F.col("cnt_total") + F.lit(m))).cast(DoubleType())
            )
            .select(col_name, te_col)
            .localCheckpoint(eager=True)
        )
        full_te_maps[col_name] = full_map
        val_df = val_df.join(F.broadcast(full_map), on=col_name, how="left").fillna({te_col: prior})
        test_df = test_df.join(F.broadcast(full_map), on=col_name, how="left").fillna({te_col: prior})

    return train_df, val_df, test_df, prior, full_te_maps


# ==============================================================================
# 3. HÀM TỔNG HỢP XÂY DỰNG BẢNG ĐẶC TRƯNG V1 & LƯU BẢNG TRA CỨU
# ==============================================================================
FEATURE_TABLE_COLUMNS = [
    # Định danh & Nhãn (không đưa vào features)
    "row_id", "Delay",
    # Biến gốc đã chuẩn hóa (bao gồm cờ Length_invalid từ TV1)
    "Airline", "Flight", "AirportFrom", "AirportTo", "DayOfWeek", "Time", "Length", "Length_invalid",
    # Đặc trưng cơ bản
    "hour", "hour_cat", "time_bucket", "is_weekend", "len_bin", "route", "airline_flight",
    # Mật độ chuyến bay (tính từ tập Train)
    "density_from", "density_to",
    # Nhóm gộp hiếm (OTHER)
    "AirportFrom_g", "AirportTo_g", "route_g",
    # Frequency Encoding
    "log_freq_from", "log_freq_to", "log_freq_route", "log_freq_flight",
    # Out-of-Fold Target Encoding
    "te_route", "te_flight", "te_from", "te_to"
]


def build_features_v1(
    train_raw: DataFrame,
    val_raw: DataFrame,
    test_raw: DataFrame,
    smoothing_m: float = SMOOTHING_M,
    min_count_airport: int = MIN_COUNT_AIRPORT,
    min_count_route: int = MIN_COUNT_ROUTE,
    fit_density_on_train_only: bool = True
) -> Tuple[DataFrame, DataFrame, DataFrame, dict]:
    """
    Thực thi toàn bộ quy trình Kỹ thuật Đặc trưng Phase 3 trên 3 tập train/val/test.
    Trả về: (train_fe, val_fe, test_fe, artifacts_dict)
    """
    train_df = ensure_schema_and_fold(train_raw, is_train=True)
    val_df = ensure_schema_and_fold(val_raw, is_train=False)
    test_df = ensure_schema_and_fold(test_raw, is_train=False)

    # Bước 1: Đặc trưng cơ bản
    train_df = add_basic_features(train_df)
    val_df = add_basic_features(val_df)
    test_df = add_basic_features(test_df)

    # Bước 2: Gộp nhóm hiếm thành OTHER (chỉ tính ngưỡng trên train) và cắt lineage để tái sử dụng nhanh
    train_df, val_df, test_df, rare_maps = fit_and_apply_rare_grouping(
        train_df, val_df, test_df,
        min_count_airport=min_count_airport,
        min_count_route=min_count_route
    )
    train_df = train_df.localCheckpoint(eager=True)

    # Bước 3: Mật độ chuyến bay (mặc định fit trên tập train)
    train_df, val_df, test_df, dens_from_map, dens_to_map = compute_flight_density(
        train_df, val_df, test_df, fit_on_train_only=fit_density_on_train_only
    )

    # Bước 4: Frequency Encoding (log1p count trên train)
    train_df, val_df, test_df, freq_tables = fit_and_apply_frequency_encoding(train_df, val_df, test_df)

    # Bước 5: Out-of-Fold Target Encoding với smoothing m
    train_df, val_df, test_df, prior, full_te_maps = fit_and_apply_oof_target_encoding(
        train_df, val_df, test_df, m=smoothing_m
    )

    # Sắp xếp cột và đóng gói bảng đã tính toán xong bằng localCheckpoint để cắt toàn bộ LogicalPlan cũ
    train_cols = FEATURE_TABLE_COLUMNS + ["fold"]
    train_out = train_df.select(*train_cols).localCheckpoint(eager=True)
    val_out = val_df.select(*FEATURE_TABLE_COLUMNS).localCheckpoint(eager=True)
    test_out = test_df.select(*FEATURE_TABLE_COLUMNS).localCheckpoint(eager=True)

    artifacts = {
        "prior": prior,
        "smoothing_m": smoothing_m,
        "rare_maps": rare_maps,
        "freq_tables": freq_tables,
        "full_te_maps": full_te_maps,
        "dens_from_map": dens_from_map,
        "dens_to_map": dens_to_map,
    }
    return train_out, val_out, test_out, artifacts


def save_features_and_lookups(
    train_fe: DataFrame,
    val_fe: DataFrame,
    test_fe: DataFrame,
    artifacts: dict,
    output_base_dir: str = "data/features/v1"
) -> None:
    """
    Ghi 3 bảng đặc trưng ra Parquet (KHÔNG ghi cột đã qua StringIndexer để giữ nguyên tính tổng quát)
    và lưu toàn bộ bảng tra cứu (lookup_maps) phục vụ Demo ở Phase 6.
    """
    os.makedirs(output_base_dir, exist_ok=True)
    for split_name, df_part in [("train", train_fe), ("val", val_fe), ("test", test_fe)]:
        out_path = os.path.join(output_base_dir, split_name)
        df_part.write.mode("overwrite").parquet(out_path)
        print(f"[SAVE] Đã ghi tập {split_name} ra: {out_path}")

    lookup_dir = os.path.join(output_base_dir, "lookup_maps")
    os.makedirs(lookup_dir, exist_ok=True)

    # Lưu metadata & danh sách nhóm hợp lệ (JSON)
    meta_payload = {
        "prior": artifacts["prior"],
        "smoothing_m": artifacts["smoothing_m"],
        "min_count_airport": artifacts["rare_maps"]["min_count_airport"],
        "min_count_route": artifacts["rare_maps"]["min_count_route"],
        "valid_AirportFrom": artifacts["rare_maps"]["valid_AirportFrom"],
        "valid_AirportTo": artifacts["rare_maps"]["valid_AirportTo"],
        "valid_route": artifacts["rare_maps"]["valid_route"],
    }
    with open(os.path.join(lookup_dir, "encoding_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2, ensure_ascii=False)

    # Lưu các bảng tra cứu Target Encoding, Frequency Encoding, Flight Density
    for col_name, df_map in artifacts["full_te_maps"].items():
        df_map.write.mode("overwrite").parquet(os.path.join(lookup_dir, f"te_{col_name}.parquet"))

    for col_name, df_map in artifacts["freq_tables"].items():
        df_map.write.mode("overwrite").parquet(os.path.join(lookup_dir, f"freq_{col_name}.parquet"))

    artifacts["dens_from_map"].write.mode("overwrite").parquet(os.path.join(lookup_dir, "density_from.parquet"))
    artifacts["dens_to_map"].write.mode("overwrite").parquet(os.path.join(lookup_dir, "density_to.parquet"))
    print(f"[SAVE] Đã lưu đầy đủ bảng tra cứu (Lookup Maps) tại: {lookup_dir}")


# ==============================================================================
# 4. HAI HÀM TẠO PIPELINE CHUẨN CHO TV3 (LR) VÀ TV4 (GBT)
# ==============================================================================
def make_pipeline_lr(
    reg_param: float = 0.0,
    elastic_net_param: float = 0.0,
    max_iter: int = 100,
    include_te: bool = False,
    include_length_invalid: bool = True,
    include_estimator: bool = True
) -> Pipeline:
    """
    Nhiệm vụ 6A: Tạo Pipeline chuẩn cho TV3 (Logistic Regression).
    - Biến phân loại đưa vào OneHotEncoder: Airline, DayOfWeek, hour_cat (giờ ở dạng chuỗi).
    - Biến số: Length, log_freq_from, log_freq_to, log_freq_route, density_from, density_to
      (+ Length_invalid nếu include_length_invalid=True).
    - Tuyệt đối không đưa Delay, row_id, fold vào danh sách cột đầu vào của VectorAssembler.
    """
    cat_cols = ["Airline", "DayOfWeek", "hour_cat"]
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_i", handleInvalid="keep")
        for c in cat_cols
    ]
    ohe = OneHotEncoder(
        inputCols=[f"{c}_i" for c in cat_cols],
        outputCols=[f"{c}_oh" for c in cat_cols]
    )

    num_cols = [
        "Length",
        "log_freq_from",
        "log_freq_to",
        "log_freq_route",
        "density_from",
        "density_to"
    ]
    if include_length_invalid:
        num_cols = num_cols + ["Length_invalid"]
    if include_te:
        num_cols = num_cols + ["is_weekend", "te_route", "te_flight", "te_from", "te_to"]

    leaked = FORBIDDEN_PREDICTORS.intersection(cat_cols + num_cols)
    assert not leaked, f"Cấm đưa cột {leaked} vào predictors của LR!"

    assembler = VectorAssembler(
        inputCols=[f"{c}_oh" for c in cat_cols] + num_cols,
        outputCol="features"
    )

    stages = indexers + [ohe, assembler]
    if include_estimator:
        lr = LogisticRegression(
            labelCol="Delay",
            featuresCol="features",
            maxIter=max_iter,
            regParam=reg_param,
            elasticNetParam=elastic_net_param,
            standardization=True
        )
        stages.append(lr)

    return Pipeline(stages=stages)


def make_pipeline_gbt(
    max_bins: int = 300,
    max_iter: int = 100,
    max_depth: int = 5,
    step_size: float = 0.1,
    subsampling_rate: float = 1.0,
    seed: int = SEED,
    include_length_invalid: bool = True,
    include_estimator: bool = True
) -> Pipeline:
    """
    Nhiệm vụ 6B: Tạo Pipeline chuẩn cho TV4 (GBTClassifier).
    - Biến phân loại: Airline, DayOfWeek, AirportFrom_g, AirportTo_g.
      StringIndexer nằm trực tiếp trong cùng Pipeline với VectorAssembler & GBTClassifier
      để giữ nguyên metadata 'nominal' cho các nút cây quyết định.
    - max_bins mặc định = 300 (>= số lượng category tối đa của AirportFrom_g/AirportTo_g).
    - Tuyệt đối không đưa Delay, row_id, fold vào danh sách cột đầu vào của VectorAssembler.
    """
    cat_cols = ["Airline", "DayOfWeek", "AirportFrom_g", "AirportTo_g"]
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_i", handleInvalid="keep")
        for c in cat_cols
    ]

    num_cols = [
        "hour",
        "Length",
        "te_route",
        "te_flight",
        "log_freq_route",
        "density_from",
        "density_to"
    ]
    if include_length_invalid:
        num_cols = num_cols + ["Length_invalid"]

    leaked = FORBIDDEN_PREDICTORS.intersection(cat_cols + num_cols)
    assert not leaked, f"Cấm đưa cột {leaked} vào predictors của GBT!"

    assembler = VectorAssembler(
        inputCols=[f"{c}_i" for c in cat_cols] + num_cols,
        outputCol="features"
    )

    stages = indexers + [assembler]
    if include_estimator:
        gbt = GBTClassifier(
            labelCol="Delay",
            featuresCol="features",
            maxIter=max_iter,
            maxDepth=max_depth,
            stepSize=step_size,
            subsamplingRate=subsampling_rate,
            maxBins=max_bins,
            seed=seed
        )
        stages.append(gbt)

    return Pipeline(stages=stages)


# ==============================================================================
# 5. KIỂM THỬ CHỐNG RÒ RỈ NHÃN (NHIỆM VỤ 8 - BẮT BUỘC)
# ==============================================================================
def verify_no_label_leakage(
    train_raw: DataFrame,
    train_fe: DataFrame,
    val_fe: DataFrame,
    prior: float,
    m: float = SMOOTHING_M
) -> dict:
    """
    Nhiệm vụ 8: Thực hiện 3 bài kiểm thử bắt buộc để chứng minh KHÔNG rò rỉ nhãn (Label Leakage):
      1. Đối chiếu thủ công giá trị te_route trên tập VAL với công thức (S + m*prior)/(N + m) tính trực tiếp từ train_raw.
      2. Đối chiếu thủ công giá trị OOF te_route trên tập TRAIN với công thức dùng 4 folds còn lại (gồm cả OOF prior).
      3. Huấn luyện mô hình chỉ gồm duy nhất đặc trưng `te_route` rồi so sánh ROC-AUC trên Train và ROC-AUC trên Val.
    """
    print("\n=== BẮT ĐẦU KIỂM THỬ CHỐNG RÒ RỈ NHÃN (LABEL LEAKAGE CHECKS) ===")

    # --- Kiểm thử 1: Đối chiếu thủ công 5 tuyến bay trong tập VAL ---
    sample_routes = [
        r["route"] for r in val_fe.select("route", "te_route").distinct().limit(5).collect()
    ]
    train_with_route = ensure_schema_and_fold(train_raw, is_train=True).withColumn(
        "route", F.concat_ws("_", F.col("AirportFrom"), F.col("AirportTo"))
    )
    manual_checks = []
    max_abs_diff = 0.0

    for rt in sample_routes:
        stat = (
            train_with_route.filter(F.col("route") == rt)
            .agg(
                F.sum(F.col("Delay").cast(DoubleType())).alias("S"),
                F.count("*").cast(DoubleType()).alias("N")
            )
            .first()
        )
        S_val = float(stat["S"]) if stat["S"] is not None else 0.0
        N_val = float(stat["N"]) if stat["N"] is not None else 0.0
        expected_te = (S_val + m * prior) / (N_val + m) if N_val > 0 else prior

        actual_te = float(
            val_fe.filter(F.col("route") == rt).select("te_route").first()["te_route"]
        )
        diff = abs(expected_te - actual_te)
        max_abs_diff = max(max_abs_diff, diff)
        manual_checks.append({
            "route": rt,
            "train_S": S_val,
            "train_N": N_val,
            "manual_te": round(expected_te, 6),
            "pipeline_val_te": round(actual_te, 6),
            "abs_diff": diff
        })
        print(f"  [Check 1 - Val] Route={rt:<10} | Manual={expected_te:.6f} | Val_TE={actual_te:.6f} | Diff={diff:.2e}")

    assert max_abs_diff < 1e-6, f"PHÁT HIỆN SAI LỆCH TE_ROUTE TRÊN VAL: max_abs_diff = {max_abs_diff}"

    # --- Kiểm thử 2: Đối chiếu thủ công OOF te_route (gồm OOF prior từ 4 folds còn lại) trên tập TRAIN ---
    oof_samples = train_fe.select("route", "fold", "te_route").distinct().limit(3).collect()
    max_oof_diff = 0.0
    for row in oof_samples:
        rt, fld, actual_oof = row["route"], int(row["fold"]), float(row["te_route"])
        other_folds = train_with_route.filter(F.col("fold") != fld)
        other_global = other_folds.agg(
            F.avg(F.col("Delay").cast(DoubleType())).alias("oof_prior")
        ).first()
        oof_prior = float(other_global["oof_prior"])
        other_rt = other_folds.filter(F.col("route") == rt).agg(
            F.sum(F.col("Delay").cast(DoubleType())).alias("S_oof"),
            F.count("*").cast(DoubleType()).alias("N_oof")
        ).first()
        s_oof = float(other_rt["S_oof"]) if other_rt["S_oof"] is not None else 0.0
        n_oof = float(other_rt["N_oof"]) if other_rt["N_oof"] is not None else 0.0
        expected_oof = (s_oof + m * oof_prior) / (n_oof + m)
        diff_oof = abs(expected_oof - actual_oof)
        max_oof_diff = max(max_oof_diff, diff_oof)
        print(f"  [Check 2 - OOF] Route={rt:<10} fold={fld} | Manual_OOF={expected_oof:.6f} | Train_TE={actual_oof:.6f} | Diff={diff_oof:.2e}")

    assert max_oof_diff < 1e-6, f"PHÁT HIỆN SAI LỆCH OOF TE_ROUTE TRÊN TRAIN: max_oof_diff = {max_oof_diff}"

    # --- Kiểm thử 3: Kiểm tra mô hình đơn biến chỉ gồm `te_route` (So sánh AUC Train vs Val) ---
    asm_single = VectorAssembler(inputCols=["te_route"], outputCol="features_single")
    lr_single = LogisticRegression(
        labelCol="Delay", featuresCol="features_single", maxIter=20
    )
    single_model = Pipeline(stages=[asm_single, lr_single]).fit(train_fe)

    evaluator = BinaryClassificationEvaluator(
        labelCol="Delay", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    auc_train_single = float(evaluator.evaluate(single_model.transform(train_fe)))
    auc_val_single = float(evaluator.evaluate(single_model.transform(val_fe)))
    auc_gap = auc_train_single - auc_val_single

    print(f"  [Check 3 - AUC] Single-Feature (te_route) ROC-AUC -> Train: {auc_train_single:.4f} | Val: {auc_val_single:.4f} | Gap: {auc_gap:+.4f}")

    leakage_passed = (max_abs_diff < 1e-6) and (max_oof_diff < 1e-6) and (abs(auc_gap) < 0.03)
    status_str = "PASSED (Không phát hiện rò rỉ nhãn)" if leakage_passed else "WARNING (Cần kiểm tra lại)"
    print(f"=== KẾT LUẬN KIỂM THỬ RÒ RỈ NHÃN: {status_str} ===\n")

    return {
        "status": status_str,
        "max_manual_abs_diff": max_abs_diff,
        "max_oof_abs_diff": max_oof_diff,
        "single_feature_te_route_auc_train": round(auc_train_single, 4),
        "single_feature_te_route_auc_val": round(auc_val_single, 4),
        "auc_gap_train_minus_val": round(auc_gap, 4),
        "manual_route_checks": manual_checks
    }
