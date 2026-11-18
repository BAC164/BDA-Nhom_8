import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql import functions as F
except ImportError:
    SparkSession = None
    DataFrame = object
    F = None

def wilson_ci_95(p_hat: np.ndarray, n: np.ndarray, z: float = 1.96):
    """
    Tính khoảng tin cậy Wilson 95% (cận dưới, cận trên) cho tỷ lệ nhị phân p_hat với cỡ mẫu n.
    """
    denom = 1.0 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2.0 * n)) / denom
    half_width = (z * np.sqrt((p_hat * (1.0 - p_hat) / n) + ((z ** 2) / (4.0 * (n ** 2))))) / denom
    return np.clip(center - half_width, 0.0, 1.0), np.clip(center + half_width, 0.0, 1.0)


def prepare_eda_dataframe(df_raw: DataFrame) -> DataFrame:
    """Chuẩn bị các cột phục vụ phân tích EDA Phase 1."""
    return (
        df_raw
        .withColumn("Airline", F.col("Airline").cast("string"))
        .withColumn("Flight_str", F.col("Flight").cast("int").cast("string"))
        .withColumn("AirportFrom", F.col("AirportFrom").cast("string"))
        .withColumn("AirportTo", F.col("AirportTo").cast("string"))
        .withColumn("DayOfWeek", F.col("DayOfWeek").cast("int"))
        .withColumn("Time", F.col("Time").cast("double"))
        .withColumn("Length", F.col("Length").cast("double"))
        .withColumn("Delay", F.col("Delay").cast("double"))
        .withColumn("hour", F.floor(F.col("Time") / 60.0).cast("int"))
        .withColumn("route", F.concat_ws("_", F.col("AirportFrom"), F.col("AirportTo")))
        .withColumn("airline_flight", F.concat_ws("_", F.col("Airline"), F.col("Flight_str")))
    )


def _save_heatmap_hour_x_dayofweek(agg_pdf: pd.DataFrame, out_dir: str) -> None:
    """Lưu CSV và vẽ Heatmap Giờ × Ngày trong tuần."""
    agg_pdf.to_csv(os.path.join(out_dir, "tv2_hour_x_dayofweek.csv"), index=False)
    pivot_rate = agg_pdf.pivot(index="hour", columns="DayOfWeek", values="delay_rate_pct")
    dow_map = {1: "T2 (1)", 2: "T3 (2)", 3: "T4 (3)", 4: "T5 (4)", 5: "T6 (5)", 6: "T7 (6)", 7: "CN (7)"}
    pivot_rate = pivot_rate.rename(columns={k: v for k, v in dow_map.items() if k in pivot_rate.columns})

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_rate,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "Tỷ lệ trễ (%)"}
    )
    plt.title("Heatmap Tỷ lệ Chuyến bay bị trễ (%) theo Giờ khởi hành × Ngày trong tuần", fontsize=13, fontweight="bold")
    plt.xlabel("Ngày trong tuần (1 = Thứ Hai ... 7 = Chủ Nhật)")
    plt.ylabel("Giờ khởi hành dự kiến trong ngày (hour = Time // 60)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tv2_heatmap_hour_x_dayofweek.png"), dpi=180)
    plt.close()


def _save_heatmap_airline_x_hour(agg_pdf: pd.DataFrame, out_dir: str, min_cell_flights: int = 30) -> None:
    """Lưu CSV và vẽ Heatmap Hãng bay × Giờ khởi hành (05h - 23h)."""
    agg_pdf.to_csv(os.path.join(out_dir, "tv2_airline_x_hour.csv"), index=False)
    agg_filtered = agg_pdf.copy()
    agg_filtered.loc[agg_filtered["n_flights"] < min_cell_flights, "delay_rate_pct"] = np.nan

    weighted = (
        agg_pdf.assign(_w=agg_pdf["delay_rate_pct"] * agg_pdf["n_flights"])
        .groupby("Airline")[["_w", "n_flights"]]
        .sum()
    )
    airline_order = (weighted["_w"] / weighted["n_flights"]).sort_values(ascending=False).index.tolist()
    pivot_air = agg_filtered.pivot(index="Airline", columns="hour", values="delay_rate_pct").loc[airline_order]

    plt.figure(figsize=(14, 7))
    sns.heatmap(
        pivot_air,
        annot=True,
        fmt=".0f",
        cmap="RdYlBu_r",
        cbar_kws={"label": "Tỷ lệ trễ (%)"},
        linewidths=0.4
    )
    plt.title("Heatmap Tỷ lệ Trễ (%) theo Hãng hàng không (Airline) × Giờ khởi hành (05h - 23h)", fontsize=13, fontweight="bold")
    plt.xlabel("Giờ khởi hành trong ngày (hour)")
    plt.ylabel("Hãng hàng không (Sắp xếp theo tỷ lệ trễ giảm dần)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tv2_heatmap_airline_x_hour.png"), dpi=180)
    plt.close()


def _save_longtail_and_wilson(
    freq_dfs: dict,
    total_rows: int,
    out_dir: str,
    train_scale: float = 0.70
) -> pd.DataFrame:
    """Vẽ đường cong độ phủ đuôi dài, xuất bảng đề xuất ngưỡng OTHER và vẽ Top 15 sân bay Wilson 95% CI."""
    threshold_rows = []
    plt.figure(figsize=(11, 6.5))

    for col_name, pdf in freq_dfs.items():
        pdf = pdf.sort_values("count", ascending=False).reset_index(drop=True)
        pdf["count_train_est"] = np.round(pdf["count"] * train_scale).astype(int)
        pdf["rank"] = np.arange(1, len(pdf) + 1)
        pdf["cum_flights"] = pdf["count"].cumsum()
        pdf["cum_coverage_pct"] = (pdf["cum_flights"] / total_rows) * 100.0
        freq_dfs[col_name] = pdf

        plt.plot(pdf["rank"], pdf["cum_coverage_pct"], linewidth=2.4, label=f"{col_name} ({len(pdf):,} giá trị phân biệt)")

        for min_n in [20, 50, 100, 200, 500, 1000]:
            kept = pdf[pdf["count_train_est"] >= min_n]
            cov_pct = (kept["count"].sum() / total_rows) * 100.0
            threshold_rows.append({
                "Feature": col_name,
                "Min_Train_Count_N": min_n,
                "Total_Categories": len(pdf),
                "Kept_Categories": len(kept),
                "Grouped_To_OTHER": len(pdf) - len(kept),
                "Row_Coverage_Pct": round(cov_pct, 2),
                "OTHER_Row_Share_Pct": round(100.0 - cov_pct, 2)
            })

    plt.xscale("log")
    plt.axhline(90, color="gray", linestyle="--", alpha=0.7, label="Ngưỡng phủ 90% số dòng")
    plt.axhline(95, color="green", linestyle="--", alpha=0.7, label="Ngưỡng phủ 95% số dòng")
    plt.axhline(99, color="red", linestyle="--", alpha=0.7, label="Ngưỡng phủ 99% số dòng")
    plt.title("Đường cong Độ phủ Lũy kế (Long-Tail Cumulative Coverage) theo Top-k Nhóm", fontsize=13, fontweight="bold")
    plt.xlabel("Số lượng nhóm giữ lại Top-k (Thang đo Log)")
    plt.ylabel("Độ phủ lũy kế trên 539,383 chuyến bay (%)")
    plt.legend(loc="lower right")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tv2_longtail_coverage_curves.png"), dpi=180)
    plt.close()

    thresh_df = pd.DataFrame(threshold_rows)
    thresh_df.to_csv(os.path.join(out_dir, "tv2_threshold_proposals.csv"), index=False)

    # Xếp hạng Sân bay >= 1,000 chuyến kèm Khoảng tin cậy Wilson 95%
    ap_pdf = freq_dfs["AirportFrom"]
    ap_major = ap_pdf[ap_pdf["count"] >= 1000].copy()
    ci_low, ci_high = wilson_ci_95(ap_major["delay_rate"].values, ap_major["count"].values)
    ap_major["delay_rate_pct"] = (ap_major["delay_rate"] * 100.0).round(2)
    ap_major["wilson_95_low_pct"] = (ci_low * 100.0).round(2)
    ap_major["wilson_95_high_pct"] = (ci_high * 100.0).round(2)
    ap_major = ap_major.sort_values("delay_rate_pct", ascending=False)
    ap_major.to_csv(os.path.join(out_dir, "tv2_major_airports_wilson_ci.csv"), index=False)

    top15_ci = ap_major.head(15).iloc[::-1]
    plt.figure(figsize=(10, 6.5))
    x_val = top15_ci["delay_rate_pct"].values
    err_low = x_val - top15_ci["wilson_95_low_pct"].values
    err_high = top15_ci["wilson_95_high_pct"].values - x_val
    plt.errorbar(
        x_val, top15_ci["AirportFrom"],
        xerr=[err_low, err_high],
        fmt="o", color="#b91c1c", ecolor="#1e293b", elinewidth=2, capsize=4
    )
    plt.title("Top 15 Sân bay (≥ 1,000 chuyến) có Tỷ lệ Trễ cao nhất kèm Khoảng tin cậy Wilson 95%", fontsize=12, fontweight="bold")
    plt.xlabel("Tỷ lệ trễ (%) và Khoảng tin cậy Wilson 95%")
    plt.ylabel("Mã Sân bay khởi hành (AirportFrom)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tv2_top_airports_wilson_ci.png"), dpi=180)
    plt.close()

    return thresh_df


def plot_heatmap_hour_x_dayofweek(df: DataFrame, out_dir: str) -> pd.DataFrame:
    agg_pdf = (
        df.groupBy("hour", "DayOfWeek")
        .agg(
            F.count("*").alias("n_flights"),
            (F.avg("Delay") * 100.0).alias("delay_rate_pct")
        )
        .orderBy("hour", "DayOfWeek")
        .toPandas()
    )
    _save_heatmap_hour_x_dayofweek(agg_pdf, out_dir)
    return agg_pdf


def plot_heatmap_airline_x_hour(df: DataFrame, out_dir: str, min_cell_flights: int = 30) -> pd.DataFrame:
    agg_pdf = (
        df.filter((F.col("hour") >= 5) & (F.col("hour") <= 23))
        .groupBy("Airline", "hour")
        .agg(
            F.count("*").alias("n_flights"),
            (F.avg("Delay") * 100.0).alias("delay_rate_pct")
        )
        .toPandas()
    )
    _save_heatmap_airline_x_hour(agg_pdf, out_dir, min_cell_flights=min_cell_flights)
    return agg_pdf


def analyze_long_tail_and_thresholds(df: DataFrame, out_dir: str) -> pd.DataFrame:
    total_rows = df.count()
    cols_to_check = ["AirportFrom", "AirportTo", "route", "airline_flight"]
    freq_dfs = {}

    for col_name in cols_to_check:
        freq_dfs[col_name] = (
            df.groupBy(col_name)
            .agg(
                F.count("*").alias("count"),
                F.avg("Delay").alias("delay_rate")
            )
            .orderBy(F.desc("count"))
            .toPandas()
        )

    return _save_longtail_and_wilson(freq_dfs, total_rows, out_dir, train_scale=0.70)


def run_tv2_phase1_eda_pandas(parquet_path: str = "data/dataset_1169.pq", out_dir: str = "reports/eda") -> pd.DataFrame:
  
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_parquet(parquet_path)
    df["Airline"] = df["Airline"].astype(str)
    df["Flight_str"] = df["Flight"].astype(int).astype(str)
    df["AirportFrom"] = df["AirportFrom"].astype(str)
    df["AirportTo"] = df["AirportTo"].astype(str)
    df["DayOfWeek"] = df["DayOfWeek"].astype(int)
    df["Time"] = df["Time"].astype(float)
    df["Length"] = df["Length"].astype(float)
    df["Delay"] = df["Delay"].astype(float)
    df["hour"] = np.floor(df["Time"] / 60.0).astype(int)
    df["route"] = df["AirportFrom"] + "_" + df["AirportTo"]
    df["airline_flight"] = df["Airline"] + "_" + df["Flight_str"]

    h_dow = (
        df.groupby(["hour", "DayOfWeek"])
        .agg(n_flights=("Delay", "count"), delay_rate_pct=("Delay", lambda s: s.mean() * 100.0))
        .reset_index()
    )
    _save_heatmap_hour_x_dayofweek(h_dow, out_dir)

    df_day = df[(df["hour"] >= 5) & (df["hour"] <= 23)]
    air_h = (
        df_day.groupby(["Airline", "hour"])
        .agg(n_flights=("Delay", "count"), delay_rate_pct=("Delay", lambda s: s.mean() * 100.0))
        .reset_index()
    )
    _save_heatmap_airline_x_hour(air_h, out_dir, min_cell_flights=30)

    freq_dfs = {
        col_name: (
            df.groupby(col_name)
            .agg(count=("Delay", "count"), delay_rate=("Delay", "mean"))
            .reset_index()
        )
        for col_name in ["AirportFrom", "AirportTo", "route", "airline_flight"]
    }
    return _save_longtail_and_wilson(freq_dfs, len(df), out_dir, train_scale=0.70)


def run_tv2_phase1_eda(parquet_path: str = "data/dataset_1169.pq", out_dir: str = "reports/eda"):
    os.makedirs(out_dir, exist_ok=True)
    try:
        spark = (
            SparkSession.builder
            .appName("BDA20_Phase1_EDA_TV2")
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        raw_df = spark.read.parquet(parquet_path)
        eda_df = prepare_eda_dataframe(raw_df).cache()
        plot_heatmap_hour_x_dayofweek(eda_df, out_dir)
        plot_heatmap_airline_x_hour(eda_df, out_dir)
        thresh_df = analyze_long_tail_and_thresholds(eda_df, out_dir)
        spark.stop()
    except Exception:
        print("[INFO] Chạy tổng hợp EDA trực tiếp trên 539,383 dòng bằng Pandas/PyArrow...")
        thresh_df = run_tv2_phase1_eda_pandas(parquet_path, out_dir)

    print("\n=== BẢNG ĐỘ PHỦ THEO NGƯỠNG GỘP NHÓM HIẾM (THRESHOLD PROPOSALS) ===")
    print(thresh_df.to_string(index=False))


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_tv2_phase1_eda(
        parquet_path=os.path.join(base, "data", "dataset_1169.pq"),
        out_dir=os.path.join(base, "reports", "eda")
    )

