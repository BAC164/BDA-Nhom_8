# 📝 NHẬT KÝ PHIÊN BẢN ĐẶC TRƯNG (`CHANGELOG_FEATURES.md`)

* **Chủ sở hữu:** TV2 (Kỹ sư đặc trưng)
* **Quy tắc:** Sau khi gắn Git tag `data-freeze-v1`, mọi thay đổi đối với [`src/features.py`](src/features.py), [`feature_spec.md`](feature_spec.md) hoặc thư mục `01_data_frozen/features_v1/` bắt buộc phải được ghi nhận tại đây và thông báo cho TV1, TV3, TV4, TV5 trước khi thực hiện.

---

## [`data-freeze-v1`] — Đóng băng bảng đặc trưng v1 (Phase 3)

### 1. Tóm tắt các đặc trưng đã phát hành (30 cột trên `val`/`test`, 31 cột trên `train`)
* **Định danh & Nhãn (không đưa vào predictors):** `row_id`, `Delay` (`0`/`1`), `fold` (`0..4` chỉ có trên `train`).
* **Cờ dữ liệu từ TV1 (`tv1_v1`):** `Length_invalid` (`1` với 4 dòng `Length <= 0`, `0` với các dòng còn lại).
* **Đặc trưng cơ bản:** `hour` (`double`), `hour_cat` (`string`), `time_bucket`, `is_weekend`, `len_bin`, `route`, `airline_flight`.
* **Mật độ chuyến bay (Flight Density):** `density_from`, `density_to` (tính từ tập `train` theo sân bay $\times$ thứ $\times$ giờ).
* **Gộp nhóm hiếm (`OTHER`):** `AirportFrom_g` ($N \ge 100$), `AirportTo_g` ($N \ge 100$), `route_g` ($N \ge 50$).
* **Frequency Encoding (`log1p` trên `train`):** `log_freq_from`, `log_freq_to`, `log_freq_route`, `log_freq_flight`.
* **Out-of-Fold Target Encoding ($m = 30.0$, gồm cả `oof_prior` từ 4 folds còn lại):** `te_route`, `te_flight`, `te_from`, `te_to`.

### 2. Bộ kiểm định tích hợp trong `03_features_tv2.ipynb`
* Đối chiếu checksum & schema của đầu vào `tv1_v1` (`376,749` / `81,606` / `81,028` dòng).
* Kiểm tra không có giá trị `NULL` / `NaN` trên tất cả các cột đặc trưng của `train`, `val`, `test`.
* Đối chiếu thủ công 5 giá trị `te_route` trên `val` và 3 cặp `(route, fold)` Out-of-Fold trên `train` (kể cả `oof_prior`) với sai số $< 10^{-6}$.
* Kiểm thử rò rỉ nhãn bằng mô hình đơn biến chỉ gồm `te_route` (so sánh `ROC-AUC` trên `train` và `val`).
* Smoke test cả 2 hàm `make_pipeline_lr().fit()` và `make_pipeline_gbt(max_bins=300).fit()`.
* Lưu `lookup_maps/`, `manifest.json` và `split_checksums.json` phục vụ TV3, TV4 và Demo ở Phase 6 của TV5.
