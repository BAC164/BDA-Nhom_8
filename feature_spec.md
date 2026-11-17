# 📋 ĐẶC TẢ BẢNG ĐẶC TRƯNG V1 (`feature_spec.md`)

* **Chủ sở hữu:** TV2 (Kỹ sư đặc trưng — Phase 3, Nhóm 8 - Đề tài 20)
* **Đầu vào từ Phase 2 (TV1 - Thịnh, bản `tv1_v1`):** `splits/{train,val,test}` (`376,749` / `81,606` / `81,028` dòng)
* **Môi trường thống nhất:** `Python 3.13` / `PySpark 4.0.1` / `Java 17`
* **Phiên bản đóng băng:** `data-freeze-v1`
* **Đường dẫn lưu trữ Parquet:** `data/features/v1/{train,val,test}/` (trên Drive: `BigData_Topic20/01_data_frozen/features_v1/`)
* **Đường dẫn bảng tra cứu (Lookup Maps cho Demo Phase 6):** `data/features/v1/lookup_maps/`
* **Module mã nguồn tạo đặc trưng & Pipeline:** [`src/features.py`](src/features.py)

---

## 1. Bảng quy ước & Tham số cố định ở Phase 3

| Tham số | Giá trị | Ý nghĩa kỹ thuật |
| :--- | :---: | :--- |
| `SEED` | `42` | Hạt giống ngẫu nhiên toàn dự án (chia tập `SHA256('split:42:' + row_id)` và chia 5 folds `SHA256('fold:42:' + row_id)`). |
| `NUM_FOLDS` | `5` | Số fold trên tập `train` (`0..4`) phục vụ Out-of-Fold Target Encoding (kể cả OOF `prior`). |
| `SMOOTHING_M` ($m$) | `30.0` | Hệ số làm mượt $m$-estimate trong công thức Target Encoding. |
| `MIN_COUNT_AIRPORT` ($N_{\text{airport}}$) | `100` | Ngưỡng số chuyến tối thiểu trong tập `train` để giữ nguyên mã sân bay; dưới ngưỡng gộp vào `"OTHER"`. |
| `MIN_COUNT_ROUTE` ($N_{\text{route}}$) | `50` | Ngưỡng số chuyến tối thiểu trong tập `train` để giữ nguyên mã tuyến bay; dưới ngưỡng gộp vào `"OTHER"`. |
| `maxBins` (cho GBT) | `300` | Số lượng bin tối đa cho `GBTClassifier` ($\ge$ số lượng category lớn nhất của `AirportFrom_g` / `AirportTo_g` tính cả `"OTHER"` và bucket `"unseen"`). |

---

## 2. Danh mục chi tiết các cột trong `features_v1/`

> **Lưu ý quan trọng cho TV3 và TV4:** Bảng Parquet `features_v1/` **KHÔNG** lưu các cột đã qua `StringIndexer` (`*_i`) hay `OneHotEncoder` (`*_oh`) vì việc lưu ra Parquet sẽ làm mất metadata `NominalAttribute` của Spark MLlib. Các bước `StringIndexer` và `OneHotEncoder` đã được đóng gói sẵn bên trong hai hàm `make_pipeline_lr()` và `make_pipeline_gbt(max_bins=300)` trong [`src/features.py`](src/features.py). Tuyệt đối không đưa `Delay`, `row_id`, `fold` vào predictors.

| STT | Tên cột | Kiểu dữ liệu | Cách tạo / Công thức | Dùng cho | Ghi chú & Chống rò rỉ nhãn |
| :-: | :--- | :--- | :--- | :---: | :--- |
| 1 | `row_id` | `long` | Vị trí dòng gốc (`0..539382`) từ Phase 2 | Metadata / Join | Dùng để truy vết dòng và ghép kết quả dự đoán ở Phase 4–5. Tuyệt đối **không** đưa vào `features`. |
| 2 | `Delay` | `int` | Ép kiểu `0` / `1` từ dữ liệu gốc | **Nhãn (`labelCol`)** | Lớp dương là `Delay = 1` (chuyến bay bị trễ). |
| 3 | `fold` | `int` | `SHA256('fold:42:' + row_id) % 5` (giá trị `0..4`) | Chỉ có ở `train` | Dùng nội bộ cho Out-of-Fold Target Encoding trên tập `train`. Không đưa vào `features`. |
| 4 | `Airline` | `string` | Giữ nguyên mã 18 hãng bay | **Cả hai (LR & GBT)** | LR dùng `StringIndexer` + `OneHotEncoder`; GBT dùng `StringIndexer` trực tiếp trong pipeline. |
| 5 | `Flight` | `string` | Ép kiểu sang chuỗi từ mã số hiệu chuyến bay (`6,585` giá trị) | Trung gian | Bản chất là mã định danh, không đưa trực tiếp dạng số vào mô hình để tránh hiểu nhầm thứ tự độ lớn. |
| 6 | `AirportFrom` | `string` | Mã IATA sân bay đi (`293` giá trị) | Trung gian | Biến phân loại nhiều giá trị gốc; dùng để tạo `AirportFrom_g`, `route`, `log_freq_from`, `density_from`. |
| 7 | `AirportTo` | `string` | Mã IATA sân bay đến (`293` giá trị) | Trung gian | Biến phân loại nhiều giá trị gốc; dùng để tạo `AirportTo_g`, `route`, `log_freq_to`, `density_to`. |
| 8 | `DayOfWeek` | `string` | Chuỗi `"1"` (Thứ Hai) đến `"7"` (Chủ Nhật) | **Cả hai (LR & GBT)** | LR dùng One-Hot Encoding; GBT dùng `StringIndexer`. |
| 9 | `Time` | `double` | Phút dự kiến khởi hành tính từ 00:00 nửa đêm (`0..1439`) | Trung gian | Dùng để trích xuất `hour`, `hour_cat`, `time_bucket`. |
| 10 | `Length` | `double` | Thời lượng chuyến bay dự kiến tính bằng phút (đã điền trung vị Train `115.0` cho 4 dòng `<= 0`) | **Cả hai (LR & GBT)** | Biến số liên tục hợp lệ tại thời điểm khởi hành dự kiến. |
| 11 | `Length_invalid` | `int` | Cờ đánh dấu từ TV1 (`1` nếu `Length` gốc `<= 0`, ngược lại `0`) | **Cả hai (LR & GBT)** | Có 4 dòng bị đánh dấu (`3` ở train, `1` ở val, `0` ở test). Cho phép bật/tắt qua `include_length_invalid`. |
| 12 | `hour` | `double` | `floor(Time / 60.0)` (nhận giá trị `0.0` $\rightarrow$ `23.0`) | **GBT** | Cây quyết định GBT tự cắt ngưỡng phi tuyến theo giờ trong ngày. |
| 13 | `hour_cat` | `string` | `cast(floor(Time / 60) as string)` (`"0"` $\rightarrow$ `"23"`) | **LR** | Quan hệ giữa giờ bay và tỷ lệ trễ không đơn điệu (tăng về chiều tối, giảm về đêm) nên LR bắt buộc dùng One-Hot Encoding qua `hour_cat`. |
| 14 | `time_bucket` | `string` | `"morning"` (5–11h), `"afternoon"` (12–16h), `"evening"` (17–20h), `"night"` (21–4h) | Bổ trợ / Phân tích lỗi | Dùng cho phân tích lát cắt (Slice Analysis) và thử nghiệm mở rộng. |
| 15 | `is_weekend` | `double` | `1.0` nếu `DayOfWeek` $\in \{\text{"6"}, \text{"7"}\}$, ngược lại `0.0` | Bổ trợ (LR / GBT) | Phản ánh sự sụt giảm mật độ bay thương mại vào cuối tuần. |
| 16 | `len_bin` | `string` | `"short"` ($\le 90$), `"medium"` ($91\text{–}180$), `"long"` ($181\text{–}300$), `"ultra_long"` ($>300$) | Bổ trợ / EDA | Phân nhóm độ dài chuyến bay. |
| 17 | `route` | `string` | `concat_ws("_", AirportFrom, AirportTo)` (ví dụ `"ATL_ORD"`) | Trung gian | Cặp tuyến bay đi – đến (~4,000+ cặp). |
| 18 | `airline_flight` | `string` | `concat_ws("_", Airline, Flight)` (ví dụ `"WN_1420"`) | Trung gian | Định danh duy nhất của số hiệu chuyến bay theo từng hãng. |
| 19 | `density_from` | `double` | Số chuyến bay theo `(AirportFrom, DayOfWeek, hour)` tính **trên tập `train`** | **Cả hai (LR & GBT)** | Không dùng nhãn `Delay`; chỉ học từ `train` để tuân thủ chặt quy tắc của TV1 (`0.0` nếu unseen). |
| 20 | `density_to` | `double` | Số chuyến bay theo `(AirportTo, DayOfWeek, hour)` tính **trên tập `train`** | **Cả hai (LR & GBT)** | Đo mức độ tắc nghẽn tại sân bay hạ cánh; chỉ học từ `train` (`0.0` nếu unseen). |
| 21 | `AirportFrom_g` | `string` | Giữ mã sân bay nếu `count_train >= 100`, ngược lại gán `"OTHER"` | **GBT** | Đưa vào `StringIndexer` trong `make_pipeline_gbt(max_bins=300)`. |
| 22 | `AirportTo_g` | `string` | Giữ mã sân bay nếu `count_train >= 100`, ngược lại gán `"OTHER"` | **GBT** | Đưa vào `StringIndexer` trong `make_pipeline_gbt(max_bins=300)`. |
| 23 | `route_g` | `string` | Giữ mã tuyến bay nếu `count_train >= 50`, ngược lại gán `"OTHER"` | Bổ trợ | Ngưỡng xác định hoàn toàn trên tập `train`. |
| 24 | `log_freq_from` | `double` | $\ln(1 + \text{count}_{\text{train}}(\text{AirportFrom}))$ | **LR** (và Ablation GBT) | Chỉ đếm trên `train`. Unseen category ở `val`/`test` nhận `0.0`. |
| 25 | `log_freq_to` | `double` | $\ln(1 + \text{count}_{\text{train}}(\text{AirportTo}))$ | **LR** (và Ablation GBT) | Chỉ đếm trên `train`. Unseen category ở `val`/`test` nhận `0.0`. |
| 26 | `log_freq_route` | `double` | $\ln(1 + \text{count}_{\text{train}}(\text{route}))$ | **Cả hai (LR & GBT)** | Đo độ phổ biến của đường bay trên `train`. Unseen nhận `0.0`. |
| 27 | `log_freq_flight` | `double` | $\ln(1 + \text{count}_{\text{train}}(\text{airline\_flight}))$ | Bổ trợ (Ablation) | Tần suất xuất hiện của số hiệu chuyến bay trong `train`. |
| 28 | `te_route` | `double` | Out-of-Fold Smoothed Target Encoding của `route` ($m=30$) | **GBT** | Trên `train` tính từ 4 folds còn lại (kể cả OOF `prior`); trên `val`/`test` tính từ toàn bộ `train`. |
| 29 | `te_flight` | `double` | Out-of-Fold Smoothed Target Encoding của `airline_flight` ($m=30$) | **GBT** | Khai thác lịch sử trễ của từng số hiệu chuyến bay mà không làm nổ số chiều. |
| 30 | `te_from` | `double` | Out-of-Fold Smoothed Target Encoding của `AirportFrom_g` ($m=30$) | Bổ trợ / Ablation | Tỷ lệ trễ làm mượt của sân bay khởi hành. |
| 31 | `te_to` | `double` | Out-of-Fold Smoothed Target Encoding của `AirportTo_g` ($m=30$) | Bổ trợ / Ablation | Tỷ lệ trễ làm mượt của sân bay hạ cánh. |

---

## 3. Công thức toán học của Out-of-Fold Target Encoding (kể cả OOF `prior`)

Gọi $m = 30$ là hệ số làm mượt (smoothing parameter):

1. **Đối với bản ghi thuộc fold $f \in \{0,1,2,3,4\}$ trong tập `train`:**
   * Tính tỷ lệ trễ trung bình Out-of-Fold (loại trừ chính fold $f$):
     $$p_{\text{OOF}}(f) = \frac{S_{\text{toàn}} - S_{\text{fold}}(f)}{N_{\text{toàn}} - N_{\text{fold}}(f)}$$
   * Giá trị Target Encoding cho nhóm $c$ ở fold $f$:
     $$\text{te}_{\text{OOF}}(c, f) = \frac{S_{\text{toàn}}(c) - S_{\text{fold}}(c, f) + m \cdot p_{\text{OOF}}(f)}{N_{\text{toàn}}(c) - N_{\text{fold}}(c, f) + m}$$
   *(Tuyệt đối không dùng nhãn của chính fold $f$ cho cả tử/mẫu nhóm lẫn `prior`).*

2. **Đối với bản ghi trong tập `val`, `test` hoặc khi suy luận Demo (Phase 6):**
   $$\text{te}_{\text{full}}(c) = \frac{S_{\text{toàn}}(c) + m \cdot p}{N_{\text{toàn}}(c) + m}, \quad \text{với } p = \bar{y}_{\text{train}}$$
   Nếu category $c$ chưa từng xuất hiện trong tập `train` ($N_{\text{toàn}}(c) = 0$), giá trị trả về mặc định là $\text{prior} = p$.

---

## 4. Hướng dẫn gọi hàm nhanh cho TV3 và TV4

```python
from src.features import make_pipeline_lr, make_pipeline_gbt

# Dành cho TV3 (Logistic Regression):
pipe_lr = make_pipeline_lr(reg_param=0.01, elastic_net_param=0.5, max_iter=100)
model_lr = pipe_lr.fit(train_df)
preds_val_lr = model_lr.transform(val_df)

# Dành cho TV4 (GBTClassifier):
pipe_gbt = make_pipeline_gbt(max_bins=300, max_iter=100, max_depth=5, step_size=0.1, seed=42)
model_gbt = pipe_gbt.fit(train_df)
preds_val_gbt = model_gbt.transform(val_df)
```
