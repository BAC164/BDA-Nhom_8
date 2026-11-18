# Big Data — Nhóm 8, Đề tài 20

Dự đoán trễ chuyến bay với Airlines OpenML 1169.

## TV1 — Thịnh
Đã hoàn thành chuẩn bị dữ liệu, kiểm tra, làm sạch, chia tập và EDA.

Code trong `Thịnh_phase1,2/`:
- `TV1_Data_Preparation.ipynb`: chuẩn bị dữ liệu và EDA.
- `02_Read_Shared_Data.ipynb`: đọc dữ liệu để làm tiếp.

## Dữ liệu
[Thư mục tv1_v1 trên Drive](https://drive.google.com/drive/folders/1lRwo009SUVGNZnKkvHhX858NBGwnGU-Q?usp=sharing)

Đọc `README_HANDOFF.md` trong Drive, mở notebook đọc dữ liệu
và sửa `BASE` theo đường dẫn Drive của mình.
Nếu chưa có quyền truy cập, liên hệ Thịnh.

## Quy tắc
- Dùng nguyên `splits/`: Train 376749, Validation 81606, Test 81028 dòng.
- Không chia lại; không đưa Delay, row_id, fold vào predictors.
- Fit tiền xử lý trên Train; target encoding dùng OOF.
- Khi dùng CV, fit tiền xử lý trong từng lượt huấn luyện.
- Validation để chọn mô hình/ngưỡng; Test chỉ đánh giá cuối.
- Khoảng 77% Val/Test trùng bộ đặc trưng với Train:
  không diễn giải kết quả thành khả năng dự báo tương lai/lịch bay mới.

Bảng kiểm tra, biểu đồ và nhận xét nằm trong `reports/` trên Drive.


---

## TV2 — Hùng
Đã hoàn thành phần EDA phối hợp (Phase 1) và Feature Engineering & Đóng băng `features_v1` (Phase 3).

Code / tài liệu bàn giao:
- `01_eda_tv2_part.ipynb` & `eda_tv2_phase1.py`: EDA phần TV2 (`eda_findings.md`, `reports/eda/`).
- `03_features_tv2.ipynb` & `features.py`: tạo đặc trưng `features_v1`, pipeline `make_pipeline_lr()` / `make_pipeline_gbt()`, kiểm định rò rỉ nhãn.
- `feature_spec.md` & `CHANGELOG_FEATURES.md`: đặc tả 30 cột (`val`/`test`) và 31 cột (`train` gồm `fold`).

### Dữ liệu TV2
https://drive.google.com/drive/folders/1ksFuS9Ox4Juok_fw4b0ZuOGf-2rnvPGE?usp=drive_link
Đọc `feature_spec.md`, sửa đường dẫn tới `01_data_frozen/features_v1` trên Drive của mình để chạy tiếp Phase 4–6.
