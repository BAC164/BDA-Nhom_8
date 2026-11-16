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
