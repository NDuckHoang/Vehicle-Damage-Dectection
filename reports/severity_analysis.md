# Phân tích ước lượng mức độ hư hỏng (rule-based)

> Mức độ chỉ là ước lượng theo tiêu chí thử nghiệm (diện tích, số lượng và loại hư hỏng), không phải kết luận giám định chuyên nghiệp.

Sinh tự động bởi `scripts/analyze_severity.py`, model `yolov8n_seg/best.pt`, tập val 2.324 ảnh.

## 0. Phương pháp

VehiDE **không có nhãn mức độ**, nên mức độ được ước lượng bằng luật (rule-based), cài đặt trong `app/services/severity.py`:

1. **Từng vùng:** `điểm = trọng số loại + bậc diện tích`.
   - Trọng số loại là **giả định thử nghiệm**: Scratch = 1; Dent, Torn, Punctured = 2; Broken light, Broken glass, Lost part = 3 (bộ phận phải thay thế hoặc ảnh hưởng an toàn).
   - Bậc diện tích so tỉ lệ diện tích mask/ảnh với phân vị **của chính class đó** trong tập train: ≤ p50 là nhỏ (0), p50–p90 là vừa (1), > p90 là lớn (2).
   - Điểm ≤ 2 là **Nhẹ**, bằng 3 là **Trung bình**, ≥ 4 là **Nặng**.
2. **Từng ảnh:** lấy mức cao nhất trong các vùng; tăng 1 mức nếu tổng diện tích hư hỏng (hợp các mask) hoặc số vùng vượt p90 của dataset.
3. **Nhiều ảnh cùng trường hợp** (do người dùng nhóm): lấy mức cao nhất trong các ảnh; số vùng được cộng dồn, **không** loại bỏ trùng lặp giữa các góc chụp.

## 1. Ngưỡng diện tích (tính từ nhãn tập train)

| Class | Trọng số loại | Số vùng | p50 (% ảnh) | p90 (% ảnh) |
|---|---|---|---|---|
| Dent | 2 | 4709 | 3.08 | 19.75 |
| Scratch | 1 | 12266 | 0.53 | 7.19 |
| Broken glass | 3 | 1824 | 25.56 | 84.33 |
| Lost part | 3 | 2370 | 10.61 | 48.23 |
| Punctured | 2 | 2006 | 0.32 | 4.89 |
| Torn | 2 | 4545 | 0.70 | 5.83 |
| Broken light | 3 | 2324 | 3.38 | 24.08 |

Ngưỡng lan rộng cấp ảnh: tổng diện tích > 50.9% ảnh hoặc số vùng > 5.

## 2. Phân bố mức độ cấp ảnh

| Mức | Từ nhãn thật | Từ dự đoán |
|---|---|---|
| Nhẹ | 413 (17.8%) | 384 (16.5%) |
| Trung bình | 963 (41.4%) | 794 (34.2%) |
| Nặng | 948 (40.8%) | 843 (36.3%) |
| Không phát hiện | 0 (0.0%) | 303 (13.0%) |

## 3. Mức độ từng vùng theo loại hư hỏng (nhãn thật)

| | Nhẹ | Trung bình | Nặng |
|---|---|---|---|
| Broken glass | 0 | 223 | 174 |
| Broken light | 0 | 219 | 239 |
| Dent | 458 | 400 | 114 |
| Lost part | 0 | 224 | 224 |
| Punctured | 225 | 151 | 41 |
| Scratch | 2113 | 267 | 0 |
| Torn | 499 | 376 | 88 |

## 4. Nhất quán giữa mức từ nhãn thật và từ dự đoán

Hàng: mức tính từ nhãn thật. Cột: mức tính từ dự đoán của model.

| | Nhẹ | Trung bình | Nặng | Không phát hiện |
|---|---|---|---|---|
| Nhẹ | 258 | 45 | 10 | 100 |
| Trung bình | 94 | 581 | 127 | 161 |
| Nặng | 32 | 168 | 706 | 42 |

- Trùng khớp chính xác: **66.5%** số ảnh.
- Lệch không quá 1 mức (không tính ảnh bị bỏ sót): 85.2%.
- Dự đoán thấp hơn: 294 ảnh; cao hơn: 182 ảnh; model không phát hiện gì: 303 ảnh.

Lưu ý: bảng này **không** đo độ chính xác của mức độ (không có nhãn severity thật), chỉ đo mức ảnh hưởng của sai số model lên kết quả ước lượng.

Biểu đồ: `results/severity/severity_distribution.png`. Chi tiết từng ảnh: `results/severity/severity_per_image.csv`.
