# Test case – Hệ thống ước lượng hư hỏng phương tiện

> Sinh tự động bởi `scripts/run_test_cases.py`. Kết quả là **kết quả chạy thực tế** của model `yolov8n_seg/best.pt` (ngưỡng confidence 0.25). Ảnh hư hỏng được chọn **ngẫu nhiên** (seed cố định) từ tập validation theo nhãn thật, không chọn lọc.

**Tổng kết: 8/10 test case đạt.**

| Mã | Test case | Đầu vào | Kết quả mong đợi | Kết quả thực tế | Đánh giá |
|---|---|---|---|---|---|
| TC01 | Ảnh xe bị móp | `06012020_082651image242461.jpg` (nhãn thật: 1 vùng Dent) | Phát hiện ít nhất 1 vùng **Dent** | 2 vùng: Dent 0.37 (1.4%, Nhẹ), Scratch 0.36 (1.3%, Nhẹ). Mức ảnh: **Nhẹ** | ✅ Đạt |
| TC02 | Ảnh xe trầy xước | `02012020_102506image157881.jpg` (nhãn thật: 1 vùng Scratch) | Phát hiện ít nhất 1 vùng **Scratch** | 0 vùng | ❌ Không đạt |
| TC03 | Ảnh vỡ kính | `14102020_092741image226664.jpg` (nhãn thật: 1 vùng Broken glass) | Phát hiện ít nhất 1 vùng **Broken glass** | 1 vùng: Broken glass 0.86 (23.7%, Trung bình). Mức ảnh: **Trung bình** | ✅ Đạt |
| TC04 | Ảnh có nhiều loại hư hỏng | `08022020_090048image657700.jpg` (nhãn thật: Lost part, Scratch, Torn) | Phát hiện ≥ 2 loại có trong nhãn thật | 1 vùng: Torn 0.38 (5.7%, Trung bình). Mức ảnh: **Trung bình**. Trùng nhãn thật: Torn | ❌ Không đạt |
| TC05 | Ảnh xe không hư hỏng | `bus.jpg` (ảnh mẫu Ultralytics, xe buýt nguyên vẹn) | Không phát hiện vùng hư hỏng → "Không phát hiện hư hỏng" | 0 vùng → Không phát hiện hư hỏng | ✅ Đạt |
| TC06a | Ảnh chất lượng thấp (nhỏ + mờ) | Ảnh TC01 thu nhỏ 240×180, làm mờ Gaussian 7×7 | Vẫn xử lý, có cảnh báo độ phân giải thấp và mờ | Cảnh báo: Ảnh độ phân giải thấp (240x180), kết quả có thể kém chính xác.; Ảnh có vẻ bị mờ (độ nét 16).. 0 vùng | ✅ Đạt |
| TC06b | Ảnh chất lượng thấp (quá tối) | Ảnh TC01 giảm độ sáng còn 12% | Vẫn xử lý, có cảnh báo ảnh quá tối | Cảnh báo: Ảnh có vẻ bị mờ (độ nét 2).; Ảnh quá tối (độ sáng trung bình 16/255).. 0 vùng | ✅ Đạt |
| TC07 | Ảnh không hợp lệ | Tệp văn bản đổi đuôi `.jpg` và tệp rỗng `.jpg` | Không lỗi chương trình; mỗi tệp báo "không phải ảnh hợp lệ"; mức chung "Không phát hiện" | Lỗi từng ảnh: ['Tệp không phải ảnh hợp lệ hoặc ảnh bị hỏng.', 'Tệp không phải ảnh hợp lệ hoặc ảnh bị hỏng.']; mức chung: Không phát hiện hư hỏng | ✅ Đạt |
| TC08 | Nhiều ảnh cùng một trường hợp | 3 ảnh của TC01, TC02, TC03 trong 1 lần tải lên | Tổng số vùng = tổng các ảnh; mức chung = mức cao nhất; có ghi chú chưa loại trùng lặp | Mức từng ảnh: ['Nhẹ', 'Không phát hiện hư hỏng', 'Trung bình']; mức chung: **Trung bình**; 3 vùng; loại: Dent×1, Scratch×1, Broken glass×1 | ✅ Đạt |
| TC09 | Web demo (Flask test client) | Các request tới các route | Mọi route trả đúng mã HTTP | GET /: OK; POST không có ảnh → quay về trang chủ: OK; POST ảnh + tệp hỏng + .txt → trang kết quả: OK; GET trang kết quả: OK; Tải báo cáo html: OK; Tải báo cáo csv: OK; Tải báo cáo json: OK; Mã trường hợp sai → 404: OK; Định dạng tải sai → 404: OK | ✅ Đạt |

Ảnh kết quả (trái: ảnh gốc, phải: dự đoán): `results/test_cases/TCxx.jpg`. Báo cáo của TC07, TC08: `results/test_cases/cases/`.

Ghi chú:
- TC01–TC05 đánh giá **chất lượng model**: kết quả "Không đạt" phản ánh hạn chế thực tế của model (recall thấp với vùng nhỏ, báo nhầm), không phải lỗi chương trình.
- TC06–TC09 đánh giá **hành vi hệ thống** (cảnh báo, xử lý lỗi, tổng hợp, web).

<!-- PHAN TICH VIET TAY: noi dung ben duoi duoc giu nguyen khi chay lai -->

## Phân tích các test case không đạt

- **TC02 (trầy xước):** nhãn thật chỉ có **1 vết trầy rất mảnh** (polygon 9 điểm) ở mép bậc lên xuống, gần như không nhìn thấy ở độ phân giải 640. Model không phát hiện, đúng với phân tích ở `model_evaluation.md`: Scratch có recall thấp nhất (0,24). Muốn cải thiện có thể hạ ngưỡng confidence, tăng `imgsz` hoặc dùng model lớn hơn.
- **TC04 (nhiều loại hư hỏng):** nhãn thật gồm Torn, 2 vùng Scratch và Lost part. Model nhận đúng mảnh ốp bị rách (**Torn**) nhưng bỏ sót các vết trầy nhỏ và phần mất bộ phận nằm ở góc ảnh. Mức độ ước lượng vì vậy thấp hơn thực tế.
- **TC08:** ảnh TC02 không phát hiện được gì nên mức chung chỉ dựa trên 2 ảnh còn lại. Đây là lợi ích của việc dùng nhiều góc chụp: một ảnh bỏ sót thì các ảnh khác vẫn bổ sung được.

## Kết luận
- **Hành vi hệ thống** (TC05–TC09): đạt toàn bộ. Hệ thống không phát hiện nhầm trên ảnh xe nguyên vẹn, cảnh báo đúng ảnh chất lượng thấp, xử lý được tệp lỗi mà không dừng chương trình, tổng hợp nhiều ảnh đúng luật, và các route web hoạt động đúng.
- **Chất lượng model** (TC01–TC04): đạt 2/4. Model tốt với vùng lớn, rõ (kính vỡ) và còn yếu với vết trầy nhỏ, đúng như các chỉ số mAP đã đo.
