# Danh gia model YOLOv8-Seg - VehiDE

> Bao cao sinh tu dong boi `scripts/evaluate.py`. Moi so lieu lay tu `model.val()` cua Ultralytics tren tap validation goc (2.324 anh, 6.035 vung hu hong).

## 1. Bang tong hop

| Model | Epoch (best) | Thoi gian train | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 | Inference (ms/anh) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| yolov8n_seg | 100 (94) | 5.59 h | 0.576 | 0.449 | 0.468 | 0.292 | 0.544 | 0.425 | **0.430** | **0.230** | 5.1 |

## 2. Chi so tung class - yolov8n_seg

| Class | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|---|---|---|---|
| Broken glass | 0.775 | 0.733 | 0.785 | 0.569 | 0.741 | 0.700 | 0.740 | 0.473 |
| Lost part | 0.658 | 0.648 | 0.656 | 0.459 | 0.640 | 0.630 | 0.640 | 0.402 |
| Broken light | 0.654 | 0.515 | 0.538 | 0.303 | 0.576 | 0.454 | 0.448 | 0.203 |
| Punctured | 0.537 | 0.463 | 0.452 | 0.274 | 0.524 | 0.451 | 0.437 | 0.241 |
| Torn | 0.443 | 0.285 | 0.291 | 0.167 | 0.430 | 0.276 | 0.264 | 0.109 |
| Dent | 0.487 | 0.230 | 0.260 | 0.115 | 0.470 | 0.222 | 0.244 | 0.098 |
| Scratch | 0.475 | 0.266 | 0.291 | 0.154 | 0.430 | 0.241 | 0.234 | 0.084 |

Nhan xet tu dong:
- Class tot nhat (Mask mAP50): **Broken glass** (0.740).
- Class kem nhat (Mask mAP50): **Scratch** (0.234).
- Mask Precision 0.544 vs Recall 0.425: model bo sot nhieu vung hu hong (recall thap hon precision).

Hinh anh:
- Bieu do training: `results/evaluation/yolov8n_seg/training_curves.png`
- Confusion matrix: `results/evaluation/yolov8n_seg/confusion_matrix_normalized.png`
- Duong cong PR (mask): `results/evaluation/yolov8n_seg/MaskPR_curve.png`
- Anh du doan: `results/evaluation/yolov8n_seg/predictions/`

<!-- PHAN TICH VIET TAY: noi dung ben duoi duoc giu nguyen khi chay lai -->
## 3. Phân tích kết quả (YOLOv8n-Seg)

### 3.1. Quá trình huấn luyện
- Train 100 epoch (batch 8, imgsz 640, GPU RTX 3050 Laptop 4GB), mất 5,6 giờ. Early stopping (patience 20) không kích hoạt. Epoch tốt nhất là 94.
- mAP trên tập val tăng nhanh trong khoảng 40 epoch đầu, sau đó gần như bão hòa từ khoảng epoch 70 (tăng dưới 0,01).
- `val/seg_loss` gần như không giảm từ khoảng epoch 50, trong khi `train/seg_loss` vẫn giảm, nhất là sau epoch 90 khi Ultralytics tắt augmentation mosaic. Đây là dấu hiệu model bắt đầu khớp quá mức (overfitting) nhẹ. Train thêm epoch với YOLOv8n khó cải thiện đáng kể.

### 3.2. Theo từng class
- **Tốt:** Broken glass (Mask mAP50 = 0,74) và Lost part (0,64). Đây là các vùng lớn, ranh giới rõ, ít nhầm với class khác. Theo thống kê ở Giai đoạn 1, Broken glass có diện tích trung vị khoảng 25% ảnh.
- **Trung bình:** Broken light (0,45) và Punctured (0,44).
- **Kém:** Torn (0,26), Dent (0,24), Scratch (0,23).
  - Scratch và Torn là các vùng **nhỏ và mảnh** (diện tích trung vị lần lượt khoảng 0,5% và 0,7% ảnh). Ở imgsz 640, các chi tiết này chỉ còn vài pixel, nên mask rất khó khớp chính xác. Vì vậy mAP50-95 rất thấp (0,08–0,11).
  - Dent là vết móp, chủ yếu thể hiện qua sự thay đổi phản chiếu ánh sáng; ranh giới không rõ và phụ thuộc người gán nhãn.

### 3.3. Confusion matrix (chuẩn hóa theo cột của nhãn thật)
- Tỉ lệ dự đoán đúng class trên đường chéo đạt 0,65–0,89. Như vậy **khi đã phát hiện được vùng hư hỏng, model phân loại khá đúng**.
- Nhầm lẫn chính:
  - 14% vùng Dent bị dự đoán là Scratch, vì móp thường đi kèm trầy sơn.
  - 13% vùng Torn bị dự đoán là Scratch, và 12% vùng Punctured bị dự đoán là Torn.
- Hàng **background** cho thấy 6–15% vùng thật bị bỏ sót hoàn toàn.
- Cột **background** cho thấy trong các dự đoán sai, phần lớn (53%) bị gán là Scratch. Model hay báo trầy xước ở chỗ không được gán nhãn, như vết bẩn, phản chiếu hoặc chữ viết tay của giám định viên.

### 3.4. Precision và Recall
- Mask Precision là 0,544, cao hơn Recall 0,425. Model khá "thận trọng": các vùng nó báo phần lớn là đúng, nhưng nó **bỏ sót** khoảng một nửa số vùng hư hỏng, chủ yếu là vết trầy và vết rách nhỏ.
- Khi dùng trong Web demo có thể hạ ngưỡng confidence (ví dụ từ 0,25 xuống 0,15) để tăng recall, chấp nhận nhiều báo nhầm hơn.

### 3.5. Hạn chế và hướng cải thiện
- Thử **YOLOv8s-Seg** (nhiều tham số hơn khoảng 3,5 lần), dự kiến cải thiện các class nhỏ, nhưng cần khoảng 10–12 giờ train trên máy cá nhân.
- Tăng `imgsz` lên 800 hoặc 1024 để giữ chi tiết của vết trầy và vết rách nhỏ (tốn VRAM và thời gian hơn).
- Dữ liệu lệch class: Scratch có 12.267 polygon, gấp khoảng 6,7 lần Broken glass (1.824).
- Nhãn hư hỏng vốn mang tính chủ quan (ranh giới của vết móp và vết trầy), nên mAP50-95 khó đạt cao.
