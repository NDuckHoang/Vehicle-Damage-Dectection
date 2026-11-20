# Ước lượng mức độ hư hỏng phương tiện từ ảnh hiện trường

Bài tập lớn môn Xử lý ảnh. Hệ thống nhận ảnh xe tại hiện trường và:

1. **Phát hiện và phân đoạn** vùng hư hỏng (segmentation mask) bằng **YOLOv8n-Seg** fine-tune trên dataset **VehiDE**.
2. **Phân loại** 7 loại hư hỏng: Móp/lõm (Dent), Trầy sơn (Scratch), Vỡ kính (Broken glass), Mất bộ phận (Lost part), Thủng (Punctured), Rách (Torn), Bể đèn (Broken light).
3. **Ước lượng mức độ** Nhẹ / Trung bình / Nặng bằng luật (rule-based), có giải thích căn cứ.
4. **Tổng hợp nhiều góc chụp** của cùng một trường hợp do người dùng nhóm, rồi xuất báo cáo HTML/CSV/JSON.
5. Cung cấp **Web demo** bằng Flask.

> ⚠ Mức độ hư hỏng là **ước lượng theo tiêu chí thử nghiệm**, không phải kết luận giám định chuyên nghiệp. Dataset VehiDE không có nhãn mức độ.

---

## 🚀 Dùng nhanh: chạy Web demo, KHÔNG cần train, KHÔNG cần dataset

Model đã train có sẵn trong repo (`models/trained/yolov8n_seg/weights/best.pt`, khoảng 7MB).

```powershell
git clone https://github.com/NDuckHoang/Vehicle-Damage-Dectection.git
cd Vehicle-Damage-Dectection
python -m venv .venv
```

**Cài thư viện.** Chọn **một** trong hai cách:
```powershell
# (a) Máy có GPU NVIDIA: cài theo requirements.txt (PyTorch CUDA, tải khoảng 2,5GB)
.venv\Scripts\python.exe -m pip install -r requirements.txt

# (b) Máy KHÔNG có GPU NVIDIA: cài PyTorch bản CPU (nhẹ hơn nhiều) rồi cài phần còn lại
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install ultralytics opencv-python numpy pandas matplotlib scikit-learn Flask
```

**Chạy Web.** Mọi lệnh phải chạy **trong thư mục repo**:
```powershell
.venv\Scripts\python.exe app\app.py
```
Mở trình duyệt tại **http://127.0.0.1:5000**, kéo thả ảnh xe rồi bấm **Phân tích**. Chạy bằng CPU thì mỗi ảnh mất khoảng 1 giây.

Chỉ khi muốn **train lại hoặc đánh giá lại** mới cần dataset VehiDE (xem các bước 4–7 bên dưới).

---

## Kết quả chính

| Hạng mục | Kết quả |
|---|---|
| Dataset | 11.621 ảnh train, 2.324 ảnh val, 36.081 polygon, 7 class; ảnh và nhãn ghép khớp 100% theo tên file |
| Model | YOLOv8n-Seg, 100 epoch, imgsz 640, batch 8, train 5,6 giờ trên RTX 3050 Laptop 4GB |
| Mask mAP50 / mAP50-95 (val) | **0,430 / 0,230** (Box: 0,468 / 0,292) |
| Class tốt nhất / kém nhất | Broken glass 0,74 / Scratch 0,23 (Mask mAP50) |
| Tốc độ suy luận | khoảng 5 ms/ảnh (GPU) |
| Test case | 8/10 đạt (hành vi hệ thống 6/6; chất lượng model 2/4) |

Chi tiết xem trong thư mục [reports/](reports/):
- [dataset_analysis.txt](reports/dataset_analysis.txt)
- [model_evaluation.md](reports/model_evaluation.md)
- [severity_analysis.md](reports/severity_analysis.md)
- [test_cases.md](reports/test_cases.md)

---

## Cấu trúc project

```
VehicleDamageProject/
├── dataset/                  # dữ liệu YOLO (tạo bởi convert_via_to_yolo.py)
│   ├── images/{train,val}/   #   hardlink tới ảnh gốc ở D:\Dataset (không tốn thêm dung lượng)
│   ├── labels/{train,val}/   #   file .txt YOLO-Seg
│   └── data.yaml
├── models/
│   ├── pretrained/           # yolov8n-seg.pt (tự tải)
│   └── trained/yolov8n_seg/  # best.pt, last.pt, results.csv, biểu đồ
├── scripts/
│   ├── inspect_dataset.py        # GĐ1: kiểm tra dataset, thống kê, biểu đồ
│   ├── convert_via_to_yolo.py    # GĐ2: VIA JSON → YOLOv8-Seg
│   ├── validate_dataset.py       # GĐ2/4: kiểm tra dataset YOLO (cả đổi ngược tọa độ)
│   ├── visualize_annotations.py  # GĐ3: vẽ polygon gốc và YOLO cạnh nhau
│   ├── train.py                  # GĐ4: fine-tune, có hỗ trợ resume
│   ├── evaluate.py               # GĐ5: chỉ số, confusion matrix, ảnh dự đoán, báo cáo
│   ├── analyze_severity.py       # GĐ6: phân tích luật ước lượng mức độ
│   └── run_test_cases.py         # GĐ9: chạy test case, ghi kết quả thực tế
├── app/
│   ├── app.py                    # GĐ8: Flask web demo
│   ├── services/
│   │   ├── detection.py          # chạy model, tính diện tích mask, vẽ kết quả, cảnh báo chất lượng ảnh
│   │   ├── severity.py           # luật ước lượng mức độ + hiệu chỉnh ngưỡng
│   │   ├── severity_config.json  # ngưỡng p50/p90 tính từ tập train
│   │   └── report.py             # xử lý nhiều ảnh, tổng hợp, xuất báo cáo
│   ├── templates/                # base, index, result, report_export (báo cáo tải về)
│   └── static/{css,js,results}/
├── results/                  # ảnh minh họa, đánh giá, test case
├── reports/                  # các báo cáo .md/.txt
├── requirements.txt
└── README.md
```

> Ghi chú: `inspect_dataset.py` đã gồm cả phần thống kê và phân tích dataset, nên không tạo thêm `analyze_dataset.py`.

---

## Hướng dẫn cài đặt và chạy (Windows)

### 1. Cài Python
- Tải Python **3.10 trở lên** từ https://www.python.org/downloads/. Project đã được kiểm thử với **Python 3.14.5**.
- Khi cài, tick **"Add python.exe to PATH"**.
- Kiểm tra: `python --version`.
- Nếu có GPU NVIDIA: cập nhật driver mới, rồi kiểm tra bằng `nvidia-smi`.

### 2. Tạo virtual environment
```powershell
cd D:\Dataset\VehicleDamageProject
python -m venv .venv
```
Tất cả các lệnh bên dưới đều gọi trực tiếp `.venv\Scripts\python.exe`, nên **không cần** kích hoạt venv. Cách này cũng tránh lỗi Execution Policy của PowerShell.

### 3. Cài thư viện
```powershell
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
`requirements.txt` cài PyTorch bản **CUDA 12.8**. Kiểm tra GPU:
```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Nếu kết quả là `False`: kiểm tra driver NVIDIA. Máy không có GPU vẫn chạy được Web demo bằng CPU (chậm hơn), nhưng **không nên train** bằng CPU.

### 4. Chuẩn bị dataset
Đặt dataset VehiDE gốc tại `D:\Dataset` theo cấu trúc:
```
D:\Dataset
├── image\image\*.jpg            (11.621 ảnh train)
├── validation\validation\*.jpg  (2.324 ảnh val)
├── 0Train_via_annos.json
└── 0Val_via_annos.json
```
Nếu đặt ở chỗ khác, sửa biến `DATASET_ROOT` trong các script.

Kiểm tra dataset (khoảng 5 phút). Kết quả ghi vào `reports/dataset_analysis.txt` cùng các biểu đồ:
```powershell
.venv\Scripts\python.exe scripts\inspect_dataset.py
```

### 5. Chuyển đổi annotation
```powershell
.venv\Scripts\python.exe scripts\convert_via_to_yolo.py      # khoảng 2,5 phút; log ghi vào reports/conversion_log.txt
.venv\Scripts\python.exe scripts\validate_dataset.py         # phải in "KET QUA: HOP LE"
.venv\Scripts\python.exe scripts\visualize_annotations.py    # ảnh minh họa ghi vào results/annotations/
```
Cách xử lý dữ liệu:
- Ảnh và nhãn được ghép theo **tên file chính xác** (kể cả ảnh có tên dạng hash).
- Polygon vượt biên được clip vào khung ảnh; polygon dưới 3 điểm hoặc có diện tích bằng 0 bị bỏ qua và ghi log.
- Giữ nguyên cách chia train/val gốc. **Không sửa** file JSON và ảnh gốc.

### 6. Huấn luyện
Nên chạy trong một cửa sổ PowerShell riêng, máy cắm sạc. 100 epoch mất khoảng 5–6 giờ trên RTX 3050.
```powershell
.venv\Scripts\python.exe scripts\train.py                                    # yolov8n-seg, 100 epoch, batch 8
.venv\Scripts\python.exe scripts\train.py --epochs 1 --fraction 0.1 --name smoke_test   # chạy thử nhanh
.venv\Scripts\python.exe scripts\train.py --resume models\trained\yolov8n_seg\weights\last.pt   # train tiếp khi bị gián đoạn
.venv\Scripts\python.exe scripts\train.py --model yolov8s-seg.pt --batch 4  # thử model lớn hơn
```
- Gặp lỗi `CUDA out of memory`: giảm `--batch` xuống 4.
- Nếu tên thư mục kết quả đã tồn tại, Ultralytics tự tạo `yolov8n_seg2`, `yolov8n_seg3`… nên **không ghi đè** model cũ.
- Script **không tự chuyển sang CPU**. Muốn train bằng CPU phải chỉ định `--device cpu`.

### 7. Đánh giá
```powershell
.venv\Scripts\python.exe scripts\evaluate.py                     # đánh giá best.pt trên tập val
.venv\Scripts\python.exe app\services\severity.py --calibrate    # tính ngưỡng diện tích cho luật mức độ (đã có sẵn)
.venv\Scripts\python.exe scripts\analyze_severity.py             # phân tích luật mức độ trên tập val
.venv\Scripts\python.exe scripts\run_test_cases.py               # chạy test case
```
Kết quả được ghi vào `results/evaluation/`, `results/severity/`, `results/test_cases/` và `reports/`.

### 8. Chạy Web demo
> ⚠ Mọi lệnh trong README phải chạy **từ thư mục project** (`D:\Dataset\VehicleDamageProject`). Nếu đứng ở `D:\Dataset`, PowerShell sẽ báo lỗi *"The module '.venv' could not be loaded"*.

```powershell
cd D:\Dataset\VehicleDamageProject
.venv\Scripts\python.exe app\app.py
```
Mở trình duyệt tại **http://127.0.0.1:5000**, kéo thả một hoặc nhiều ảnh của cùng một trường hợp rồi bấm **Phân tích**. Trang kết quả có nút tải báo cáo HTML, CSV và JSON. Mỗi trường hợp được lưu tại `app/static/results/<mã>/`.

Có thể xử lý từ dòng lệnh mà không cần web:
```powershell
# Thay bằng đường dẫn ảnh thật, ví dụ 2 ảnh trong tập val:
.venv\Scripts\python.exe -m app.services.report dataset\images\val\21012020_104903image172712.jpg dataset\images\val\02012020_141718image85820.jpg --name "Xe ABC"
```

### 9. Đánh giá chủ quan (người chấm)
Mức độ hư hỏng không có nhãn thật, nên hệ thống được đánh giá thêm bằng **ý kiến của người chấm**:
```powershell
.venv\Scripts\python.exe scripts\subjective_prepare.py   # chọn 10 ảnh val, tạo subjective\phieu_danh_gia.html
```
1. Gửi file `subjective\phieu_danh_gia.html` (khoảng 3,4MB) cho từng người chấm. Họ mở bằng Chrome hoặc Edge, chấm 10 ảnh rồi bấm **Tải file kết quả (CSV)**.
2. Nhận các file `danh_gia_<tên>.csv` và chép vào `subjective\responses\`.
3. Phân tích:
```powershell
.venv\Scripts\python.exe scripts\subjective_analyze.py   # tạo reports\subjective_evaluation.md
```
Các chỉ số được tính: MOS chất lượng phát hiện (1–5); tỉ lệ đồng ý giữa hệ thống và người chấm về mức độ, kèm Cohen's Kappa có trọng số; Fleiss' Kappa giữa các người chấm.

> Không chạy lại `subjective_prepare.py` sau khi đã gửi phiếu. Script sẽ từ chối ghi đè `sample.json`, để danh sách ảnh luôn khớp với phiếu đã gửi.

---

## Phương pháp ước lượng mức độ (tóm tắt)

1. **Từng vùng:** `điểm = trọng số loại + bậc diện tích`.
   - Trọng số loại: Scratch = 1; Dent, Torn, Punctured = 2; Broken light, Broken glass, Lost part = 3. Đây là **giả định thử nghiệm**, có thể chỉnh trong `TYPE_WEIGHT`.
   - Bậc diện tích: so tỉ lệ diện tích mask/ảnh với phân vị p50 và p90 **của chính class đó** trong tập train. Kết quả là 0 (nhỏ), 1 (vừa) hoặc 2 (lớn).
   - Điểm ≤ 2 là Nhẹ, bằng 3 là Trung bình, ≥ 4 là Nặng.
2. **Từng ảnh:** lấy mức cao nhất trong các vùng; tăng 1 mức nếu tổng diện tích hư hỏng hoặc số vùng vượt p90 của dataset.
3. **Nhiều ảnh:** lấy mức cao nhất trong các ảnh. Số vùng được cộng dồn, **không** loại bỏ trùng lặp giữa các góc chụp.

## Hạn chế
- Recall còn thấp với vết trầy, rách nhỏ (Mask mAP50 của Scratch là 0,23). Mô hình bỏ sót khoảng 13% ảnh val.
- Mức độ dựa trên tỉ lệ diện tích trong ảnh, nên phụ thuộc khoảng cách chụp. Trọng số theo loại hư hỏng chưa được chuyên gia kiểm chứng.
- Hệ thống không nhận biết hai ảnh có cùng chụp một vết hư hỏng hay không.
- Hướng cải thiện: dùng YOLOv8s-Seg, tăng `imgsz`, thu thập nhãn mức độ từ chuyên gia để học có giám sát (ordinal classification).
