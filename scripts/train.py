"""
Giai doan 4 - Fine-tune YOLOv8-Seg tren dataset VehiDE.

Vi du:
    # Train moi (mac dinh yolov8n-seg, 100 epoch, batch 8)
    python scripts/train.py

    # Doi model / batch
    python scripts/train.py --model yolov8s-seg.pt --batch 4

    # Chay thu nhanh: 1 epoch tren 10% du lieu
    python scripts/train.py --epochs 1 --fraction 0.1 --name smoke_test

    # Tiep tuc training bi gian doan
    python scripts/train.py --resume models/trained/yolov8n_seg/weights/last.pt

    # Bat buoc train bang CPU (phai chi dinh ro, script khong tu chuyen)
    python scripts/train.py --device cpu

Ket qua luu tai models/trained/<name>/ :
    weights/best.pt, weights/last.pt, results.csv (log tung epoch), args.yaml, cac bieu do.
Neu thu muc <name> da ton tai, Ultralytics tu tao <name>2, <name>3... -> khong ghi de model cu.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = PROJECT_ROOT / "dataset" / "data.yaml"
PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"
TRAINED_DIR = PROJECT_ROOT / "models" / "trained"


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLOv8-Seg tren VehiDE")
    p.add_argument("--model", default="yolov8n-seg.pt", help="yolov8n-seg.pt | yolov8s-seg.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8, help="giam xuong 4 neu thieu VRAM")
    p.add_argument("--patience", type=int, default=20, help="early stopping")
    p.add_argument("--workers", type=int, default=2, help="Windows: nen de 0-4")
    p.add_argument("--device", default="auto", help="auto | 0 | cpu")
    p.add_argument("--fraction", type=float, default=1.0, help="ti le du lieu train (de chay thu)")
    p.add_argument("--name", default=None, help="ten thu muc ket qua")
    p.add_argument("--resume", default=None, help="duong dan last.pt de train tiep")
    p.add_argument("--skip-check", action="store_true", help="bo qua validate_dataset.py")
    return p.parse_args()


def choose_device(requested):
    """Chon thiet bi. KHONG tu dong chuyen sang CPU khi nguoi dung chua dong y."""
    if requested == "cpu":
        print("[!] Ban da chon train bang CPU -> se RAT CHAM.")
        return "cpu"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU: {name} | VRAM trong: {free / 1e9:.1f} / {total / 1e9:.1f} GB")
        return 0 if requested == "auto" else requested
    print("[LOI] Khong tim thay GPU CUDA.")
    print("  - Kiem tra driver NVIDIA (nvidia-smi) va ban PyTorch co CUDA (torch.version.cuda).")
    print("  - Neu muon train bang CPU, chay lai voi: --device cpu")
    sys.exit(1)


def get_pretrained(model_name):
    """Tai pretrained weights vao models/pretrained/ neu chua co."""
    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRETRAINED_DIR / model_name
    if not path.exists():
        print(f"Tai pretrained weights: {model_name} ...")
        attempt_download_asset(str(path))
    return path


def check_dataset():
    """Chay validate_dataset.py; dung lai neu dataset loi."""
    print("Kiem tra dataset truoc khi train ...")
    r = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "validate_dataset.py")])
    if r.returncode != 0:
        print("[LOI] Dataset khong hop le -> dung training. Xem output o tren.")
        sys.exit(1)


def main():
    args = parse_args()
    device = choose_device(args.device)

    # --- Tiep tuc training bi gian doan ---
    if args.resume:
        print(f"Tiep tuc training tu: {args.resume}")
        model = YOLO(args.resume)
        model.train(resume=True, device=device)
        return

    if not args.skip_check:
        check_dataset()

    weights = get_pretrained(args.model)
    name = args.name or Path(args.model).stem.replace("-", "_")  # vd: yolov8n_seg
    model = YOLO(str(weights))

    start = time.time()
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=device,
        fraction=args.fraction,
        project=str(TRAINED_DIR),
        name=name,
        exist_ok=False,  # khong ghi de: neu trung ten -> tu them hau to so
        seed=0,
        plots=True,
    )
    hours = (time.time() - start) / 3600
    save_dir = Path(model.trainer.save_dir)
    print(f"\nHoan thanh sau {hours:.2f} gio. Ket qua: {save_dir}")
    print(f"  best.pt : {save_dir / 'weights' / 'best.pt'}")
    print(f"  last.pt : {save_dir / 'weights' / 'last.pt'}")
    print(f"  log     : {save_dir / 'results.csv'}")


if __name__ == "__main__":  # bat buoc tren Windows khi dung workers > 0
    main()
