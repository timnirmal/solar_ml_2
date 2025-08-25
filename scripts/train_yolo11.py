import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    p = argparse.ArgumentParser(description="Train YOLOv11 on merged dataset")
    p.add_argument("--data", type=Path, default=Path("labled_data/merged/data.yaml"))
    p.add_argument("--model", type=str, default="yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--name", type=str, default="yolo11-merged")
    p.add_argument("--device", type=str, default="0")
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        name=args.name,
        device=args.device,
        project="runs/yolo11",
        optimizer="adamw",
        patience=20,
        mosaic=0.5,
        mixup=0.1,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
        fliplr=0.5, flipud=0.0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


