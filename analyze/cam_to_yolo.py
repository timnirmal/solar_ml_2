import argparse
import sys
import json
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from matplotlib import cm
from torchvision import transforms

# Ensure local imports when run as a script
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from models.convnext_classifier import ConvNeXtClassifier


def load_convnext_for_cam(variant: str, num_classes: int, device: str, weights_path: Path | None):
    model = ConvNeXtClassifier(num_classes=num_classes, variant=variant, pretrained=True, dropout=0.0, freeze_until=None)
    if weights_path and weights_path.exists():
        state = torch.load(str(weights_path), map_location="cpu")
        # Support loading either full state_dict or under 'state_dict'
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    model.eval().to(device)
    # Feature dimension equals embed dim depending on variant
    feat_dim = model.head[-1].in_features  # type: ignore[attr-defined]
    return model, feat_dim


def preprocess(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def find_latest_weights(runs_root: Path) -> Optional[Path]:
    if not runs_root.exists():
        return None
    candidates = [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("convnext-run-")]
    if not candidates:
        return None
    # Sort by modified time
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in candidates:
        best = run_dir / "best.pt"
        last = run_dir / "last.pt"
        if best.exists():
            return best
        if last.exists():
            return last
    return None


def cam_heatmap(features: torch.Tensor, weights: torch.Tensor, class_idx: int) -> np.ndarray:
    # features: [B,C,H,W], weights: [num_classes, C]
    with torch.no_grad():
        w = weights[class_idx].view(1, -1, 1, 1)  # [1,C,1,1]
        cam = (features * w).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = F.relu(cam)
        cam = cam.squeeze(0).squeeze(0).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def cam_to_bbox(cam: np.ndarray, orig_w: int, orig_h: int, thresh: float = 0.4) -> Tuple[int, int, int, int]:
    mask = (cam >= thresh).astype(np.uint8)
    if mask.sum() == 0:
        return 0, 0, orig_w - 1, orig_h - 1
    ys, xs = np.where(mask > 0)
    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max(), ys.max()
    # scale back to original size
    h_cam, w_cam = cam.shape
    x1 = int(x1 * (orig_w / w_cam))
    x2 = int(x2 * (orig_w / w_cam))
    y1 = int(y1 * (orig_h / h_cam))
    y2 = int(y2 * (orig_h / h_cam))
    return x1, y1, x2, y2


def bbox_to_yolo(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[float, float, float, float]:
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1)
    bh = (y2 - y1)
    return cx / w, cy / h, bw / w, bh / h


def write_yolo_label(out_path: Path, class_idx: int, box: Tuple[float, float, float, float]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cx, cy, bw, bh = box
    out_path.write_text(f"{class_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate CAM pseudo-label boxes from ConvNeXt for YOLO training")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root with splits and class folders (e.g., final_dataset_splits)")
    parser.add_argument("--splits", type=str, nargs="*", default=["train", "val", "test"], help="Splits to process")
    parser.add_argument("--labels-root", type=Path, required=True, help="Root to write YOLO labels (mirrors split/class structure)")
    parser.add_argument("--overlay-root", type=Path, default=Path("cam_overlays"), help="Base dir to write overlay images; a new run folder will be created inside")
    parser.add_argument("--variant", type=str, default="tiny", choices=["tiny", "small", "base", "large"])
    parser.add_argument("--weights", type=Path, default=None, help="Path to trained ConvNeXt checkpoint (best.pt)")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--classes", type=str, nargs="*", default=None, help="Class names (if not provided, inferred from dirs)")
    parser.add_argument("--use-pred", action="store_true", help="Use predicted class for CAM (otherwise use folder class)")
    parser.add_argument("--thresh", type=float, default=0.4)
    parser.add_argument("--margin", type=float, default=0.05, help="Relative margin to expand bbox (0-0.5)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Determine classes by scanning first available split if not provided
    if args.classes is None:
        for sp in args.splits:
            split_dir = args.root / sp
            if split_dir.exists():
                args.classes = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
                break
    if not args.classes:
        raise RuntimeError("Could not determine class names. Provide --classes or ensure split dirs exist.")

    weights_path = args.weights
    if weights_path is None:
        # Auto-pick latest run weights
        runs_root = _repo_root / "runs"
        auto_w = find_latest_weights(runs_root)
        if auto_w is not None:
            weights_path = auto_w
            print(f"[info] Using latest weights: {weights_path}")
        else:
            print("[warn] No weights provided and none found in runs/. Using ImageNet-pretrained ConvNeXt.")
    model, feat_dim = load_convnext_for_cam(args.variant, num_classes=len(args.classes), device=device, weights_path=weights_path)
    tf = preprocess(args.image_size)

    # Hook to grab last feature map
    target_feats: torch.Tensor = None  # type: ignore

    def hook_fn(module, inp, out):
        nonlocal target_feats
        target_feats = out.detach()

    last_block = list(model.features.children())[-1]
    handle = last_block.register_forward_hook(hook_fn)

    # Prepare overlay run dir
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    overlay_run: Path = args.overlay_root / f"cam-run-{ts}"
    overlay_run.mkdir(parents=True, exist_ok=True)

    # Stats for log
    processed_counts: Dict[str, Dict[str, int]] = {}

    try:
        classes = list(args.classes)
        name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(classes)}
        for sp in args.splits:
            split_dir = args.root / sp
            if not split_dir.exists():
                continue
            processed_counts[sp] = {}
            for cls_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
                class_name = cls_dir.name
                if class_name not in name_to_idx:
                    continue
                class_idx_folder = name_to_idx[class_name]
                img_paths: List[Path] = [p for p in cls_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}]
                processed_counts[sp][class_name] = 0
                for p in img_paths:
                    img = Image.open(p).convert("RGB")
                    orig_w, orig_h = img.size
                    x = tf(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(x)
                        pred_idx = int(torch.argmax(logits, dim=1).item())
                    use_idx = pred_idx if args.use_pred else class_idx_folder
                    # In our ConvNeXtClassifier, linear weights are model.head[-1]
                    weights = model.head[-1].weight.detach()
                    cam = cam_heatmap(target_feats, weights, use_idx)
                    x1, y1, x2, y2 = cam_to_bbox(cam, orig_w, orig_h, thresh=args.thresh)
                    # Expand by margin
                    if args.margin > 0:
                        dw = int(round(min(orig_w, orig_h) * args.margin))
                        x1 = max(0, x1 - dw)
                        y1 = max(0, y1 - dw)
                        x2 = min(orig_w - 1, x2 + dw)
                        y2 = min(orig_h - 1, y2 + dw)
                    yolo_box = bbox_to_yolo(x1, y1, x2, y2, orig_w, orig_h)
                    out_txt = args.labels_root / sp / class_name / (p.stem + ".txt")
                    write_yolo_label(out_txt, use_idx, yolo_box)

                    # Build visuals: colored CAM, CAM overlay on image, bbox image, composite 2x2
                    # Resize CAM to original size [0..1]
                    cam_uint8 = (np.clip(cam, 0.0, 1.0) * 255).astype(np.uint8)
                    cam_resized = np.array(Image.fromarray(cam_uint8).resize((orig_w, orig_h), resample=Image.BILINEAR)) / 255.0

                    # Colored CAM (JET colormap)
                    cam_rgba = (cm.jet(cam_resized) * 255).astype(np.uint8)  # RGBA
                    cam_color = Image.fromarray(cam_rgba[:, :, :3])

                    # CAM overlay on original image
                    overlay_cam = Image.blend(img.convert("RGB"), cam_color.convert("RGB"), alpha=0.45)

                    # Image with bbox
                    boxed = img.copy()
                    draw = ImageDraw.Draw(boxed)
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

                    # Composite (2x2): [original | cam colored; cam overlay | bbox]
                    W, H = img.size
                    composite = Image.new("RGB", (W * 2, H * 2), (0, 0, 0))
                    composite.paste(img, (0, 0))
                    composite.paste(cam_color, (W, 0))
                    composite.paste(overlay_cam, (0, H))
                    composite.paste(boxed, (W, H))

                    # Save composite under run dir
                    ov_dir = overlay_run / sp / class_name
                    ov_dir.mkdir(parents=True, exist_ok=True)
                    composite.save(ov_dir / (p.stem + "_cam_composite.jpg"), quality=90)

                    processed_counts[sp][class_name] += 1
    finally:
        handle.remove()
        # Write manifest/log
        manifest = {
            "run_dir": str(overlay_run),
            "args": {
                "root": str(args.root),
                "splits": args.splits,
                "labels_root": str(args.labels_root),
                "overlay_root": str(args.overlay_root),
                "variant": args.variant,
                "weights": str(args.weights) if args.weights else None,
                "image_size": args.image_size,
                "classes": classes,
                "use_pred": bool(args.use_pred),
                "thresh": float(args.thresh),
                "margin": float(args.margin),
                "timestamp": ts,
            },
            "counts": processed_counts,
        }
        (overlay_run / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"CAM overlays written to: {overlay_run}")


if __name__ == "__main__":
    main()



