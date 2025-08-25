import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def find_latest_weights(runs_root: Path) -> Optional[Path]:
    if not runs_root.exists():
        return None
    candidates = [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("convnext-run-")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for rd in candidates:
        best = rd / "best.pt"
        last = rd / "last.pt"
        if best.exists():
            return best
        if last.exists():
            return last
    return None


def load_convnext(variant: str, num_classes: int, device: str, weights: Optional[Path]) -> ConvNeXtClassifier:
    model = ConvNeXtClassifier(num_classes=num_classes, variant=variant, pretrained=True, dropout=0.0, freeze_until=None)
    if weights and weights.exists():
        state = torch.load(str(weights), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


def preprocess(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def register_cam_hooks(model: torch.nn.Module, layer: torch.nn.Module):
    activations = {}
    gradients = {}

    def fwd_hook(_, __, output):
        activations["value"] = output.detach()

    def bwd_hook(_, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    handle_f = layer.register_forward_hook(fwd_hook)
    handle_b = layer.register_full_backward_hook(bwd_hook)
    return activations, gradients, handle_f, handle_b


def grad_cam(acts: torch.Tensor, grads: torch.Tensor) -> np.ndarray:
    # acts, grads: [B,C,H,W]
    weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP over spatial
    cam = (weights * acts).sum(dim=1, keepdim=True)  # [B,1,H,W]
    cam = F.relu(cam)
    cam = cam.squeeze(0).squeeze(0).cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def percentile_threshold(cam: np.ndarray, pct: float) -> np.ndarray:
    t = np.percentile(cam, pct * 100.0)
    return (cam >= t).astype(np.uint8)


def morph_open(mask: torch.Tensor, k: int) -> torch.Tensor:
    # mask: [1,1,H,W] uint8 {0,1}
    # erosion then dilation via max_pool on inverted
    inv = 1 - mask
    eroded = 1 - F.max_pool2d(inv.float(), kernel_size=k, stride=1, padding=k // 2)
    dilated = F.max_pool2d(eroded, kernel_size=k, stride=1, padding=k // 2)
    return (dilated > 0.5).to(mask.dtype)


def morph_close(mask: torch.Tensor, k: int) -> torch.Tensor:
    # dilation then erosion
    dilated = F.max_pool2d(mask.float(), kernel_size=k, stride=1, padding=k // 2)
    inv = 1 - dilated
    eroded = 1 - F.max_pool2d(inv, kernel_size=k, stride=1, padding=k // 2)
    return (eroded > 0.5).to(mask.dtype)


def largest_bbox(mask_np: np.ndarray) -> Tuple[int, int, int, int]:
    # Fallback: bbox of all positives (fast, robust)
    ys, xs = np.where(mask_np > 0)
    if ys.size == 0:
        return 0, 0, mask_np.shape[1] - 1, mask_np.shape[0] - 1
    return xs.min(), ys.min(), xs.max(), ys.max()


def save_composite(img: Image.Image, cam: np.ndarray, bbox: Tuple[int, int, int, int], out_path: Path) -> None:
    W, H = img.size
    cam_uint8 = (np.clip(cam, 0.0, 1.0) * 255).astype(np.uint8)
    cam_resized = np.array(Image.fromarray(cam_uint8).resize((W, H), resample=Image.BILINEAR)) / 255.0
    cam_rgba = (cm.jet(cam_resized) * 255).astype(np.uint8)
    cam_color = Image.fromarray(cam_rgba[:, :, :3])
    overlay_cam = Image.blend(img.convert("RGB"), cam_color.convert("RGB"), alpha=0.45)
    boxed = img.copy()
    draw = ImageDraw.Draw(boxed)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)

    comp = Image.new("RGB", (W * 2, H * 2), (0, 0, 0))
    comp.paste(img, (0, 0))
    comp.paste(cam_color, (W, 0))
    comp.paste(overlay_cam, (0, H))
    comp.paste(boxed, (W, H))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comp.save(out_path, quality=90)


def main() -> int:
    p = argparse.ArgumentParser(description="Refined CAM -> YOLO labels with composites")
    p.add_argument("--root", type=Path, required=True, help="Dataset root with splits/classes (e.g., final_dataset_splits)")
    p.add_argument("--splits", nargs="*", default=["train", "val", "test"])  # type: ignore[arg-type]
    p.add_argument("--labels-root", type=Path, required=True)
    p.add_argument("--variant", type=str, default="tiny", choices=["tiny", "small", "base", "large"])
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--classes", nargs="*", default=None)
    p.add_argument("--use-pred", action="store_true")
    p.add_argument("--percentile", type=float, default=0.9, help="CAM threshold percentile (0-1)")
    p.add_argument("--morph", type=str, default="both", choices=["none", "open", "close", "both"])
    p.add_argument("--morph-k", type=int, default=3)
    p.add_argument("--hook-index", type=int, default=-2, help="Which features layer index to hook (negative indexes allowed)")
    p.add_argument("--margin", type=float, default=0.03)
    p.add_argument("--overlay-root", type=Path, default=Path("cam_overlays"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Infer classes if not provided
    if args.classes is None:
        for sp in args.splits:
            d = args.root / sp
            if d.exists():
                args.classes = sorted([q.name for q in d.iterdir() if q.is_dir()])
                break
    if not args.classes:
        raise RuntimeError("No classes found; provide --classes or valid splits")

    # Weights auto-pick
    weights = args.weights
    if weights is None:
        auto = find_latest_weights(_repo_root / "runs")
        if auto:
            weights = auto
            print(f"[info] Using latest weights: {weights}")
        else:
            print("[warn] No weights found; using ImageNet-pretrained ConvNeXt")

    model = load_convnext(args.variant, num_classes=len(args.classes), device=device, weights=weights)
    tf = preprocess(args.image_size)

    # Pick layer to hook by index
    feats_modules = list(model.features.children())
    hook_idx = args.hook_index if args.hook_index < len(feats_modules) else -1
    layer = feats_modules[hook_idx]
    acts, grads, h_f, h_b = register_cam_hooks(model, layer)

    # Run dir
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.overlay_root / f"cam-refined-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    processed: Dict[str, Dict[str, int]] = {}

    try:
        name_to_idx = {n: i for i, n in enumerate(args.classes)}
        for sp in args.splits:
            sp_dir = args.root / sp
            if not sp_dir.exists():
                continue
            processed[sp] = {}
            for cls_dir in sorted([p for p in sp_dir.iterdir() if p.is_dir()]):
                cname = cls_dir.name
                if cname not in name_to_idx:
                    continue
                cidx_folder = name_to_idx[cname]
                imgs = [p for p in cls_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}]
                processed[sp][cname] = 0
                for ip in imgs:
                    img = Image.open(ip).convert("RGB")
                    W, H = img.size
                    x = tf(img).unsqueeze(0).to(device)
                    # Forward WITH grads so CAM can backprop
                    logits = model(x)
                    pred = int(torch.argmax(logits, dim=1).item())

                    use_idx = pred if args.use_pred else cidx_folder

                    # Backprop to get gradients
                    model.zero_grad(set_to_none=True)
                    one_hot = torch.zeros_like(logits)
                    one_hot[0, use_idx] = 1.0
                    logits.backward(gradient=one_hot, retain_graph=False)

                    cam = grad_cam(acts["value"], grads["value"])  # [0..1]

                    # Threshold by percentile and clean with morphology
                    mask_np = percentile_threshold(cam, args.percentile)
                    mask_t = torch.from_numpy(mask_np).view(1, 1, *mask_np.shape)
                    if args.morph in ("open", "both"):
                        mask_t = morph_open(mask_t, args.morph_k)
                    if args.morph in ("close", "both"):
                        mask_t = morph_close(mask_t, args.morph_k)
                    mask_np = mask_t.squeeze().numpy().astype(np.uint8)

                    # BBox from cleaned mask
                    x1, y1, x2, y2 = largest_bbox(mask_np)
                    if args.margin > 0:
                        pad = int(round(min(W, H) * args.margin))
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(W - 1, x2 + pad)
                        y2 = min(H - 1, y2 + pad)

                    # Save YOLO label
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    bw = (x2 - x1)
                    bh = (y2 - y1)
                    out_txt = args.labels_root / sp / cname / (ip.stem + ".txt")
                    out_txt.parent.mkdir(parents=True, exist_ok=True)
                    out_txt.write_text(f"{use_idx} {cx/W:.6f} {cy/H:.6f} {bw/W:.6f} {bh/H:.6f}\n")

                    # Save composite
                    out_img = run_dir / sp / cname / (ip.stem + "_cam_composite.jpg")
                    save_composite(img, cam, (x1, y1, x2, y2), out_img)
                    processed[sp][cname] += 1
    finally:
        h_f.remove()
        h_b.remove()
        manifest = {
            "run_dir": str(run_dir),
            "args": {
                "root": str(args.root),
                "splits": args.splits,
                "labels_root": str(args.labels_root),
                "variant": args.variant,
                "weights": str(weights) if weights else None,
                "image_size": args.image_size,
                "classes": args.classes,
                "use_pred": bool(args.use_pred),
                "percentile": float(args.percentile),
                "morph": args.morph,
                "morph_k": int(args.morph_k),
                "hook_index": int(args.hook_index),
                "margin": float(args.margin),
                "timestamp": ts,
            },
            "counts": processed,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"Refined CAM written to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


