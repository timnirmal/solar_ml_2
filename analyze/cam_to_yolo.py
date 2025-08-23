import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


def load_convnext_for_cam(variant: str = "tiny", num_classes: int = 6, device: str = "cpu"):
    if variant == "tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        feat_dim = 768
    elif variant == "small":
        weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        model = models.convnext_small(weights=weights)
        feat_dim = 768
    elif variant == "base":
        weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        model = models.convnext_base(weights=weights)
        feat_dim = 1024
    else:
        weights = models.ConvNeXt_Large_Weights.IMAGENET1K_V1
        model = models.convnext_large(weights=weights)
        feat_dim = 1536

    # Replace classifier to match dataset classes; keep a handle for CAM
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
    model.eval().to(device)
    return model, feat_dim


def preprocess(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


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
    parser = argparse.ArgumentParser(description="Generate pseudo-label boxes from ConvNeXt CAM for YOLO training")
    parser.add_argument("--images", type=Path, required=True, help="Path to images folder (flat or nested)")
    parser.add_argument("--labels-out", type=Path, required=True, help="Folder to write YOLO txt labels")
    parser.add_argument("--variant", type=str, default="tiny", choices=["tiny", "small", "base", "large"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--class-idx", type=int, required=True, help="Class index to extract CAM for (0..N-1)")
    parser.add_argument("--thresh", type=float, default=0.4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, feat_dim = load_convnext_for_cam(args.variant, num_classes=args.num_classes, device=device)
    tf = preprocess(args.image_size)

    # Hook to grab last feature map
    target_feats: torch.Tensor = None  # type: ignore

    def hook_fn(module, inp, out):
        nonlocal target_feats
        target_feats = out.detach()

    last_block = list(model.features.children())[-1]
    handle = last_block.register_forward_hook(hook_fn)

    try:
        img_paths: List[Path] = [p for p in args.images.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        for p in img_paths:
            img = Image.open(p).convert("RGB")
            orig_w, orig_h = img.size
            x = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                pred_idx = int(torch.argmax(logits, dim=1).item())
            # CAM using classifier weights for predicted or specified class
            class_idx = args.class_idx if args.class_idx >= 0 else pred_idx
            weights = model.classifier[2].weight.detach()  # [num_classes, C]
            cam = cam_heatmap(target_feats, weights, class_idx)
            x1, y1, x2, y2 = cam_to_bbox(cam, orig_w, orig_h, thresh=args.thresh)
            yolo_box = bbox_to_yolo(x1, y1, x2, y2, orig_w, orig_h)
            out_txt = args.labels_out / (p.stem + ".txt")
            write_yolo_label(out_txt, class_idx, yolo_box)
    finally:
        handle.remove()


if __name__ == "__main__":
    main()



