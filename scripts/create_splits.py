import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List

from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images_by_class(source_root: Path) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = {}
    for class_dir in sorted([p for p in source_root.iterdir() if p.is_dir()]):
        images = [p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        if images:
            mapping[class_dir.name] = images
    if not mapping:
        raise RuntimeError(f"No class folders with images found under {source_root}")
    return mapping


def split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    train_n: int = -1,
    val_n: int = -1,
    test_n: int = -1,
) -> Dict[str, List[int]]:
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)

    # If absolute per-class counts are provided, they take precedence
    if train_n >= 0 or val_n >= 0 or test_n >= 0:
        tn = max(0, train_n)
        vn = max(0, val_n)
        wn = max(0, test_n)
        # Cap to available n
        total = min(n, tn + vn + wn if (tn + vn + wn) > 0 else n)
        tn = min(tn, total)
        vn = min(vn, total - tn)
        wn = min(wn, total - tn - vn)
        train_idx = idxs[:tn]
        val_idx = idxs[tn:tn + vn]
        test_idx = idxs[tn + vn:tn + vn + wn]
        return {"train": train_idx, "val": val_idx, "test": test_idx}

    # Ratio path: allow partial sums; remainder goes to test
    n_train = int(n * max(0.0, train_ratio))
    n_val = int(n * max(0.0, val_ratio))
    used = min(n, n_train + n_val)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:used]
    test_idx = idxs[used:n]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def remove_dir_if_exists(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)


def compress_save(src: Path, dst: Path, *, quality: int = 85, convert_to_jpeg: bool = False, max_side: int = 0) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img_format = (img.format or "JPEG").upper()
        # Optional downscale keeping aspect ratio
        if max_side and max(img.size) > max_side:
            scale = max_side / float(max(img.size))
            new_w = max(1, int(round(img.size[0] * scale)))
            new_h = max(1, int(round(img.size[1] * scale)))
            img = img.resize((new_w, new_h), resample=Image.BILINEAR)

        if convert_to_jpeg or img_format not in {"JPEG", "JPG"}:
            # Convert to JPEG (lossy) for strong size reduction
            img = img.convert("RGB")
            dst = dst.with_suffix(".jpg")
            img.save(dst, format="JPEG", quality=int(quality), optimize=True)
        else:
            # Re-encode JPEG with quality/optimize
            img.save(dst, format="JPEG", quality=int(quality), optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create stratified train/val/test splits from a flat class dataset")
    parser.add_argument("--source", type=Path, default=Path("final_dataset"), help="Source root with class subfolders")
    parser.add_argument("--dest", type=Path, default=Path("final_dataset_splits"), help="Destination root to write splits")
    parser.add_argument("--train", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.1, help="Val ratio")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42)
    # Optional absolute per-class counts (override ratios if provided)
    parser.add_argument("--train-n", type=int, default=-1, help="Absolute images per class for train (overrides ratio)")
    parser.add_argument("--val-n", type=int, default=-1, help="Absolute images per class for val (overrides ratio)")
    parser.add_argument("--test-n", type=int, default=-1, help="Absolute images per class for test (overrides ratio)")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying (ignored if --compress)")
    parser.add_argument("--compress", action="store_true", help="Re-encode images on write to reduce size")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality when --compress (1-100)")
    parser.add_argument("--convert-to-jpeg", action="store_true", help="Convert any format to JPEG when compressing")
    parser.add_argument("--max-side", type=int, default=0, help="Optional max side for downscaling when compressing (0=disable)")
    args = parser.parse_args()

    # Validate inputs
    if args.train_n < 0 and args.val_n < 0 and args.test_n < 0:
        # Ratios mode; allow partial sums and non-negative values
        if args.train < 0 or args.val < 0 or args.test < 0:
            raise ValueError("Ratios must be non-negative")

    # Clean destination if exists (fresh run)
    remove_dir_if_exists(args.dest)

    cls_to_paths = list_images_by_class(args.source)

    for split in ("train", "val", "test"):
        for cls in cls_to_paths.keys():
            ensure_dir(args.dest / split / cls)

    for cls, paths in cls_to_paths.items():
        idxs = split_indices(
            len(paths),
            args.train,
            args.val,
            args.seed,
            train_n=args.train_n,
            val_n=args.val_n,
            test_n=args.test_n,
        )
        for split, indices in idxs.items():
            for i in tqdm(indices, desc=f"{cls}:{split}", leave=False):
                src = paths[i]
                dst = args.dest / split / cls / src.name
                if dst.exists():
                    continue
                if args.compress:
                    compress_save(src, dst, quality=args.quality, convert_to_jpeg=args.convert_to_jpeg, max_side=args.max_side)
                else:
                    if args.move:
                        shutil.move(str(src), str(dst))
                    else:
                        shutil.copy2(str(src), str(dst))

    print(f"Wrote splits to {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



