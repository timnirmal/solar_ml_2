import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List

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


def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[int]]:
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train:n_train + n_val]
    test_idx = idxs[n_train + n_val:]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create stratified train/val/test splits from a flat class dataset")
    parser.add_argument("--source", type=Path, default=Path("final_dataset"), help="Source root with class subfolders")
    parser.add_argument("--dest", type=Path, default=Path("final_dataset_splits"), help="Destination root to write splits")
    parser.add_argument("--train", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.1, help="Val ratio")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    args = parser.parse_args()

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    cls_to_paths = list_images_by_class(args.source)

    for split in ("train", "val", "test"):
        for cls in cls_to_paths.keys():
            ensure_dir(args.dest / split / cls)

    for cls, paths in cls_to_paths.items():
        idxs = split_indices(len(paths), args.train, args.val, args.seed)
        for split, indices in idxs.items():
            for i in tqdm(indices, desc=f"{cls}:{split}", leave=False):
                src = paths[i]
                dst = args.dest / split / cls / src.name
                if dst.exists():
                    continue
                if args.move:
                    shutil.move(str(src), str(dst))
                else:
                    shutil.copy2(str(src), str(dst))

    print(f"Wrote splits to {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



