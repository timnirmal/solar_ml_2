import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def load_names_from_yaml(yaml_path: Path) -> List[str]:
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names") or data.get("classes")
    if isinstance(names, dict):
        # sometimes Roboflow exports dict index->name
        # convert to list sorted by numeric key
        names = [v for _, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
    if not isinstance(names, list):
        raise RuntimeError(f"Could not read names list from {yaml_path}")
    return [str(n) for n in names]


def yolo_dirs(root: Path) -> Dict[str, Tuple[Path, Path]]:
    ds: Dict[str, Tuple[Path, Path]] = {}
    for split in ("train", "valid", "val", "test"):
        images = root / split / "images"
        labels = root / split / "labels"
        if images.exists() and labels.exists():
            key = "val" if split == "valid" else split
            ds[key] = (images, labels)
    return ds


def list_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for img in images_dir.iterdir():
        if not img.is_file():
            continue
        stem = img.stem
        lbl = labels_dir / f"{stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def remap_label_file(src_label: Path, dst_label: Path, src_names: List[str], canonical_names: List[str]) -> None:
    # build id map
    name_to_id = {n: i for i, n in enumerate(canonical_names)}
    with src_label.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    out_lines: List[str] = []
    for ln in lines:
        parts = ln.split()
        if not parts:
            continue
        cls_id = int(parts[0])
        src_name = src_names[cls_id]
        if src_name not in name_to_id:
            # skip unknown classes
            continue
        new_id = name_to_id[src_name]
        parts[0] = str(new_id)
        out_lines.append(" ".join(parts))
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    with dst_label.open("w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + ("\n" if out_lines else ""))


def copy_with_prefix(src_img: Path, src_lbl: Path, dst_images: Path, dst_labels: Path, prefix: str,
                     src_names: List[str], canonical_names: List[str]) -> None:
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    new_stem = f"{prefix}_{src_img.stem}"
    dst_img = dst_images / f"{new_stem}{src_img.suffix}"
    dst_lbl = dst_labels / f"{new_stem}.txt"
    shutil.copy2(src_img, dst_img)
    remap_label_file(src_lbl, dst_lbl, src_names, canonical_names)


def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[int]]:
    idxs = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": idxs[:n_train],
        "val": idxs[n_train:n_train + n_val],
        "test": idxs[n_train + n_val:],
    }


def write_merged_yaml(dest_root: Path, canonical_names: List[str]) -> None:
    data = {
        "train": "./train/images",
        "val": "./val/images",
        "test": "./test/images",
        "nc": len(canonical_names),
        "names": canonical_names,
    }
    with (dest_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge two YOLO datasets into one with canonical classes and splits")
    ap.add_argument("--src1", type=Path, default=Path("labled_data/1"))
    ap.add_argument("--src2", type=Path, default=Path("labled_data/2"))
    ap.add_argument("--dest", type=Path, default=Path("labled_data/merged"))
    ap.add_argument("--classes", nargs="*", required=True,
                    help="Canonical class names in order (length = nc)")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    canonical = list(args.classes)
    dest = args.dest
    # Prepare dest folders
    for sp in ("train", "val", "test"):
        (dest / sp / "images").mkdir(parents=True, exist_ok=True)
        (dest / sp / "labels").mkdir(parents=True, exist_ok=True)

    # Load names
    src1_names = load_names_from_yaml(args.src1 / "data.yaml")
    src2_names = load_names_from_yaml(args.src2 / "data.yaml")

    # Dataset 1: copy its existing splits
    d1 = yolo_dirs(args.src1)
    for sp, (imgs_dir, lbls_dir) in d1.items():
        pairs = list_pairs(imgs_dir, lbls_dir)
        for img, lbl in pairs:
            copy_with_prefix(img, lbl, dest / sp / "images", dest / sp / "labels", "d1",
                             src1_names, canonical)

    # Dataset 2: only train split; redistribute per ratios
    d2 = yolo_dirs(args.src2)
    if "train" in d2:
        pairs2 = list_pairs(*d2["train"])
        idxs = split_indices(len(pairs2), args.train, args.val, args.seed)
        for sp, ids in idxs.items():
            for i in ids:
                img, lbl = pairs2[i]
                copy_with_prefix(img, lbl, dest / sp / "images", dest / sp / "labels", "d2",
                                 src2_names, canonical)

    write_merged_yaml(dest, canonical)
    print(f"Merged dataset written to: {dest}")


if __name__ == "__main__":
    main()


