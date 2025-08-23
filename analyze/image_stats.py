import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


@dataclass
class ImageRecord:
    path: Path
    clazz: str
    width: int
    height: int
    area: int
    aspect_ratio: float
    format: Optional[str]
    mode: Optional[str]
    size_bytes: Optional[int]


def detect_default_root() -> Path:
    here = Path(__file__).resolve().parent.parent
    final = here / "final_dataset"
    if final.exists():
        return final
    datasets = here / "datasets"
    return datasets if datasets.exists() else here


def find_class_dirs(root: Path, splits: List[str]) -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = {}
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            # If no split dirs found, fallback to flat class layout directly under root
            # i.e., root/<class>/*.*
            return {"": [p for p in root.iterdir() if p.is_dir()]}
        class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
        result[split] = class_dirs
    return result


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(root: Path, splits: List[str]) -> List[ImageRecord]:
    split_to_dirs = find_class_dirs(root, splits)
    # If key "" exists, flat layout; else split layout
    flat_layout = "" in split_to_dirs
    records: List[ImageRecord] = []

    if flat_layout:
        for class_dir in split_to_dirs[""]:
            clazz = class_dir.name
            paths = [p for p in class_dir.rglob("*") if p.is_file() and is_image_file(p)]
            for p in tqdm(paths, desc=f"scan:{clazz}", unit="img", leave=False):
                rec = read_image_record(p, clazz)
                if rec:
                    records.append(rec)
    else:
        for split, class_dirs in split_to_dirs.items():
            for class_dir in class_dirs:
                clazz = class_dir.name
                paths = [p for p in class_dir.rglob("*") if p.is_file() and is_image_file(p)]
                for p in tqdm(paths, desc=f"scan:{split}/{clazz}", unit="img", leave=False):
                    rec = read_image_record(p, clazz)
                    if rec:
                        records.append(rec)
    return records


def read_image_record(path: Path, clazz: str) -> Optional[ImageRecord]:
    try:
        size_bytes = path.stat().st_size if path.exists() else None
        with Image.open(path) as img:
            width, height = img.size
            fmt = img.format
            mode = img.mode
        area = int(width) * int(height)
        ratio = float(width) / float(height) if height > 0 else float("nan")
        return ImageRecord(path=path, clazz=clazz, width=width, height=height, area=area, aspect_ratio=ratio, format=fmt, mode=mode, size_bytes=size_bytes)
    except (UnidentifiedImageError, OSError):
        return None


def summarize_numeric(records: List[ImageRecord]) -> Dict[str, Dict[str, float]]:
    if not records:
        return {}
    # overall
    as_np = {
        "width": np.array([r.width for r in records], dtype=np.int64),
        "height": np.array([r.height for r in records], dtype=np.int64),
        "area": np.array([r.area for r in records], dtype=np.int64),
        "ratio": np.array([r.aspect_ratio for r in records], dtype=np.float64),
    }

    def stats(arr: np.ndarray) -> Dict[str, float]:
        return {
            "count": float(arr.size),
            "min": float(np.nanmin(arr)),
            "p5": float(np.nanpercentile(arr, 5)),
            "p50": float(np.nanmedian(arr)),
            "mean": float(np.nanmean(arr)),
            "p95": float(np.nanpercentile(arr, 95)),
            "max": float(np.nanmax(arr)),
            "std": float(np.nanstd(arr)),
        }

    out: Dict[str, Dict[str, float]] = {
        "width": stats(as_np["width"]),
        "height": stats(as_np["height"]),
        "area": stats(as_np["area"]),
        "ratio": stats(as_np["ratio"]),
    }
    return out


def summarize_by_class(records: List[ImageRecord]) -> Dict[str, Dict[str, Dict[str, float]]]:
    by_class: Dict[str, List[ImageRecord]] = defaultdict(list)
    for r in records:
        by_class[r.clazz].append(r)
    per: Dict[str, Dict[str, Dict[str, float]]] = {}
    for clazz, lst in by_class.items():
        per[clazz] = summarize_numeric(lst)
    return per


def top_size_combinations(records: List[ImageRecord], top_k: int = 15) -> List[Tuple[Tuple[int, int], int]]:
    counter: Counter[Tuple[int, int]] = Counter()
    for r in records:
        counter[(r.width, r.height)] += 1
    return counter.most_common(top_k)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_histogram(data: np.ndarray, title: str, xlabel: str, out_path: Path, bins: int = 60) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data[~np.isnan(data)], bins=bins, color="#377eb8", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter_wh(widths: np.ndarray, heights: np.ndarray, out_path: Path, sample: int = 5000) -> None:
    import matplotlib.pyplot as plt

    w = widths
    h = heights
    if w.size > sample:
        idx = np.random.choice(w.size, size=sample, replace=False)
        w = w[idx]
        h = h[idx]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(w, h, s=4, alpha=0.35, c="#4daf4a")
    ax.set_title("Width vs Height (sampled)")
    ax.set_xlabel("width (px)")
    ax.set_ylabel("height (px)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bar_top_sizes(pairs_and_counts: List[Tuple[Tuple[int, int], int]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [f"{w}x{h}" for (w, h), _ in pairs_and_counts]
    counts = [c for _, c in pairs_and_counts]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, counts, color="#984ea3", alpha=0.9)
    ax.set_title("Top image size combinations")
    ax.set_xlabel("size (width x height)")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(description="Analyze image sizes and aspect ratios, output stats and plots.")
    parser.add_argument("--root", type=Path, default=detect_default_root(), help="Dataset root (defaults to final_dataset or datasets)")
    parser.add_argument("--splits", type=str, nargs="*", default=["train", "val"], help="Split folder names to scan")
    parser.add_argument("--outdir", type=Path, default=Path("runs") / "analysis", help="Output directory for stats and plots")
    parser.add_argument("--topk", type=int, default=15, help="Top-K size combinations to report")
    args = parser.parse_args(list(argv))

    t0 = time.time()
    ensure_dir(args.outdir)

    records = collect_images(args.root, args.splits)
    if not records:
        print(f"No images found under {args.root}")
        return 2

    overall = summarize_numeric(records)
    per_class = summarize_by_class(records)
    top_sizes = top_size_combinations(records, top_k=args.topk)

    # Save numeric outputs
    (args.outdir / "summary_overall.json").write_text(json.dumps(overall, indent=2))
    (args.outdir / "summary_per_class.json").write_text(json.dumps(per_class, indent=2))
    (args.outdir / "top_sizes.json").write_text(json.dumps({"top": [[list(pair), count] for pair, count in top_sizes]}, indent=2))

    # Plots
    widths = np.array([r.width for r in records], dtype=np.int64)
    heights = np.array([r.height for r in records], dtype=np.int64)
    ratios = np.array([r.aspect_ratio for r in records], dtype=np.float64)

    plot_histogram(widths.astype(float), "Width distribution", "width (px)", args.outdir / "hist_width.png")
    plot_histogram(heights.astype(float), "Height distribution", "height (px)", args.outdir / "hist_height.png")
    plot_histogram(ratios, "Aspect ratio distribution (W/H)", "ratio", args.outdir / "hist_ratio.png")
    plot_scatter_wh(widths, heights, args.outdir / "scatter_wh.png")
    plot_bar_top_sizes(top_sizes, args.outdir / "bar_top_sizes.png")

    print(f"Wrote stats and plots to {args.outdir} in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))



