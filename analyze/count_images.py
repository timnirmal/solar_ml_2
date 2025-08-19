import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Set


IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def count_images_in_directory(directory: Path, recursive: bool = True) -> int:
    if not directory.exists() or not directory.is_dir():
        return 0
    if recursive:
        return sum(1 for p in directory.rglob("*") if is_image_file(p))
    return sum(1 for p in directory.iterdir() if is_image_file(p))


def list_immediate_subdirectories(root: Path) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def count_images_per_class(dataset_root: Path, recursive: bool = True) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for subdir in list_immediate_subdirectories(dataset_root):
        counts[subdir.name] = count_images_in_directory(subdir, recursive=recursive)
    return counts


def print_human_readable(counts: Dict[str, int]) -> None:
    if not counts:
        print("No class folders found.")
        return
    name_width = max(len(name) for name in counts.keys())
    total = 0
    for name in sorted(counts.keys()):
        count = counts[name]
        total += count
        print(f"{name.ljust(name_width)} : {count}")
    print("-" * (name_width + 3 + max(5, len(str(total)))))
    print(f"{'TOTAL'.ljust(name_width)} : {total}")


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(description="Count image files per class folder under a dataset root.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "datasets",
        help="Path to dataset root containing class subfolders (default: ../datasets)",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only count images directly inside each class folder (do not recurse)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output counts as JSON instead of human-readable text",
    )
    args = parser.parse_args(list(argv))

    dataset_root: Path = args.root
    recursive = not args.non_recursive

    counts = count_images_per_class(dataset_root, recursive=recursive)

    if args.json:
        print(json.dumps({"root": str(dataset_root), "counts": counts, "total": sum(counts.values())}, indent=2))
    else:
        print(f"Dataset root: {dataset_root}")
        print_human_readable(counts)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))




# {
#   "root": "datasets",
#   "counts": {
#     "bird_drop": 201,
#     "clean": 191,
#     "dusty": 182,
#     "electrical_damage": 90,
#     "physical_damage": 66,
#     "snow_covered": 114
#   },
#   "total": 844
# }