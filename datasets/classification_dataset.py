from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


class ClassificationImageFolder(Dataset):
    def __init__(
        self,
        root: Path,
        class_names: List[str],
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.class_names = class_names
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        split_dir = self.root / split
        for idx, class_name in enumerate(self.class_names):
            cls_dir = split_dir / class_name
            if not cls_dir.exists():
                # skip silently; allows partial availability
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
                for p in cls_dir.glob(ext):
                    self.samples.append((p, idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {split_dir}. Expected class subdirs: {self.class_names}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label



