from typing import Tuple

from PIL import Image


class LetterboxToSquare:
    """Resize image preserving aspect ratio and pad to square (like YOLO letterbox).

    - target_size: output square size (e.g., 640, 896, 1024)
    - color: RGB padding color (default 114 like YOLO)
    """

    def __init__(self, target_size: int, color: Tuple[int, int, int] = (114, 114, 114)) -> None:
        self.target_size = int(target_size)
        self.color = color

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == 0 or h == 0:
            return img
        scale = self.target_size / float(max(w, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img_resized = img.resize((new_w, new_h), resample=Image.BILINEAR)

        out = Image.new("RGB", (self.target_size, self.target_size), self.color)
        pad_x = (self.target_size - new_w) // 2
        pad_y = (self.target_size - new_h) // 2
        out.paste(img_resized, (pad_x, pad_y))
        return out



