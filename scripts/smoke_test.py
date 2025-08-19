import datetime as dt
from pathlib import Path

import torch

from models.inception_classifier import InceptionV3SEClassifier


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionV3SEClassifier(num_classes=6, pretrained=False).to(device)
    dummy = torch.randn(2, 3, 299, 299, device=device)
    with torch.no_grad():
        logits = model(dummy)
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / f"run-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "smoke.txt").write_text(str(logits.shape), encoding="utf-8")
    print(f"Smoke test ok, logits shape: {logits.shape}. Wrote runs/{run_dir.name}/smoke.txt")


if __name__ == "__main__":
    main()



