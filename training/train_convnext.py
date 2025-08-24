import argparse
import sys
from pathlib import Path as _Path
import datetime as dt
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml

# Ensure repo root on sys.path for local imports
_repo_root = _Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from datasets.classification_dataset import ClassificationImageFolder
from models.convnext_classifier import ConvNeXtClassifier
from training.utils import EarlyStopper, compute_classification_metrics, save_json, set_seed
from training.letterbox import LetterboxToSquare


def build_transforms(image_size: int) -> Dict[str, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            LetterboxToSquare(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            LetterboxToSquare(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return {"train": train_tf, "eval": eval_tf}


def create_run_dir(base: Path) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"convnext-run-{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--variant", type=str, default="tiny", choices=["tiny", "small", "base", "large"])
    parser.add_argument("--image_size", type=int, default=896)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cudnn.benchmark = True
    scaler = GradScaler(device="cuda", enabled=(device.type == "cuda"))
    image_size = args.image_size
    tf = build_transforms(image_size)

    classes: List[str] = cfg["data"]["classes"]
    data_root = Path(cfg["data"]["classification_root"]).resolve()
    train_ds = ClassificationImageFolder(data_root, classes, split=cfg["data"]["train_split"], transform=tf["train"])
    val_ds = ClassificationImageFolder(data_root, classes, split=cfg["data"]["val_split"], transform=tf["eval"])

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
        pin_memory=True,
    )

    model = ConvNeXtClassifier(
        num_classes=len(classes),
        variant=args.variant,
        pretrained=bool(cfg["model"]["pretrained"]),
        dropout=float(cfg["model"]["dropout"]),
        freeze_until=None,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]), weight_decay=float(cfg["training"]["weight_decay"]))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["training"]["max_epochs"]))

    run_dir = create_run_dir(Path(args.runs_dir))
    save_json(cfg, run_dir / "config.json")
    writer = SummaryWriter(log_dir=str(run_dir))

    stopper = EarlyStopper(patience=int(cfg["training"]["early_stopping_patience"]))
    best_val_f1 = -1.0

    for epoch in range(int(cfg["training"]["max_epochs"])):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"train[{epoch}]", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * images.size(0)
            pbar.set_postfix({"loss": f"{float(loss.detach()):.4f}"})

        model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        with torch.no_grad():
            pbarv = tqdm(val_loader, desc=f"val[{epoch}]", leave=False)
            for images, labels in pbarv:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                    logits = model(images)
                preds = torch.argmax(logits, dim=1)
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        metrics = compute_classification_metrics(y_true, y_pred)
        epoch_loss = total_loss / len(train_ds)

        log = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            **metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(log)
        # TensorBoard scalars
        writer.add_scalar("loss/train", epoch_loss, epoch)
        writer.add_scalar("metrics/accuracy", metrics["accuracy"], epoch)
        writer.add_scalar("metrics/macro_f1", metrics["macro_f1"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        save_json(log, run_dir / f"epoch-{epoch:03d}.json")

        if metrics["macro_f1"] > best_val_f1:
            best_val_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), run_dir / "best.pt")

        if stopper.step(metrics["macro_f1"]):
            print("Early stopping.")
            break

        scheduler.step()

    torch.save(model.state_dict(), run_dir / "last.pt")
    writer.close()


if __name__ == "__main__":
    main()


