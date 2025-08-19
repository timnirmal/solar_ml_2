import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    assert y_true_arr.shape == y_pred_arr.shape

    classes = np.unique(np.concatenate([y_true_arr, y_pred_arr]))
    num_classes = int(classes.max()) + 1 if classes.size > 0 else 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true_arr, y_pred_arr):
        cm[t, p] += 1

    true_positive = np.diag(cm).astype(np.float64)
    predicted_positive = cm.sum(axis=0).astype(np.float64)
    actual_positive = cm.sum(axis=1).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision_per_class = np.divide(true_positive, predicted_positive, out=np.zeros_like(true_positive), where=predicted_positive>0)
        recall_per_class = np.divide(true_positive, actual_positive, out=np.zeros_like(true_positive), where=actual_positive>0)
        f1_per_class = np.divide(2 * precision_per_class * recall_per_class, precision_per_class + recall_per_class, out=np.zeros_like(true_positive), where=(precision_per_class + recall_per_class)>0)

    acc = float((y_true_arr == y_pred_arr).mean()) if y_true_arr.size > 0 else 0.0
    macro_precision = float(np.mean(precision_per_class)) if precision_per_class.size > 0 else 0.0
    macro_recall = float(np.mean(recall_per_class)) if recall_per_class.size > 0 else 0.0
    macro_f1 = float(np.mean(f1_per_class)) if f1_per_class.size > 0 else 0.0

    return {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


@dataclass
class EarlyStopper:
    patience: int
    mode: str = "max"
    best_score: float = None
    counter: int = 0

    def step(self, current: float) -> bool:
        if self.best_score is None:
            self.best_score = current
            return False
        improve = current > self.best_score if self.mode == "max" else current < self.best_score
        if improve:
            self.best_score = current
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


