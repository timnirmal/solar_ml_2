### Implementation Plan

This plan outlines concrete steps to build a reproducible solar panel fault detection pipeline with a U-Net-based optional segmentation stage and an InceptionV3-based classifier head enhanced with SE blocks, residual connections, and GAP.

### Tech Stack

- Python 3.12 in local `.venv`
- PyTorch, TorchVision
- Albumentations or TorchVision transforms for augmentation
- PyYAML or JSON for configs

### Repository Structure

- `configs/`:
  - `default.yaml`: dataset paths, hyperparams, model toggles
- `data/`:
  - Placeholder; dataset path configurable and not committed
- `datasets/`:
  - `classification_dataset.py`: 6-class dataset loader, splits, weights
  - `segmentation_dataset.py`: mask-based dataset (optional)
- `models/`:
  - `unet.py`: U-Net for PV region segmentation
  - `inception_classifier.py`: InceptionV3 backbone with SE-enhanced classification head
  - `layers/se.py`: Squeeze-and-Excitation blocks
- `training/`:
  - `train_classifier.py`: training loop for classifier
  - `train_segmenter.py`: training loop for U-Net (optional)
  - `eval_classifier.py`: evaluation + confusion matrix
  - `utils.py`: metrics, logging, seed, checkpointing
- `scripts/`:
  - `extract_pdfs.py`: already added
  - `prepare_splits.py`: create 60/20/20 splits if required
- `runs/`: outputs as `run-<datetime>`

### Data Assumptions

- Classification data directory structure (configurable):
  - `root/clean/...`, `root/dusty/...`, `root/bird-drop/...`, `root/electrical-damage/...`, `root/physical-damage/...`, `root/snow-covered/...`
- Segmentation data: images and masks paths supplied via config when used.

### Models

- U-Net:
  - Encoder-decoder with skip connections; channels [64, 128, 256, 512, 1024]; Dice/CE loss.
- InceptionV3 classifier:
  - Use ImageNet weights; freeze early layers initially; replace classifier with:
    - Global Average Pooling (if needed via adaptive pooling)
    - Residual MLP head with BatchNorm + Dropout
    - SE block applied to pooled features
    - Final `Linear(out_features=6)` + Softmax/cross-entropy training

### Training

- Input size: 256×256
- Augmentations: random resize/crop, horizontal/vertical flip, color jitter, gaussian blur, Cutout/CoarseDropout
- Optimizer: AdamW; LR scheduler: cosine or step decay
- Loss: CrossEntropy with class weights or Focal Loss
- Metrics: accuracy, macro/micro F1, precision, recall
- Checkpointing: best by macro-F1; last checkpoint each epoch
- Logging: per-epoch metrics JSON + CSV under `runs/run-<datetime>`

### CLI

- `train_classifier.py --config configs/default.yaml --runs_dir runs`
- `eval_classifier.py --checkpoint path --data_root path`
- `train_segmenter.py` similar, optional

### Milestones

1) Scaffolding, config, requirements
2) Dataset loaders + transforms
3) Models (SE block, classifier head, U-Net)
4) Training loop + evaluation
5) Reproducible runs with manifests
6) Optional segmentation integration toggle in config


