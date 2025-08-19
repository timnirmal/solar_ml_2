### Project Overview: Solar Panel Fault Detection (from SPF-Net and related proposal)

This document summarizes key insights extracted from two attached PDFs and frames a practical scope for an engineering project:

- 1-s2.0-S2352484724004682-main.pdf (Energy Reports, 2024): Proposes SPF-Net — a U-Net-based segmentation + InceptionV3 classifier pipeline with enhancements (SE blocks, residuals, GAP) for solar panel fault detection. Reports 98.34% validation accuracy and 94.35% test accuracy on a 6-class dataset: clean, dusty, bird-drop, electrical damage, physical damage, snow-covered.
- 1585069_Solar_Panel_Fault_Detection_Updated.pdf (Thesis Proposal, 2025): Advocates a two-stage pipeline (object detection with YOLOv11 followed by ConvNeXt classification), emphasizes small-defect detection, and stresses synthetic data, imbalance handling, and domain robustness.

### Problem and Goals

- Objective: Detect and classify faults on solar panels from images with reliable performance under real-world conditions.
- Primary fault classes (6): Clean, Dusty, Bird-drop, Electrical-damage, Physical-damage, Snow-covered.
- Outcomes: End-to-end training/evaluation pipeline, models, metrics, and reproducible runs saved under `runs/run-<datetime>`.

### Datasets (as noted)

- Segmentation data (~4,616 images, split 60/20/20): Ground surfaces (cropland, grassland, saline-alkali, shrubwood, water-surface) and rooftop. Used to segment/locate PV regions.
- Classification data (~885 images, split 60/20/20): Six classes listed above. Imbalance exists across classes.
- Source reference example: Aerial/satellite PV imagery dataset (e.g., Zenodo record referenced in the paper). The project will remain dataset-agnostic with pluggable loaders.

### Core Approach (inspired by SPF-Net)

- Stage A: PV region segmentation with U-Net to focus analysis on panel areas (optional when inputs are already panel-cropped).
- Stage B: Panel surface condition classification using an InceptionV3-based network with a modernized head (SE blocks, residual head, batch norm, GAP, Softmax for 6 classes).
- Data handling: 256×256 crops, normalization, augmentations (color jitter, blur, random occlusion), class-balancing strategies.

### Metrics and Targets

- Classification: Accuracy, F1, Precision, Recall; confusion matrix and per-class metrics.
- Segmentation: IoU/Dice for PV masks (if used).
- Baseline goal: Reproduce strong test metrics (≥90% accuracy, macro-F1 close to the 0.9–0.94 range).

### Risks and Mitigations

- Data imbalance and small defects: Use class weights, focal loss, re-sampling, targeted augmentations; optionally ROI extraction and two-stage zoom.
- Domain shift (lighting, weather): Strong augmentation; validate on diverse conditions; optional TTA.
- Overfitting (small datasets): Transfer learning, regularization, cross-validation.

### Scope for This Project

- Implement a PyTorch pipeline with:
  - U-Net segmentation module (optional in the training flow, enabled by config).
  - InceptionV3-based classifier with an SE-enhanced head for 6-class classification.
  - Reproducible training/eval scripts writing artifacts to `runs/run-<datetime>`.
  - Clear configs, metrics, and structured logs.

### Future Extensions (inspired by the proposal)

- Two-stage small-defect pipeline (detector + high-precision classifier).
- Synthetic data generation and domain adaptation.
- Transformer backbones and attention (e.g., Swin, RT-DETR) for small-target sensitivity.



