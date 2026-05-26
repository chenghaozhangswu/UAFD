# UAFD: User-Adaptive Fall Detection via Semi-Supervised Fine-Tuning

On-device user-adaptive fall detection using **FixMatch semi-supervised learning** + **two-stage fine-tuning** on wrist-worn IMU data.

## Overview

This project tackles a fundamental challenge in wearable fall detection: **personalization**. A population-trained model generalizes poorly to new users (domain gap of 15 F1 points), but collecting large labeled datasets per user is impractical.

**Solution:** Two-phase approach:
1. **Phase 1 (Population-level):** FixMatch semi-supervised learning on 50% labeled + 50% unlabeled data from existing subjects
2. **Phase 2 (User-adaptive):** Fine-tune on a small amount of labeled data from a new user (as few as 124 samples at 10% ratio, ~2.5 seconds of labeled falls at 50 Hz)

## Key Results

| Setting | F1 Score |
|---------|:--------:|
| FixMatch (10% labels) | 0.7126 |
| FixMatch (50% labels) | **0.7942** (97.3% of full supervision) |
| Full supervision (100%) | 0.8163 |
| User-adaptive baseline (Phase 1) | ~0.72 |
| User-adaptive (10% per-subject) | 0.7978 |
| User-adaptive (25% per-subject) | 0.9131 |
| User-adaptive (30% per-subject) | **0.9252** |
| Cross-domain zero-shot | 0.570 |
| Cross-domain (30% adapt) | **0.770** |

## Datasets

- **HIFD:** 21 subjects, wrist-worn IMU (50 Hz), 11 channels (acc + quaternion + gyro + HR), 4,175 training / 1,266 test samples
- **BITS Pilani (Zenodo):** 41 subjects, cross-domain validation target, adapted to match HIFD format

## Repository Structure

```
├── train.py                  # Main training script (Phase 1 FixMatch + Phase 2 adaptation)
├── prepare_fd.py             # Data loading, model definition, evaluation metrics
├── resplit_data.py           # Data splitting with configurable labeled ratios
├── create_adaptive_splits.py # Per-subject adaptive data splits
├── prepare_zenodo.py         # BITS Pilani (Zenodo) data preprocessing
├── normalize_zenodo_aug.py   # Zenodo data normalization to HIFD distribution
├── program.md                # Research program design document
├── pyproject.toml            # Python project configuration
├── uv.lock                   # Dependency lock file
├── .python-version           # Python version pinning
├── .gitignore
```

## Dependencies

```bash
pip install torch numpy scipy scikit-learn matplotlib
```

Or with uv:
```bash
uv sync
```

## Usage

### Data Preparation

Prepare HIFD dataset:
```bash
python prepare_fd.py
```

Prepare Zenodo (cross-domain) dataset:
```bash
python prepare_zenodo.py
python normalize_zenodo_aug.py
```

Create adaptive splits:
```bash
python create_adaptive_splits.py
```

### Training

```bash
# Phase 1: FixMatch semi-supervised learning (200s on RTX 4070 SUPER)
python train.py

# Phase 2: User-adaptive fine-tuning
# Edit ADAPT_RATIO in train.py (1-30) and run again
python train.py
```

### Reproducing Paper Results

To reproduce the main FixMatch experiment:
```python
# In train.py set:
SUPERVISED_ONLY = False   # FixMatch mode
ADAPT_RATIO = 0           # Phase 1 only
```

To reproduce user-adaptive results:
```python
ADAPT_RATIO = 30          # 30% per-subject data
ADAPTIVE_FINETUNE = True  # Phase 2 enabled
```

## Architecture

The model uses a ConvNeXt-style 1D residual CNN:
- Input: 11-channel IMU at 128 timesteps (2.56 seconds)
- Feature extractor: 4 residual Conv1D blocks (kernel size 7, hidden dim 256)
- Classifier: Global average pooling → LayerNorm → Linear(256, 2)
- **1.84M parameters** (~7.4 MB, primary model), **7.35M** (high-capacity variant, cross-domain)

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{uafd2026,
  title={On-Device User-Adaptive Fall Detection via Semi-Supervised Fine-Tuning on Wearable IMU Data},
  author={Zhang, Chenghao},
  booktitle={MobiQuitous},
  year={2026}
}
```

## License

This project is provided for academic research purposes.