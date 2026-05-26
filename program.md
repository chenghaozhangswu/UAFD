# Fall Detection Semi-Supervised Research Agent

Optimize a fall detection model using the HIFD dataset (Heart Rate + IMU sensor data at wrist). Only **10% labeled data** (417 samples) with **90% unlabeled** (3,758 samples) — push the F1 score as high as possible.

## Dataset

- **Source**: `D:\hd_imu\processed\` (preprocessed, loaded by `prepare_fd.py`)
- **21 subjects**, wrist-worn sensor, 50 Hz
- **11 channels**: ax, ay, az (accel), w, x, y, z (quaternions), droll, dpitch, dyaw (gyro), heart (PPG)
- **Window**: 128 timesteps (2.56 seconds)
- **Split**: labeled train 417, unlabeled train 3,758, test 1,266

## Files

- `train.py` — **EDIT THIS** — model, optimizer, hyperparams, training loop. Everything is fair game.
- `prepare_fd.py` — **DO NOT EDIT** — data loading, evaluation (F1 metric), constants.

## Running

```bash
conda activate autoresearch
cd C:\Users\Administrator\autoresearch
python train.py
```

Prints a summary:
```
test_f1:          0.6741
training_seconds: 300.0
peak_vram_mb:     392.3
num_steps:        2667
num_params:       784,962
```

## Hyperparameters (in train.py)

### Architecture
- `HIDDEN_DIM`: 128 — classifier hidden dimension
- `N_LAYER`: 3 — number of Conv1D blocks
- `KERNEL_SIZE`: 7 — conv kernel
- `DROPOUT`: 0.2

### Optimization
- `BATCH_SIZE`: 64
- `LEARNING_RATE`: 1e-3
- `WEIGHT_DECAY`: 1e-4
- `WARMUP_RATIO`: 0.1, `WARMDOWN_RATIO`: 0.3

### Semi-supervised
- `SEMI_SUPERVISED`: `"none"`, `"self_training"`, `"mean_teacher"`, `"fixmatch"`
- `CONF_THRESHOLD`: 0.90

## Goal

Maximize **test_f1** within the 5-minute time budget. Current baseline:
- Supervised (10% labeled): `test_f1≈0.67, best_f1≈0.69`
- Self-training (10%+90%): `test_f1≈0.67, best_f1≈0.71`

## What you can try

**Architecture**: CNN depth/width, LSTM/GRU, Transformer encoder, residual connections, attention pooling, multi-scale conv, dilation.

**Semi-supervised**: FixMatch, Mean Teacher, contrastive pretraining (SimCLR), MixMatch, temporal ensembling, pseudo-label refinement.

**Augmentation**: noise injection, time warp, scaling, mixup, CutMix.

**Optimization**: schedule tuning, different optimizers, gradient clipping, batch size, class-balanced loss.

## Logging

Results go to `experiment_log.md`. The `results.tsv` format:

```
commit	test_f1	memory_gb	status	description
abc1234	0.6741	0.4	keep	baseline self-training
```

## Experiment loop (autonomous)

1. Read current state (branch/commit, experiment_log.md, results.tsv)
2. Tune `train.py` with an idea
3. `git commit`
4. Run: `python train.py > run.log 2>&1`
5. Read results: grep `test_f1` from run.log
6. If crash: read tail of run.log, fix or skip
7. Log to results.tsv (don't commit this file)
8. If F1 improved → keep commit; if worse → `git reset --hard`
9. Repeat indefinitely — never ask "should I stop"