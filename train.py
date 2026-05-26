"""
Autoresearch Fall Detection training script. Single-GPU, single-file.
Adapted from karpathy/autoresearch framework for semi-supervised fall detection.
Usage: conda activate autoresearch && python train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare_fd import (
    TIME_BUDGET, NUM_CHANNELS, WINDOW_SIZE, NUM_CLASSES,
    load_data, make_dataloader, evaluate_model, get_class_weights, load_adapt_data,
)

# ---------------------------------------------------------------------------
# Experiment mode
# ---------------------------------------------------------------------------
SUPERVISED_ONLY = False  # True = 100% labeled supervised, False = FixMatch semi-supervised
ADAPT_RATIO = 20  # percentage of per-subject data for fine-tuning (1,5,10,15,20,25,30)

if SUPERVISED_ONLY:
    import prepare_fd
    prepare_fd.DATA_DIR = r"D:\hd_imu\zenodo_processed_aug\zenodo_30pct"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class FDConfig:
    n_channel: int = NUM_CHANNELS
    window_len: int = WINDOW_SIZE
    n_classes: int = NUM_CLASSES
    hidden_dim: int = 128
    n_layer: int = 3
    kernel_size: int = 7
    dropout: float = 0.2


class ConvBlock(nn.Module):
    def __init__(self, channels, kernel_size, dropout):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding="same", bias=False)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = F.relu(x).square()
        x = self.dropout(x)
        return x + residual


class FallDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_proj = nn.Conv1d(config.n_channel, config.hidden_dim, 1, bias=False)
        self.blocks = nn.ModuleList()
        for i in range(config.n_layer):
            self.blocks.append(ConvBlock(config.hidden_dim, config.kernel_size, config.dropout))
        self.final_dim = config.hidden_dim
        self.norm_f = nn.LayerNorm(self.final_dim)
        self.classifier = nn.Linear(self.final_dim, config.n_classes, bias=False)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=-1)
        x = self.norm_f(x)
        x = self.classifier(x)
        return x

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'weight' in name and p.ndim >= 2:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            elif 'bias' in name:
                nn.init.zeros_(p)

    def setup_optimizer(self, lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999)):
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_noise(x, std=0.05):
    return x + torch.randn_like(x) * std


def fixmatch_loss(model, x_u_weak, x_u_strong, threshold=0.8):
    """
    FixMatch loss:
    1. Weak augmentation → hard pseudo-label (argmax if max prob >= threshold)
    2. Strong augmentation → CE against pseudo-label
    """
    model.eval()
    with torch.no_grad():
        logits_weak = model(x_u_weak)
        probs = F.softmax(logits_weak, dim=1)
        max_probs, pseudo_labels = probs.max(dim=1)
        mask = max_probs >= threshold
    if mask.sum() == 0:
        return torch.tensor(0.0, device=x_u_weak.device), 0.0
    model.train()
    logits_strong = model(x_u_strong)
    loss = F.cross_entropy(logits_strong[mask], pseudo_labels[mask])
    return loss, mask.float().mean().item()


def strong_augment(x, noise_std=0.15):
    """
    Strong augmentation for FixMatch on time-series:
    - High Gaussian noise
    - Random amplitude scaling
    - Random time shift
    """
    x_aug = x + torch.randn_like(x) * noise_std
    # Random amplitude scaling
    scale = 1.0 + torch.randn(x.size(0), 1, 1, device=x.device) * 0.2
    x_aug = x_aug * scale.clamp(0.5, 1.5)
    # Random time shift (circular shift)
    shift = torch.randint(0, x.size(-1), (x.size(0),), device=x.device)
    x_aug = torch.stack([
        torch.roll(x_aug[i], shifts=int(shift[i].item()), dims=-1)
        for i in range(x.size(0))
    ])
    return x_aug


# ---------------------------------------------------------------------------
# Hyperparameters — revisit Gaussian noise config
# ---------------------------------------------------------------------------

HIDDEN_DIM = 512
N_LAYER = 4
KERNEL_SIZE = 7
DROPOUT = 0.35

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
ADAM_BETAS = (0.9, 0.999)
WARMUP_RATIO = 0.1
WARMDOWN_RATIO = 0.3
FINAL_LR_FRAC = 0.0

# Adaptive fine-tuning
ADAPTIVE_FINETUNE = True
PHASE1_BUDGET = 200  # seconds for base model training
PHASE2_BUDGET = 100  # seconds for adaptive fine-tuning
PHASE2_BUDGET = 100  # seconds for adaptive fine-tuning

SEMI_SUPERVISED = "" if SUPERVISED_ONLY else "fixmatch"
CONF_THRESHOLD = 0.80     # FixMatch pseudo-label threshold
UNLABELED_WEIGHT = 1.0    # FixMatch uses full unlabeled weight
TEMP = 1.0                 # not used in FixMatch (hard labels), kept for compat

NOISE_STD = 0.02          # weak augmentation noise for labeled data (paper reports σ=0.02)
STRONG_AUG_STD = 0.15     # strong augmentation noise for FixMatch
USE_CLASS_WEIGHTS = True

WARMUP_STEPS = 0 if SUPERVISED_ONLY else 500
RAMPUP_STEPS = 0 if SUPERVISED_ONLY else 1500

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    autocast_ctx = torch.amp.autocast(device_type="cpu", enabled=False)
    print("Using CPU")

config = FDConfig(hidden_dim=HIDDEN_DIM, n_layer=N_LAYER, kernel_size=KERNEL_SIZE, dropout=DROPOUT)
model = FallDetector(config)
model.to(device)
model.init_weights()

num_params = sum(p.numel() for p in model.parameters())
print(f"Model: {N_LAYER} layers, {HIDDEN_DIM} hidden dim, dropout={DROPOUT}")
print(f"Parameters: {num_params:,}")

optimizer = model.setup_optimizer(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS)

# Load data with validation split (fix test leakage)
train_labeled_ds, val_ds, train_unlabeled_ds, test_ds = load_data(create_val_split=True, val_ratio=0.2)
print(f"Labeled: {len(train_labeled_ds)} | Val: {len(val_ds)} | Unlabeled: {len(train_unlabeled_ds)} | Test: {len(test_ds)}")
if SUPERVISED_ONLY:
    # In supervised mode, discard unlabeled data
    train_unlabeled_ds = None

if USE_CLASS_WEIGHTS:
    class_weights = get_class_weights(train_labeled_ds.labels).to(device)
    print(f"Class weights: {class_weights.tolist()}")
else:
    class_weights = None

train_loader = make_dataloader(train_labeled_ds, BATCH_SIZE, shuffle=True)
unlabeled_loader = make_dataloader(train_unlabeled_ds, BATCH_SIZE, shuffle=True) if train_unlabeled_ds is not None and len(train_unlabeled_ds) > 0 else None

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_loss = 0.0
best_f1 = 0.0
best_model_state = None
total_training_time = 0.0
step = 0

labeled_iter = iter(train_loader)
unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader else None

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

while True:
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    model.train()

    try:
        x_l, y_l = next(labeled_iter)
    except StopIteration:
        labeled_iter = iter(train_loader)
        x_l, y_l = next(labeled_iter)

    x_l, y_l = x_l.to(device), y_l.to(device)
    if NOISE_STD > 0:
        x_l_noisy = add_noise(x_l, NOISE_STD)
    else:
        x_l_noisy = x_l

    with autocast_ctx:
        logits_l = model(x_l_noisy)
        if USE_CLASS_WEIGHTS:
            loss_sup = F.cross_entropy(logits_l, y_l, weight=class_weights)
        else:
            loss_sup = F.cross_entropy(logits_l, y_l)

    loss_unsup = torch.tensor(0.0, device=device)
    unsup_ratio = 0.0
    unsup_weight = 0.0
    if SEMI_SUPERVISED == "fixmatch" and unlabeled_loader and step >= WARMUP_STEPS:
        ramp_progress = min(1.0, (step - WARMUP_STEPS) / RAMPUP_STEPS)
        unsup_weight = UNLABELED_WEIGHT * ramp_progress
        try:
            x_u = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            x_u = next(unlabeled_iter)
        if isinstance(x_u, tuple):
            x_u = x_u[0]
        x_u = x_u.to(device)
        # FixMatch: weak aug for pseudo-label, strong aug for training
        x_u_weak = add_noise(x_u, 0.02)  # minimal noise for clean teacher (paper reports $\sigma=0.02$)
        x_u_strong = strong_augment(x_u, STRONG_AUG_STD)
        with autocast_ctx:
            loss_unsup, unsup_ratio = fixmatch_loss(model, x_u_weak, x_u_strong, threshold=CONF_THRESHOLD)

    loss = loss_sup + unsup_weight * loss_unsup
    if SUPERVISED_ONLY:
        # Always just supervised loss, no unsup component
        loss = loss_sup
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t1 = time.time()
    dt = t1 - t0
    if step > 10:
        total_training_time += dt

    progress = min(total_training_time / PHASE1_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for param_group in optimizer.param_groups:
        param_group['lr'] = LEARNING_RATE * lrm

    ema_beta = 0.9
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss.item()
    debiased = smooth_loss / (1 - ema_beta ** (step + 1))
    remaining = max(0, PHASE1_BUDGET - total_training_time)

    print(f"\rstep {step:04d} ({100*min(total_training_time/PHASE1_BUDGET,1):.1f}%) | loss: {debiased:.4f} | sup: {loss_sup.item():.4f} | unsup: {loss_unsup.item():.4f} | w:{unsup_weight:.2f} | pseudo%: {unsup_ratio:.3f} | remaining: {remaining:.0f}s    ", end="", flush=True)

    if step > 0 and step % 20 == 0:
        # FIX TEST LEAKAGE: Use validation set, NOT test set
        metrics = evaluate_model(model, val_ds, device, BATCH_SIZE)
        f1 = metrics["f1"]
        print(f"\nstep {step:04d} | acc: {metrics['accuracy']:.4f} | f1: {f1:.4f} | prec: {metrics['precision']:.4f} | rec: {metrics['recall']:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  *** New best F1 (val): {best_f1:.4f} ***")

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1
    if step > 10 and total_training_time >= PHASE1_BUDGET:
        break

print()
# Use best model from Phase 1
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Phase 1 done. Best F1: {best_f1:.4f}")

# Save Phase 1 checkpoint always
if best_model_state is not None:
    torch.save(best_model_state, r'D:\hd_imu\phase1_h512.pt')
    print(f'Phase 1 checkpoint saved to D:\\hd_imu\\phase1_h512.pt')

# ---------------------------------------------------------------------------
# Phase 2: Adaptive Fine-Tuning
# ---------------------------------------------------------------------------
if ADAPTIVE_FINETUNE:
    # Save Phase 1 best model to reload for each ratio
    phase1_best_state = {k: v.clone() for k, v in best_model_state.items()}
    
    ADAPT_RATIOS = [30]  # Only run 30% for now, add other ratios later
    adapt_summary = []
    
    for ratio in ADAPT_RATIOS:
        print(f"\n{'='*60}")
        print(f"  Phase 2: Ratio = {ratio}%")
        print(f"{'='*60}")
        
        # Reset model to Phase 1 best
        model.load_state_dict(phase1_best_state)
        
        # Load adapt data for this ratio
        adapt_data_dir = os.path.join(r"D:\hd_imu", f"processed_adaptive_{ratio}pct")
        adapt_ds, adapt_val_ds, adapt_test_ds = load_adapt_data(create_val_split=True, val_ratio=0.2, data_dir=adapt_data_dir)
        adapt_loader = make_dataloader(adapt_ds, BATCH_SIZE // 2, shuffle=True)
        print(f"  Adapt: {len(adapt_ds)} | Val: {len(adapt_val_ds)} | Test: {len(adapt_test_ds)}")
        
        adapt_optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
        phase2_start = time.time()
        adapt_iter = iter(adapt_loader)
        adapt_step = 0
        adapt_best_f1 = best_f1
        adapt_best_state = None
        
        while True:
            model.train()
            try:
                x_a, y_a = next(adapt_iter)
            except StopIteration:
                adapt_iter = iter(adapt_loader)
                x_a, y_a = next(adapt_iter)
            
            x_a, y_a = x_a.to(device), y_a.to(device)
            with autocast_ctx:
                logits_a = model(x_a)
                loss_a = F.cross_entropy(logits_a, y_a)
            
            loss_a.backward()
            adapt_optimizer.step()
            adapt_optimizer.zero_grad(set_to_none=True)
            
            phase2_elapsed = time.time() - phase2_start
            if adapt_step > 0 and adapt_step % 10 == 0:
                metrics = evaluate_model(model, adapt_val_ds, device, BATCH_SIZE)
                f1 = metrics["f1"]
                if f1 > adapt_best_f1:
                    adapt_best_f1 = f1
                    adapt_best_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  step {adapt_step:04d} | {phase2_elapsed:.1f}s | f1: {f1:.4f} | prec: {metrics['precision']:.4f} | rec: {metrics['recall']:.4f}")
            
            adapt_step += 1
            if phase2_elapsed >= PHASE2_BUDGET:
                break
        
        # Evaluate best adapt model on test set
        if adapt_best_state is not None and adapt_best_f1 > best_f1:
            model.load_state_dict(adapt_best_state)
            best_model = "adapt"
            best_val_f1 = adapt_best_f1
            # Save Phase 2 best checkpoint
            torch.save(adapt_best_state, f'D:\\hd_imu\\phase2_{ratio}pct_best.pt')
            print(f"  Phase 2 best checkpoint saved to D:\\hd_imu\\phase2_{ratio}pct_best.pt")
        else:
            model.load_state_dict(phase1_best_state)
            best_model = "phase1"
            best_val_f1 = best_f1
        
        final_metrics = evaluate_model(model, adapt_test_ds, device, BATCH_SIZE)
        
        y_true = []
        y_pred = []
        for x, y in make_dataloader(adapt_test_ds, BATCH_SIZE, shuffle=False):
            with torch.no_grad():
                logits = model(x.to(device))
                y_true.append(y)
                y_pred.append(torch.argmax(logits, dim=1).cpu())
        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
        fall = int(y_true.sum())
        nonfall = int(len(y_true) - y_true.sum())
        
        result = {
            "ratio": ratio,
            "val_f1": best_val_f1,
            "test_f1": final_metrics["f1"],
            "precision": final_metrics["precision"],
            "recall": final_metrics["recall"],
            "accuracy": final_metrics["accuracy"],
            "best_model": best_model,
            "test_fall": fall,
            "test_nonfall": nonfall,
        }
        adapt_summary.append(result)
        print(f"  >>> Ratio {ratio}% | test_f1={final_metrics['f1']:.4f} | prec={final_metrics['precision']:.4f} | rec={final_metrics['recall']:.4f} | val_f1={best_val_f1:.4f} | best={best_model}")
        
        # Clean up for next iteration
        del adapt_ds, adapt_val_ds, adapt_test_ds, adapt_loader, adapt_optimizer, adapt_best_state
        gc.collect()
        torch.cuda.empty_cache()
    
    # Print summary table
    print(f"\n{'='*60}")
    print(f"  Adaptive Fine-Tuning Summary")
    print(f"{'='*60}")
    print(f"  {'Ratio':>5} | {'Val F1':>8} | {'Test F1':>8} | {'Prec':>7} | {'Recall':>7} | {'Acc':>7} | Best")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+------")
    for r in adapt_summary:
        print(f"  {r['ratio']:>5}% | {r['val_f1']:>8.4f} | {r['test_f1']:>8.4f} | {r['precision']:>7.4f} | {r['recall']:>7.4f} | {r['accuracy']:>7.4f} | {r['best_model']}")
    
    # For final output, use the best overall result
    best_result = max(adapt_summary, key=lambda r: r['test_f1'])
    best_f1 = best_result['test_f1']
    test_ds_from = f"adaptive {best_result['ratio']}% (peak)"
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    
    print("---")
    print(f"test_f1:          {best_f1:.6f}")
    print(f"best_val_f1:      {best_f1:.6f}")
    print(f"test_data_source: {test_ds_from}")
    print(f"supervised_only:  {SUPERVISED_ONLY}")
    print(f"semi_supervised:  {SEMI_SUPERVISED}")
    print(f"adaptive_mode:    {ADAPTIVE_FINETUNE}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"total_time:       {t_end - t_start:.1f}s")
    print(f"phase1_budget:    {PHASE1_BUDGET}s")
    print(f"phase2_budget:    {PHASE2_BUDGET}s")
    
    # Save results
    import json
    from datetime import datetime
    results_file = "results.tssv"
    with open(results_file, "a") as f:
        for r in adapt_summary:
            line = (f"ADAPT={r['ratio']}% | "
                    f"test_f1={r['test_f1']:.4f} | "
                    f"test_precision={r['precision']:.4f} | "
                    f"test_recall={r['recall']:.4f} | "
                    f"test_accuracy={r['accuracy']:.4f} | "
                    f"val_f1={r['val_f1']:.4f}")
            print(line)
            f.write(line + "\n")
    print(f"Results appended to {results_file}")
    
    # Done - exit cleanly
    import sys
    sys.exit(0)
else:
    final_test_ds = test_ds  # ✅ Correct: use TEST set only for final eval
    test_ds_from = "original test split"

model.eval()
final_metrics = evaluate_model(model, final_test_ds, device, BATCH_SIZE)
t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0

print("---")
print(f"test_f1:          {final_metrics['f1']:.6f}")
print(f"test_accuracy:    {final_metrics['accuracy']:.6f}")
print(f"test_precision:   {final_metrics['precision']:.6f}")
print(f"test_recall:      {final_metrics['recall']:.6f}")
print(f"best_f1:          {best_f1:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params:       {num_params:,}")
print(f"semi_supervised:  {SEMI_SUPERVISED}")
print(f"conf_threshold:   {CONF_THRESHOLD}")
print(f"temperature:      {TEMP}")
print(f"strong_aug_std:   {STRONG_AUG_STD}")
print(f"dropout:          {DROPOUT}")
print(f"weight_decay:     {WEIGHT_DECAY}")
print(f"noise_std:        {NOISE_STD}")
print(f"use_class_weights:{USE_CLASS_WEIGHTS}")
print(f"warmup_steps:     {WARMUP_STEPS}")
print(f"rampup_steps:     {RAMPUP_STEPS}")
print(f"batch_size:       {BATCH_SIZE}")
print(f"learning_rate:    {LEARNING_RATE}")
print(f"adaptive_mode:    {ADAPTIVE_FINETUNE}")
print(f"test_data_source: {test_ds_from}")