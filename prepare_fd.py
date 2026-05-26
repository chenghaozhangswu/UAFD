"""
Fall Detection data preparation and utilities.
Fixed constants, data loading, dataloader, evaluation.
Not modified by the agent.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# ---------------------------------------------------------------------------
# Fixed Constants
# ---------------------------------------------------------------------------
TIME_BUDGET = 300  # 5 minutes per experiment
DATA_DIR = r"D:\hd_imu\processed_50pct"
NUM_CHANNELS = 11
WINDOW_SIZE = 128
NUM_CLASSES = 2

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FallDetectionDataset(Dataset):
    """Fall detection dataset with optional labels."""

    def __init__(self, samples, labels=None):
        self.samples = torch.from_numpy(samples).float()
        if labels is not None:
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]  # (WINDOW_SIZE, NUM_CHANNELS)
        if self.labels is not None:
            return x, self.labels[idx]
        return x

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def create_validation_split(samples, labels, val_ratio=0.2, random_state=42):
    """Create train/val split from labeled data."""
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        samples, labels, 
        test_size=val_ratio, 
        random_state=random_state,
        stratify=labels
    )
    
    train_ds = FallDetectionDataset(X_train, y_train)
    val_ds = FallDetectionDataset(X_val, y_val)
    
    return train_ds, val_ds


def load_data(create_val_split=True, val_ratio=0.2):
    """Load labeled, unlabeled, validation, and test datasets.
    
    Args:
        create_val_split: If True, create validation split from training data
        val_ratio: Validation ratio (default 0.2 = 20%)
    
    Returns:
        If create_val_split=True: (train_labeled_ds, val_ds, train_unlabeled_ds, test_ds)
        If create_val_split=False: (train_labeled_ds, train_unlabeled_ds, test_ds)  # backward compat
    """
    train_labeled_samples = np.load(os.path.join(DATA_DIR, "train_labeled_samples.npy"))
    train_labeled_labels = np.load(os.path.join(DATA_DIR, "train_labeled_labels.npy"))
    train_unlabeled_samples = np.load(os.path.join(DATA_DIR, "train_unlabeled_samples.npy"))
    test_samples = np.load(os.path.join(DATA_DIR, "test_samples.npy"))
    test_labels = np.load(os.path.join(DATA_DIR, "test_labels.npy"))

    if create_val_split:
        # Create validation split from training data
        train_labeled_ds, val_ds = create_validation_split(
            train_labeled_samples, train_labeled_labels, val_ratio=val_ratio
        )
    else:
        # No validation split (backward compatibility)
        train_labeled_ds = FallDetectionDataset(train_labeled_samples, train_labeled_labels)
        val_ds = None
    
    train_unlabeled_ds = FallDetectionDataset(train_unlabeled_samples)  # no labels
    test_ds = FallDetectionDataset(test_samples, test_labels)

    if create_val_split:
        return train_labeled_ds, val_ds, train_unlabeled_ds, test_ds
    else:
        return train_labeled_ds, train_unlabeled_ds, test_ds


def load_adapt_data(create_val_split=True, val_ratio=0.2, data_dir=None):
    """Load adapt and test datasets for user-adaptive fine-tuning.
    
    Args:
        create_val_split: If True, create validation split from adapt data
        val_ratio: Validation ratio (default 0.2 = 20%)
        data_dir: Optional override for data directory (default: global DATA_DIR)
    
    Returns:
        If create_val_split=True: (adapt_ds, adapt_val_ds, test_ds)
        If create_val_split=False: (adapt_ds, test_ds)  # backward compat
    """
    adapt_dir = data_dir if data_dir is not None else DATA_DIR
    adapt_samples = np.load(os.path.join(adapt_dir, "adapt_samples.npy"))
    adapt_labels = np.load(os.path.join(adapt_dir, "adapt_labels.npy"))
    test_samples = np.load(os.path.join(adapt_dir, "test_samples.npy"))
    test_labels = np.load(os.path.join(adapt_dir, "test_labels.npy"))
    
    if create_val_split:
        # Create validation split from adapt data
        adapt_ds, adapt_val_ds = create_validation_split(
            adapt_samples, adapt_labels, val_ratio=val_ratio
        )
    else:
        adapt_ds = FallDetectionDataset(adapt_samples, adapt_labels)
        adapt_val_ds = None
    
    test_ds = FallDetectionDataset(test_samples, test_labels)
    
    if create_val_split:
        return adapt_ds, adapt_val_ds, test_ds
    else:
        return adapt_ds, test_ds





def make_dataloader(dataset, batch_size, shuffle=True):
    """Create a dataloader for a given dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred):
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def evaluate_model(model, test_ds, device, batch_size):
    """Evaluate model on test set. Returns metrics dict."""
    model.eval()
    loader = make_dataloader(test_ds, batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metrics = compute_metrics(all_labels, all_preds)
    return metrics


def get_class_weights(labels):
    """Compute class weights for imbalanced data."""
    counts = np.bincount(labels.numpy())
    total = len(labels)
    weights = total / (len(counts) * counts)
    return torch.from_numpy(weights).float()


def print_metrics(metrics, prefix="eval"):
    """Pretty print metrics."""
    print(f"{prefix} | acc: {metrics['accuracy']:.4f} | f1: {metrics['f1']:.4f} | prec: {metrics['precision']:.4f} | rec: {metrics['recall']:.4f}")