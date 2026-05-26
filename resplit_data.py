"""
Re-split HIFD dataset with 20% labeled ratio.
Reads original .mat files from D:\hd_imu, creates processed_20pct/ with new split.
"""
import os, json, random
import numpy as np
import scipy.io as sio

DATA_ROOT = r"D:\hd_imu"
OUT_DIR = os.path.join(DATA_ROOT, "processed_20pct")
WINDOW_SIZE = 128
STRIDE = 64
TEST_SUBJECTS = [1, 5, 10, 15, 20]
LABELED_RATIO = 0.2
SEED = 42

def extract_windows(mat_data, label):
    """Extract sliding windows from a .mat file."""
    channels = ['w', 'x', 'y', 'z', 'droll', 'dpitch', 'dyaw', 'ax', 'ay', 'az', 'heart']
    n = mat_data['ax'].shape[0]
    windows = []
    labels = []
    for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
        win = np.stack([mat_data[ch][start:start+WINDOW_SIZE, 0].astype(np.float64) for ch in channels], axis=1)
        windows.append(win)
        labels.append(label)
    return windows, labels

def process_subject(subject_id):
    """Process all files for a subject."""
    sdir = os.path.join(DATA_ROOT, f"subject_{subject_id:02d}")
    all_windows = []
    all_labels = []
    # Fall
    fall_dir = os.path.join(sdir, "fall")
    if os.path.isdir(fall_dir):
        for f in sorted(os.listdir(fall_dir)):
            if f.endswith('.mat'):
                mat = sio.loadmat(os.path.join(fall_dir, f))
                wins, labs = extract_windows(mat, 1)
                all_windows.extend(wins)
                all_labels.extend(labs)
    # Non-fall
    nonfall_dir = os.path.join(sdir, "non-fall")
    if os.path.isdir(nonfall_dir):
        for f in sorted(os.listdir(nonfall_dir)):
            if f.endswith('.mat'):
                mat = sio.loadmat(os.path.join(nonfall_dir, f))
                wins, labs = extract_windows(mat, 0)
                all_windows.extend(wins)
                all_labels.extend(labs)
    return all_windows, all_labels

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    # Collect train data (non-test subjects)
    train_windows = []
    train_labels = []
    train_subjects = [i for i in range(1, 22) if i not in TEST_SUBJECTS]
    for sid in train_subjects:
        wins, labs = process_subject(sid)
        train_windows.extend(wins)
        train_labels.extend(labs)

    # Collect test data
    test_windows = []
    test_labels = []
    for sid in TEST_SUBJECTS:
        wins, labs = process_subject(sid)
        test_windows.extend(wins)
        test_labels.extend(labs)

    train_windows = np.array(train_windows)
    train_labels = np.array(train_labels)
    test_windows = np.array(test_windows)
    test_labels = np.array(test_labels)

    print(f"Train: {len(train_windows)} (fall={train_labels.sum()}, nonfall={len(train_labels)-train_labels.sum()})")
    print(f"Test: {len(test_windows)} (fall={test_labels.sum()}, nonfall={len(test_labels)-test_labels.sum()})")

    # Normalize using training data statistics
    mean = train_windows.mean(axis=(0, 1), keepdims=True)
    std = train_windows.std(axis=(0, 1), keepdims=True) + 1e-8
    train_windows = (train_windows - mean) / std
    test_windows = (test_windows - mean) / std

    # Split train into labeled/unlabeled with stratification
    fall_idx = np.where(train_labels == 1)[0]
    nonfall_idx = np.where(train_labels == 0)[0]

    n_labeled_fall = int(len(fall_idx) * LABELED_RATIO)
    n_labeled_nonfall = int(len(nonfall_idx) * LABELED_RATIO)

    random.shuffle(fall_idx)
    random.shuffle(nonfall_idx)

    labeled_fall = fall_idx[:n_labeled_fall]
    labeled_nonfall = nonfall_idx[:n_labeled_nonfall]
    unlabeled_fall = fall_idx[n_labeled_fall:]
    unlabeled_nonfall = nonfall_idx[n_labeled_nonfall:]

    labeled_idx = np.concatenate([labeled_fall, labeled_nonfall])
    unlabeled_idx = np.concatenate([unlabeled_fall, unlabeled_nonfall])

    train_labeled_samples = train_windows[labeled_idx]
    train_labeled_labels = train_labels[labeled_idx]
    train_unlabeled_samples = train_windows[unlabeled_idx]

    print(f"Labeled: {len(train_labeled_samples)} (fall={train_labeled_labels.sum()}, nonfall={len(train_labeled_labels)-train_labeled_labels.sum()})")
    print(f"Unlabeled: {len(train_unlabeled_samples)}")

    # Save
    np.save(os.path.join(OUT_DIR, "train_labeled_samples.npy"), train_labeled_samples)
    np.save(os.path.join(OUT_DIR, "train_labeled_labels.npy"), train_labeled_labels)
    np.save(os.path.join(OUT_DIR, "train_unlabeled_samples.npy"), train_unlabeled_samples)
    np.save(os.path.join(OUT_DIR, "test_samples.npy"), test_windows)
    np.save(os.path.join(OUT_DIR, "test_labels.npy"), test_labels)
    np.save(os.path.join(OUT_DIR, "norm_mean.npy"), mean)
    np.save(os.path.join(OUT_DIR, "norm_std.npy"), std)

    meta = {
        "num_channels": 11,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "sampling_rate": 50,
        "test_subjects": TEST_SUBJECTS,
        "labeled_ratio": LABELED_RATIO,
        "train_labeled_size": len(train_labeled_samples),
        "train_unlabeled_size": len(train_unlabeled_samples),
        "test_size": len(test_windows),
        "class_distribution": {
            "train_labeled": {
                "fall": int(train_labeled_labels.sum()),
                "nonfall": int(len(train_labeled_labels) - train_labeled_labels.sum())
            },
            "test": {
                "fall": int(test_labels.sum()),
                "nonfall": int(len(test_labels) - test_labels.sum())
            }
        }
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Saved to {OUT_DIR}")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
