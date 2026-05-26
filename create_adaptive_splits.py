"""
Create adaptive data splits at various ratios for subject-adaptive ablation.
For each test subject, splits data into adapt_ratio for fine-tuning, (1-adapt_ratio) for testing.
"""
import os, json, random
import numpy as np
import scipy.io as sio

DATA_ROOT = r"D:\hd_imu"
WINDOW_SIZE = 128
STRIDE = 64
TEST_SUBJECTS = [1, 5, 10, 15, 20]
SEED = 42

def extract_windows(mat_data, label):
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
    sdir = os.path.join(DATA_ROOT, f"subject_{subject_id:02d}")
    all_windows = []
    all_labels = []
    fall_dir = os.path.join(sdir, "fall")
    if os.path.isdir(fall_dir):
        for f in sorted(os.listdir(fall_dir)):
            if f.endswith('.mat'):
                mat = sio.loadmat(os.path.join(fall_dir, f))
                wins, labs = extract_windows(mat, 1)
                all_windows.extend(wins)
                all_labels.extend(labs)
    nonfall_dir = os.path.join(sdir, "non-fall")
    if os.path.isdir(nonfall_dir):
        for f in sorted(os.listdir(nonfall_dir)):
            if f.endswith('.mat'):
                mat = sio.loadmat(os.path.join(nonfall_dir, f))
                wins, labs = extract_windows(mat, 0)
                all_windows.extend(wins)
                all_labels.extend(labs)
    return np.array(all_windows), np.array(all_labels)

def create_split(adapt_ratio, out_dir):
    """Create adaptive split with given adapt_ratio."""
    os.makedirs(out_dir, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    # Collect train data from non-test subjects (100% labeled for Phase 1)
    train_windows = []
    train_labels = []
    train_subjects = [i for i in range(1, 22) if i not in TEST_SUBJECTS]
    for sid in train_subjects:
        wins, labs = process_subject(sid)
        train_windows.append(wins)
        train_labels.append(labs)

    if len(train_windows) > 0:
        train_windows = np.concatenate(train_windows)
        train_labels = np.concatenate(train_labels)
    else:
        train_windows = np.array(train_windows)
        train_labels = np.array(train_labels)

    # Collect test subjects data, split per-subject into adapt/test
    adapt_windows = []
    adapt_labels = []
    test_windows = []
    test_labels = []

    for sid in TEST_SUBJECTS:
        wins, labs = process_subject(sid)

        # Get fall and nonfall indices separately for stratified split
        fall_mask = labs == 1
        nonfall_mask = labs == 0
        fall_idx = np.where(fall_mask)[0]
        nonfall_idx = np.where(nonfall_mask)[0]

        # Shuffle within each class
        np.random.shuffle(fall_idx)
        np.random.shuffle(nonfall_idx)

        n_fall = len(fall_idx)
        n_nonfall = len(nonfall_idx)
        n_adapt_fall = max(1, int(n_fall * adapt_ratio))
        n_adapt_nonfall = max(1, int(n_nonfall * adapt_ratio))

        adapt_fall = fall_idx[:n_adapt_fall]
        adapt_nonfall = nonfall_idx[:n_adapt_nonfall]
        test_fall = fall_idx[n_adapt_fall:]
        test_nonfall = nonfall_idx[n_adapt_nonfall:]

        adapt_idx = np.concatenate([adapt_fall, adapt_nonfall])
        test_idx = np.concatenate([test_fall, test_nonfall])

        adapt_windows.append(wins[adapt_idx])
        adapt_labels.append(labs[adapt_idx])
        test_windows.append(wins[test_idx])
        test_labels.append(labs[test_idx])

    adapt_windows = np.concatenate(adapt_windows)
    adapt_labels = np.concatenate(adapt_labels)
    test_windows = np.concatenate(test_windows)
    test_labels = np.concatenate(test_labels)

    # Normalize using only training data statistics
    mean = train_windows.mean(axis=(0, 1), keepdims=True)
    std = train_windows.std(axis=(0, 1), keepdims=True) + 1e-8

    train_windows = (train_windows - mean) / std
    adapt_windows = (adapt_windows - mean) / std
    test_windows = (test_windows - mean) / std

    # Save
    np.save(os.path.join(out_dir, "train_labeled_samples.npy"), train_windows)
    np.save(os.path.join(out_dir, "train_labeled_labels.npy"), train_labels)
    np.save(os.path.join(out_dir, "train_unlabeled_samples.npy"), np.zeros((0, WINDOW_SIZE, 11)))  # empty placeholder
    np.save(os.path.join(out_dir, "adapt_samples.npy"), adapt_windows)
    np.save(os.path.join(out_dir, "adapt_labels.npy"), adapt_labels)
    np.save(os.path.join(out_dir, "test_samples.npy"), test_windows)
    np.save(os.path.join(out_dir, "test_labels.npy"), test_labels)

    meta = {
        "adapt_ratio": adapt_ratio,
        "train_size": len(train_windows),
        "adapt_size": len(adapt_windows),
        "test_size": len(test_windows),
        "adapt_fall": int(adapt_labels.sum()),
        "adapt_nonfall": int(len(adapt_labels) - adapt_labels.sum()),
        "test_fall": int(test_labels.sum()),
        "test_nonfall": int(len(test_labels) - test_labels.sum()),
    }
    with open(os.path.join(out_dir, "metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{out_dir}")
    print(f"  Train: {len(train_windows)} (fall={train_labels.sum()}, nonfall={train_labels.sum()})")
    print(f"  Adapt: {len(adapt_windows)} (fall={meta['adapt_fall']}, nonfall={meta['adapt_nonfall']})")
    print(f"  Test:  {len(test_windows)} (fall={meta['test_fall']}, nonfall={meta['test_nonfall']})")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    ratios = [0.15, 0.20, 0.25]  # 15%, 20%, 25% - fill gap
    for ratio in ratios:
        out_dir = os.path.join(DATA_ROOT, f"processed_adaptive_{int(ratio*100)}pct")
        create_split(ratio, out_dir)