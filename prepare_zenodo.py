"""
Prepare BITS Pilani (Zenodo) fall detection dataset for training.
Reads CSV files (long format: t,x,y,z,a,sensor_type) per user,
extracts acc+gyro+mgm channels, resamples to 50Hz, creates 128-sample windows,
saves in the same npy format as HIFD processed_* directories.
"""
import os, sys, glob, json, random
import numpy as np

DATA_ROOT = r"D:\hd_imu\zenodo_full\Dataset"
OUT_ROOT = r"D:\hd_imu\zenodo_processed_aug"
WINDOW_SIZE = 128  # 2.56 seconds at 50Hz
TARGET_RATE = 50   # Hz (match HIFD)
N_CHANNELS = 11
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# Test subjects (last 10 out of 41)
TEST_USERS = list(range(32, 42))  # user32 to user41

# ─── Helpers ──────────────────────────────────────────────────────────────

def read_zenodo_csv(filepath):
    """Read single CSV, return dict of {sensor: np.array(N, 3)}."""
    data = {}
    with open(filepath) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            sensor = parts[5].strip()
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except (ValueError, IndexError):
                continue
            if sensor not in data:
                data[sensor] = []
            data[sensor].append([x, y, z])
    return {k: np.array(v, dtype=np.float64) for k, v in data.items()}


def sensor_data_to_11ch(data):
    """
    Convert sensor dict to (T_acc, 11) matrix.
    Channels 0-2: acc_xyz
    Channels 3-5: gyro_xyz (interpolated to acc time axis)
    Channels 6-8: mgm_xyz (interpolated to acc time axis)
    Channels 9-10: zeros (no ECG/HR available in Zenodo)
    """
    if 'acc' not in data:
        return None
    acc = data['acc']                # (N_acc, 3)
    T = len(acc)
    feat = np.zeros((T, N_CHANNELS), dtype=np.float64)

    # acc
    feat[:, 0:3] = acc

    # gyro (interpolate to acc length)
    if 'gyro' in data:
        g = data['gyro']
        Tg = len(g)
        if Tg >= 2:
            idx = np.linspace(0, Tg - 1, T)
            for c in range(3):
                feat[:, 3 + c] = np.interp(idx, np.arange(Tg), g[:, c])
        elif Tg == 1:
            feat[:, 3:6] = g[0]

    # mgm
    if 'mgm' in data:
        m = data['mgm']
        Tm = len(m)
        if Tm >= 2:
            idx = np.linspace(0, Tm - 1, T)
            for c in range(3):
                feat[:, 6 + c] = np.interp(idx, np.arange(Tm), m[:, c])
        elif Tm == 1:
            feat[:, 6:9] = m[0]

    return feat  # (T, 11)


def resample_to_50hz(feat, native_rate=20, target_rate=50):
    """Resample (T, 11) to target_rate from native_rate via linear interpolation.
    Preserves proportional length instead of forcing to fixed 128.
    Returns (T_new, 11) where T_new = ceil(T * target_rate / native_rate).
    """
    T = feat.shape[0]
    if T < 1:
        return None
    if T == 1:
        n = max(WINDOW_SIZE, int(1 * target_rate / native_rate))
        return np.tile(feat, (n, 1))

    T_new = max(WINDOW_SIZE, int(T * target_rate / native_rate))
    idx_old = np.arange(T)
    idx_new = np.linspace(0, T - 1, T_new)
    out = np.zeros((T_new, N_CHANNELS), dtype=np.float64)
    for c in range(N_CHANNELS):
        out[:, c] = np.interp(idx_new, idx_old, feat[:, c])
    return out


def extract_windows(feat, stride=64):
    """
    Given (T, 11) signal (already at 50Hz), generate windows of WINDOW_SIZE.
    If T < WINDOW_SIZE, tile to fill.
    Returns list of (WINDOW_SIZE, 11) arrays — ALL windows from the signal.
    """
    windows = []
    if feat.shape[0] < WINDOW_SIZE:
        repeats = (WINDOW_SIZE + feat.shape[0] - 1) // feat.shape[0]
        w = np.tile(feat, (repeats, 1))[:WINDOW_SIZE]
        windows.append(w)
    else:
        for start in range(0, feat.shape[0] - WINDOW_SIZE + 1, stride):
            windows.append(feat[start:start + WINDOW_SIZE])
    return windows


# ─── Main processing ─────────────────────────────────────────────────────

def process_all_users():
    """Process all 41 users, return {user_id: {'windows': [...], 'labels': [...]}}.
    Uses proportional resampling to 50Hz + sliding windows.
    
    Strategy: take up to 3 windows per event (first, middle, last) to
    capture relevant signal without introducing excessive label noise
    from long fall recordings where only a small portion contains the fall.
    """
    user_data = {}

    for user_id in range(1, 42):
        windows_list = []
        labels_list = []

        for cat, label in [('fall', 1), ('adl', 0)]:
            cat_dir = os.path.join(DATA_ROOT, cat, f"user{user_id}")
            if not os.path.isdir(cat_dir):
                continue
            for fname in sorted(os.listdir(cat_dir)):
                if not fname.endswith('.csv'):
                    continue
                data = read_zenodo_csv(os.path.join(cat_dir, fname))
                feat = sensor_data_to_11ch(data)
                if feat is None:
                    continue
                # Resample to 50Hz preserving proportional length
                feat_50hz = resample_to_50hz(feat, native_rate=20, target_rate=50)
                if feat_50hz is None:
                    continue
                wins = extract_windows(feat_50hz, stride=64)
                
                # Take up to 3 representative windows: first, middle, last
                # This avoids label noise from long fall recordings
                if len(wins) <= 3:
                    selected = wins
                else:
                    mid = len(wins) // 2
                    selected = [wins[0], wins[mid], wins[-1]]
                    # Also take a few more if it's very long (10+ windows)
                    if len(wins) >= 10:
                        selected.append(wins[len(wins)//4])
                    if len(wins) >= 20:
                        selected.append(wins[3*len(wins)//4])
                
                for w in selected:
                    windows_list.append(w)
                    labels_list.append(label)

        if len(windows_list) > 0:
            user_data[user_id] = {
                'windows': np.array(windows_list),
                'labels': np.array(labels_list, dtype=np.int64)
            }
            n_fall = (user_data[user_id]['labels'] == 1).sum()
            n_adl = (user_data[user_id]['labels'] == 0).sum()
            print(f"  user{user_id:2d}: {len(windows_list):3d} windows ({int(n_fall)} fall, {int(n_adl)} adl)")
        else:
            print(f"  user{user_id:2d}: NO DATA")

    return user_data


def save_splits(user_data, out_dir, adapt_ratio=0.30):
    """
    Save train/test/adapt splits matching the HIFD format.
    - Train: all non-test users (100% labeled)
    - Test users: stratified split into adapt (adapt_ratio) and test (1-adapt_ratio)
    """
    os.makedirs(out_dir, exist_ok=True)

    train_wins = []
    train_labs = []

    adapt_wins = []
    adapt_labs = []
    test_wins = []
    test_labs = []

    for uid, udata in user_data.items():
        wins = udata['windows']
        labs = udata['labels']

        if uid in TEST_USERS:
            # Stratified split per user
            fall_idx = np.where(labs == 1)[0]
            nonfall_idx = np.where(labs == 0)[0]
            np.random.shuffle(fall_idx)
            np.random.shuffle(nonfall_idx)

            n_fall_adapt = max(1, int(len(fall_idx) * adapt_ratio))
            n_fall_test = len(fall_idx) - n_fall_adapt
            n_nf_adapt  = max(1, int(len(nonfall_idx) * adapt_ratio))
            n_nf_test   = len(nonfall_idx) - n_nf_adapt

            adapt_fall = fall_idx[:n_fall_adapt]
            test_fall  = fall_idx[n_fall_adapt:]
            adapt_nf   = nonfall_idx[:n_nf_adapt]
            test_nf    = nonfall_idx[n_nf_adapt:]

            adapt_all = np.concatenate([adapt_fall, adapt_nf])
            test_all  = np.concatenate([test_fall, test_nf])
            np.random.shuffle(adapt_all)
            np.random.shuffle(test_all)

            adapt_wins.append(wins[adapt_all])
            adapt_labs.append(labs[adapt_all])
            test_wins.append(wins[test_all])
            test_labs.append(labs[test_all])
        else:
            train_wins.append(wins)
            train_labs.append(labs)

    # Concatenate
    train_wins = np.concatenate(train_wins) if train_wins else np.array([])
    train_labs = np.concatenate(train_labs) if train_labs else np.array([])
    adapt_wins = np.concatenate(adapt_wins) if adapt_wins else np.array([])
    adapt_labs = np.concatenate(adapt_labs) if adapt_labs else np.array([])
    test_wins  = np.concatenate(test_wins)  if test_wins  else np.array([])
    test_labs  = np.concatenate(test_labs)  if test_labs  else np.array([])

    # Save
    np.save(os.path.join(out_dir, "train_labeled_samples.npy"), train_wins)
    np.save(os.path.join(out_dir, "train_labeled_labels.npy"),  train_labs)
    np.save(os.path.join(out_dir, "train_unlabeled_samples.npy"), np.zeros((0, WINDOW_SIZE, N_CHANNELS), dtype=np.float64))
    np.save(os.path.join(out_dir, "adapt_samples.npy"),  adapt_wins)
    np.save(os.path.join(out_dir, "adapt_labels.npy"),   adapt_labs)
    np.save(os.path.join(out_dir, "test_samples.npy"),   test_wins)
    np.save(os.path.join(out_dir, "test_labels.npy"),    test_labs)

    # Summary
    print(f"\n=== Split Summary ===")
    print(f"Train: {len(train_wins)} ({int(train_labs.sum())} fall / {int(len(train_labs) - train_labs.sum())} nonfall)")
    print(f"Adapt: {len(adapt_wins)} ({int(adapt_labs.sum())} fall / {int(len(adapt_labs) - adapt_labs.sum())} nonfall)")
    print(f"Test:  {len(test_wins)} ({int(test_labs.sum())} fall / {int(len(test_labs) - test_labs.sum())} nonfall)")

    # Save metadata
    meta = {
        "dataset": "BITS Pilani Zenodo Fall Detection",
        "source": str(DATA_ROOT),
        "num_users": len(user_data),
        "test_users": TEST_USERS,
        "adapt_ratio": adapt_ratio,
        "window_size": WINDOW_SIZE,
        "num_channels": N_CHANNELS,
        "target_rate_hz": TARGET_RATE,
        "train": {"count": len(train_wins), "fall": int(train_labs.sum())},
        "adapt": {"count": len(adapt_wins), "fall": int(adapt_labs.sum())},
        "test":  {"count": len(test_wins),  "fall": int(test_labs.sum())},
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to: {out_dir}")
    return meta


if __name__ == "__main__":
    print("=" * 60)
    print("Processing BITS Pilani Zenodo dataset")
    print("=" * 60)

    print("\n[Step 1] Reading and processing all CSV files...")
    user_data = process_all_users()
    print(f"\nProcessed {len(user_data)} users")

    print("\n[Step 2] Creating adaptive data splits...")
    # Full supervised (100% labeled in train) + adaptive splits
    out_dir = os.path.join(OUT_ROOT, "zenodo_30pct")
    meta = save_splits(user_data, out_dir, adapt_ratio=0.30)

    print("\n[Step 3] Creating additional splits for ablation...")
    for ratio in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25]:
        ratio_out = os.path.join(OUT_ROOT, f"zenodo_{int(ratio*100)}pct")
        meta_r = save_splits(user_data, ratio_out, adapt_ratio=ratio)

    print("\nDone! All splits created.")