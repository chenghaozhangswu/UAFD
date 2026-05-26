"""Normalize Zenodo AUG to Z-score using its training set stats"""
import numpy as np, os

SRC = r'D:\hd_imu\zenodo_processed_aug'
DST = r'D:\hd_imu\zenodo_processed_aug_norm'

# Compute stats from 30pct training set
z = np.load(f'{SRC}\\zenodo_30pct\\train_labeled_samples.npy')
z_mean = z.mean(axis=(0,1), keepdims=True)  # (1,1,11)
z_std  = z.std(axis=(0,1), keepdims=True)
z_std[z_std == 0] = 1  # avoid div by 0 for ch9-10

print('Normalizing with:')
for c in range(11):
    print(f'  ch{c:2d}: mean={z_mean[0,0,c]:+.4f} std={z_std[0,0,c]:.4f}')

ratios = [1, 5, 10, 15, 20, 25, 30]
for r in ratios:
    sd = f'{SRC}\\zenodo_{r}pct'
    dd = f'{DST}\\zenodo_{r}pct'
    os.makedirs(dd, exist_ok=True)
    for fname in ['train_labeled_samples.npy', 'train_labeled_labels.npy',
                  'train_unlabeled_samples.npy', 'adapt_samples.npy',
                  'adapt_labels.npy', 'test_samples.npy', 'test_labels.npy']:
        src_path = f'{sd}\\{fname}'
        if not os.path.exists(src_path):
            continue
        arr = np.load(src_path)
        if 'samples' in fname and arr.ndim == 3:
            arr = (arr - z_mean) / z_std
        np.save(f'{dd}\\{fname}', arr)
        print(f'  {r:2d}% {fname}: {arr.shape}')

# Copy metadata
import json
with open(f'{SRC}\\zenodo_30pct\\metadata.json') as f:
    meta = json.load(f)
meta['normalized'] = True
meta['norm_mean'] = [float(x) for x in z_mean[0,0,:]]
meta['norm_std'] = [float(x) for x in z_std[0,0,:]]
with open(f'{DST}\\zenodo_30pct\\metadata.json', 'w') as f:
    json.dump(meta, f, indent=2)

print('Done!')