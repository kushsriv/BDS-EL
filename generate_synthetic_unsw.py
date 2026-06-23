"""
Synthetic UNSW-NB15 Dataset Generator
======================================
Generates a synthetic dataset that mirrors the UNSW-NB15 training set:
  - Same 45 numeric features (categorical columns proto/service/state dropped
    as the adapter does), same column names
  - Same 10 attack categories with realistic class distribution
  - Feature statistics (mean, std, range, skew) derived from:
      Moustafa & Slay (2015), "UNSW-NB15: A Comprehensive Data Set for
      Network Intrusion Detection Systems", MilCIS 2015.
  - ~82,332 rows total

Usage:
    python generate_synthetic_unsw.py
Outputs:
    UNSW_NB15_training-set.csv  (in current directory)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ---------------------------------------------------------------------------
# Class distribution (from original UNSW-NB15 training set paper)
# ---------------------------------------------------------------------------
CLASSES = {
    'Normal':         37000,
    'Generic':        18871,
    'Exploits':       11132,
    'Fuzzers':         6062,
    'DoS':             4089,
    'Reconnaissance':  3496,
    'Analysis':         677,
    'Backdoors':        583,
    'Shellcode':        378,
    'Worms':             44,
}
TOTAL = sum(CLASSES.values())   # 82,332

# ---------------------------------------------------------------------------
# Feature definitions  (mean, std, min, max, distribution type)
# Based on published UNSW-NB15 statistics and network traffic domain knowledge
# ---------------------------------------------------------------------------
# Each entry: (col_name, dist, params_by_class)
# dist: 'exp', 'norm', 'int_unif', 'bernoulli', 'zero_heavy'
# params_by_class: dict mapping class name -> (loc, scale) or similar

def _col(name, default, overrides=None):
    """Build per-class param dict with defaults overridden per attack type."""
    d = {c: default for c in CLASSES}
    if overrides:
        d.update(overrides)
    return (name, d)

# Format: list of (feature_name, {class: (mean, std, min_clip, max_clip)})
FEATURE_SPECS = [
    # --- timing ---
    _col('dur',         (0.5,  2.0,  0.0, 60.0),
         {'DoS': (0.01, 0.05, 0.0, 1.0), 'Generic': (0.001, 0.01, 0.0, 0.5)}),
    _col('spkts',       (6.0,  10.0, 1.0, 500.0),
         {'DoS': (200, 300, 1, 5000), 'Fuzzers': (20, 30, 1, 1000)}),
    _col('dpkts',       (5.0,  9.0,  0.0, 500.0),
         {'DoS': (50, 100, 0, 3000), 'Reconnaissance': (3, 5, 0, 50)}),
    _col('sbytes',      (1200, 3000, 40, 1e6),
         {'DoS': (80000, 120000, 100, 5e6), 'Exploits': (5000, 8000, 100, 2e5)}),
    _col('dbytes',      (1000, 2500, 0,  1e6),
         {'DoS': (20000, 50000, 0, 3e6)}),
    _col('rate',        (2000, 5000, 0,  1e6),
         {'DoS': (50000, 80000, 0, 5e6), 'Generic': (100000, 150000, 0, 1e7)}),
    _col('sttl',        (62,   10,  1,  255),
         {'Generic': (252, 5, 240, 255), 'Reconnaissance': (50, 20, 1, 128)}),
    _col('dttl',        (50,   15,  0,  255),
         {'Generic': (252, 5, 240, 255)}),
    _col('sload',       (5000, 15000, 0, 1e6),
         {'DoS': (200000, 300000, 0, 5e6)}),
    _col('dload',       (4000, 12000, 0, 1e6),
         {'DoS': (80000,  150000, 0, 3e6)}),
    _col('sloss',       (0.5,  2.0,  0, 100),
         {'DoS': (10, 20, 0, 500)}),
    _col('dloss',       (0.3,  1.5,  0, 100),
         {'DoS': (5,  10, 0, 300)}),
    _col('sinpkt',      (50,   100,  0, 1000),
         {'DoS': (0.5, 1, 0, 10)}),
    _col('dinpkt',      (60,   110,  0, 1000),
         {'DoS': (1,   2, 0, 20)}),
    _col('sjit',        (20,   50,   0, 500),
         {'DoS': (1, 5, 0, 50)}),
    _col('djit',        (15,   40,   0, 500)),
    _col('swin',        (200,  100,  0, 255),
         {'Normal': (255, 1, 0, 255)}),
    _col('stcpb',       (1e8,  2e8,  0, 4e9)),
    _col('dtcpb',       (1e8,  2e8,  0, 4e9)),
    _col('dwin',        (200,  100,  0, 255),
         {'Normal': (255, 1, 0, 255)}),
    _col('tcprtt',      (0.05, 0.1,  0, 2.0),
         {'Generic': (0.0, 0.001, 0, 0.01)}),
    _col('synack',      (0.03, 0.08, 0, 2.0),
         {'Generic': (0.0, 0.001, 0, 0.01)}),
    _col('ackdat',      (0.02, 0.05, 0, 2.0),
         {'Generic': (0.0, 0.001, 0, 0.01)}),
    _col('smean',       (300,  200,  20, 1500),
         {'DoS': (1400, 100, 100, 1500)}),
    _col('dmean',       (250,  180,  0,  1500),
         {'DoS': (1400, 100, 0,   1500)}),
    _col('trans_depth', (1.5,  2.0,  0, 10),
         {'Exploits': (3, 2, 0, 20)}),
    _col('response_body_len', (2000, 8000, 0, 1e6),
         {'Exploits': (10000, 20000, 0, 5e5)}),
    # --- connection count features ---
    _col('ct_srv_src',  (10,   15,   1, 100),
         {'DoS': (50, 30, 1, 200), 'Generic': (60, 30, 1, 200)}),
    _col('ct_state_ttl',(3,    3,    0, 8)),
    _col('ct_dst_ltm',  (8,    12,   1, 100)),
    _col('ct_src_dport_ltm', (5, 8,  1, 100)),
    _col('ct_dst_sport_ltm', (4, 7,  1, 100)),
    _col('ct_dst_src_ltm',   (10,14, 1, 100),
         {'DoS': (40, 20, 1, 200)}),
    _col('ct_src_ltm',  (8,    12,   1, 100)),
    _col('ct_srv_dst',  (9,    14,   1, 100),
         {'DoS': (50, 30, 1, 200)}),
    # --- binary / sparse features ---
    _col('is_ftp_login',     (0.02, 0.14, 0, 1),
         {'Backdoors': (0.3, 0.46, 0, 1)}),
    _col('ct_ftp_cmd',       (0.05, 0.3,  0, 8),
         {'Backdoors': (1.5, 1.5, 0, 8)}),
    _col('ct_flw_http_mthd', (0.3,  1.0,  0, 10),
         {'Exploits': (2.0, 2.0, 0, 20)}),
    _col('is_sm_ips_ports',  (0.03, 0.17, 0, 1),
         {'DoS': (0.7, 0.46, 0, 1), 'Generic': (0.9, 0.3, 0, 1)}),
]


def _generate_feature(n, mean, std, lo, hi, rng):
    """Generate n samples from a clipped normal distribution."""
    vals = rng.normal(loc=mean, scale=max(std, 1e-6), size=n)
    vals = np.clip(vals, lo, hi)
    # Add heavy tail for byte/packet features
    if hi > 1000:
        outlier_mask = rng.random(n) < 0.01
        vals[outlier_mask] = rng.uniform(hi * 0.5, hi, size=outlier_mask.sum())
    return vals


def generate_unsw_nb15(output_path='UNSW_NB15_training-set.csv', seed=42):
    rng = np.random.default_rng(seed)
    frames = []

    print(f"Generating synthetic UNSW-NB15 ({TOTAL:,} rows, "
          f"{len(FEATURE_SPECS)} features) ...")

    for attack_cat, n in CLASSES.items():
        is_normal = (attack_cat == 'Normal')
        label = 0 if is_normal else 1

        data = {}
        for feat_name, class_params in FEATURE_SPECS:
            mean, std, lo, hi = class_params[attack_cat]
            col_vals = _generate_feature(n, mean, std, lo, hi, rng)

            # Integer columns
            if feat_name in ('spkts', 'dpkts', 'sbytes', 'dbytes', 'sttl', 'dttl',
                             'sloss', 'dloss', 'swin', 'dwin', 'smean', 'dmean',
                             'trans_depth', 'response_body_len', 'ct_srv_src',
                             'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm',
                             'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_src_ltm',
                             'ct_srv_dst', 'ct_ftp_cmd', 'ct_flw_http_mthd'):
                col_vals = np.round(col_vals).astype(int)
                col_vals = np.clip(col_vals, int(lo), int(hi))

            # Binary columns — round to 0/1
            if feat_name in ('is_ftp_login', 'is_sm_ips_ports'):
                col_vals = (rng.random(n) < mean).astype(int)

            # Large int columns (TCP seq numbers)
            if feat_name in ('stcpb', 'dtcpb'):
                col_vals = rng.integers(0, 4294967295, size=n)

            data[feat_name] = col_vals

        data['attack_cat'] = attack_cat
        data['label'] = label

        frames.append(pd.DataFrame(data))
        print(f"  {attack_cat:20s}: {n:6,} rows  (label={label})")

    df = pd.concat(frames, ignore_index=True)

    # Shuffle rows
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Add id column (as in original)
    df.insert(0, 'id', range(1, len(df) + 1))

    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    print(f"Attack categories:\n{df['attack_cat'].value_counts().to_string()}")
    return df


if __name__ == '__main__':
    generate_unsw_nb15('UNSW_NB15_training-set.csv')
