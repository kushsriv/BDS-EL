"""
Multi-Dataset Support for DP-IDS Framework
============================================

Provides loaders and preprocessors for multiple IDS benchmark datasets.
Drop the CSV into the project directory and run main.py with --dataset <name>.

Supported Datasets
------------------
1. NSL-KDD     — already supported (KDDTrain+.csv, 125,973 rows, 41 features)
2. UNSW-NB15   — University of New South Wales network benchmark
                 Download: https://research.unsw.edu.au/projects/unsw-nb15-dataset
                 File needed: UNSW_NB15_training-set.csv  (82,332 rows, 49 features)
3. CIC-IDS-2017 — Canadian Institute for Cybersecurity, 2017
                 Download: https://www.unb.ca/cic/datasets/ids-2017.html
                 File needed: MachineLearningCVE/ (multiple CSVs, auto-merged)

Usage
-----
    from multi_dataset import load_dataset

    df, label_col, dataset_info = load_dataset('UNSW_NB15_training-set.csv')
    # Then pass to prepare_kdd_data() in dp_ml.py — same interface

Architecture
------------
Each dataset adapter normalises column names and label values so that:
  - label_col always contains string labels ('normal' / attack names)
  - All feature columns are numeric (float-coercible)
  - The same DP modules run unchanged across all datasets
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    'NSL-KDD': {
        'filenames': ['KDDTrain+.csv', 'KDDTrain+.txt'],
        'label_column_pattern': 'dst_host_srv_rerror_rate',  # shifted column
        'normal_label': 'normal',
        'n_features': 41,
        'n_rows_approx': 125973,
        'attack_categories': {
            'Normal': ['normal'],
            'DoS': ['neptune', 'back', 'teardrop', 'smurf', 'pod', 'land',
                    'apache2', 'udpstorm', 'processtable', 'mailbomb'],
            'Probe': ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint'],
            'R2L': ['warezclient', 'warezmaster', 'imap', 'ftp_write',
                    'guess_passwd', 'phf', 'multihop', 'spy', 'named',
                    'snmpgetattack', 'snmpguess', 'xsnoop', 'xlock',
                    'sendmail', 'httptunnel'],
            'U2R': ['buffer_overflow', 'loadmodule', 'perl', 'rootkit',
                    'ps', 'sqlattack', 'xterm'],
        },
        'reference': 'Tavallaee et al. (2009), Canadian Institute for Cybersecurity',
    },
    'UNSW-NB15': {
        'filenames': ['UNSW_NB15_training-set.csv', 'UNSW-NB15_1.csv'],
        'label_column': 'label',      # binary: 0=normal, 1=attack
        'category_column': 'attack_cat',
        'normal_label': 'Normal',
        'n_features': 49,
        'n_rows_approx': 82332,
        'attack_categories': {
            'Normal': ['Normal', ''],
            'Fuzzers': ['Fuzzers'],
            'Analysis': ['Analysis'],
            'Backdoors': ['Backdoors'],
            'DoS': ['DoS'],
            'Exploits': ['Exploits'],
            'Generic': ['Generic'],
            'Reconnaissance': ['Reconnaissance'],
            'Shellcode': ['Shellcode'],
            'Worms': ['Worms'],
        },
        'drop_columns': ['id', 'proto', 'service', 'state', 'attack_cat'],
        'reference': 'Moustafa & Slay (2015), University of New South Wales',
        'download_url': 'https://research.unsw.edu.au/projects/unsw-nb15-dataset',
    },
    'CIC-IDS-2017': {
        'filenames': ['Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
                      'MachineLearningCVE/'],
        'label_column': ' Label',    # note: leading space in original
        'normal_label': 'BENIGN',
        'n_features': 78,
        'n_rows_approx': 2830743,
        'attack_categories': {
            'Normal': ['BENIGN'],
            'DDoS': ['DDoS'],
            'PortScan': ['PortScan'],
            'Bot': ['Bot'],
            'Web': ['Web Attack – Brute Force', 'Web Attack – XSS',
                    'Web Attack – Sql Injection'],
            'FTP-Patator': ['FTP-Patator'],
            'SSH-Patator': ['SSH-Patator'],
            'DoS': ['DoS slowloris', 'DoS Slowhttptest', 'DoS Hulk',
                    'DoS GoldenEye', 'Heartbleed'],
            'Infiltration': ['Infiltration'],
        },
        'drop_columns': ['Flow ID', ' Source IP', ' Destination IP',
                         ' Timestamp', 'Fwd Header Length.1'],
        'reference': 'Sharafaldin et al. (2018), Canadian Institute for Cybersecurity',
        'download_url': 'https://www.unb.ca/cic/datasets/ids-2017.html',
    },
}


# ---------------------------------------------------------------------------
# Auto-detect which dataset a file belongs to
# ---------------------------------------------------------------------------

def detect_dataset_type(filepath):
    """Return the dataset name ('NSL-KDD', 'UNSW-NB15', etc.) from the filename."""
    fname = os.path.basename(filepath).lower()
    if 'kdd' in fname or 'nsl' in fname:
        return 'NSL-KDD'
    if 'unsw' in fname or 'nb15' in fname:
        return 'UNSW-NB15'
    if 'cic' in fname or 'iscx' in fname or '2017' in fname:
        return 'CIC-IDS-2017'
    return 'UNKNOWN'


# ---------------------------------------------------------------------------
# NSL-KDD loader (existing, kept for consistency)
# ---------------------------------------------------------------------------

def load_nslkdd(filepath):
    """Load NSL-KDD CSV, detect label column, return (df, label_col, info)."""
    df = pd.read_csv(filepath)
    label_col = None
    for col in df.columns:
        dtype = df[col].dtype
        is_str = (dtype == object or
                  (hasattr(dtype, 'name') and dtype.name in ('str', 'string', 'object')))
        if not is_str:
            continue
        sample = df[col].dropna().head(50).astype(str).str.strip().str.lower()
        if 'normal' in sample.values:
            label_col = col
            break

    info = {
        'dataset': 'NSL-KDD',
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'label_col': label_col,
        'attack_distribution': df[label_col].value_counts().to_dict() if label_col else {},
        **DATASET_REGISTRY['NSL-KDD'],
    }
    return df, label_col, info


# ---------------------------------------------------------------------------
# UNSW-NB15 loader and preprocessor
# ---------------------------------------------------------------------------

def load_unsw_nb15(filepath):
    """
    Load and preprocess UNSW-NB15 training set CSV.

    The raw CSV has 49 features + label column + attack_cat column.
    We:
      1. Drop non-numeric categorical columns (proto, service, state)
      2. Normalise the label column to string format ('Normal' / attack names)
      3. Return (df, label_col, info) with the same interface as NSL-KDD

    Parameters
    ----------
    filepath : path to UNSW_NB15_training-set.csv

    Returns
    -------
    df        : cleaned DataFrame (numeric features + string label column)
    label_col : column name containing string attack labels
    info      : dataset metadata dict
    """
    meta = DATASET_REGISTRY['UNSW-NB15']

    df = pd.read_csv(filepath, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # Use attack_cat as the string label if present; else decode binary label
    label_col = None
    if 'attack_cat' in df.columns:
        df['attack_cat'] = df['attack_cat'].astype(str).str.strip()
        # Fill empty attack_cat (= normal traffic) with 'Normal'
        df.loc[df['attack_cat'].isin(['', 'nan', 'NaN', 'none', 'None']), 'attack_cat'] = 'Normal'
        # Binary label=0 rows should be Normal
        if 'label' in df.columns:
            df.loc[df['label'] == 0, 'attack_cat'] = 'Normal'
        label_col = 'attack_cat'
    elif 'label' in df.columns:
        # Binary label — map to strings
        df['label_str'] = df['label'].apply(lambda x: 'Normal' if int(x) == 0 else 'Attack')
        label_col = 'label_str'

    # Drop non-numeric and ID columns
    drop_cols = [c for c in meta.get('drop_columns', []) if c in df.columns]
    drop_cols += ['id', 'label']  # always drop binary label and id
    drop_cols = [c for c in drop_cols if c != label_col]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Coerce all non-label columns to numeric
    for col in df.columns:
        if col == label_col:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')

    info = {
        'dataset': 'UNSW-NB15',
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'label_col': label_col,
        'attack_distribution': df[label_col].value_counts().to_dict() if label_col else {},
        **meta,
    }
    return df, label_col, info


# ---------------------------------------------------------------------------
# CIC-IDS-2017 loader
# ---------------------------------------------------------------------------

def load_cicids2017(filepath_or_dir):
    """
    Load CIC-IDS-2017 dataset from a single CSV or a directory of CSVs.

    CIC-IDS-2017 has 78 features. Several columns have inf/NaN due to
    divide-by-zero in the original feature extraction.

    Parameters
    ----------
    filepath_or_dir : path to a single CSV or directory containing CSVs

    Returns
    -------
    df, label_col, info
    """
    meta = DATASET_REGISTRY['CIC-IDS-2017']

    if os.path.isdir(filepath_or_dir):
        dfs = []
        for fname in os.listdir(filepath_or_dir):
            if fname.endswith('.csv'):
                try:
                    dfs.append(pd.read_csv(
                        os.path.join(filepath_or_dir, fname),
                        low_memory=False, encoding='utf-8-sig'
                    ))
                except Exception:
                    pass
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    else:
        df = pd.read_csv(filepath_or_dir, low_memory=False, encoding='utf-8-sig')

    df.columns = [c.strip() for c in df.columns]

    label_col = 'Label' if 'Label' in df.columns else ' Label'
    if label_col not in df.columns:
        label_col = None

    # Drop non-numeric/ID columns
    drop_cols = [c for c in meta.get('drop_columns', []) if c in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Replace inf values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Coerce to numeric
    for col in df.columns:
        if col == label_col:
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')

    info = {
        'dataset': 'CIC-IDS-2017',
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'label_col': label_col,
        'attack_distribution': df[label_col].value_counts().to_dict() if label_col else {},
        **meta,
    }
    return df, label_col, info


# ---------------------------------------------------------------------------
# Universal loader (auto-detects dataset type)
# ---------------------------------------------------------------------------

def load_dataset(filepath):
    """
    Auto-detect and load any supported IDS dataset.

    Parameters
    ----------
    filepath : path to the dataset CSV (or directory for CIC-IDS-2017)

    Returns
    -------
    df        : pandas DataFrame
    label_col : column containing string attack labels
    info      : dict with dataset metadata

    Example
    -------
    >>> df, label_col, info = load_dataset('UNSW_NB15_training-set.csv')
    >>> print(info['dataset'], info['n_rows'], info['attack_distribution'])
    """
    dtype = detect_dataset_type(filepath)

    if dtype == 'NSL-KDD':
        return load_nslkdd(filepath)
    elif dtype == 'UNSW-NB15':
        return load_unsw_nb15(filepath)
    elif dtype == 'CIC-IDS-2017':
        return load_cicids2017(filepath)
    else:
        # Generic fallback: try NSL-KDD loader
        print(f'[multi_dataset] Unknown dataset type for {filepath}; trying generic loader.')
        return load_nslkdd(filepath)


# ---------------------------------------------------------------------------
# Dataset info CLI
# ---------------------------------------------------------------------------

def print_dataset_info():
    """Print download URLs and expected file names for all supported datasets."""
    print('\nSupported IDS Datasets for DP Research Framework')
    print('=' * 60)
    for name, meta in DATASET_REGISTRY.items():
        print(f'\n{name}')
        print(f'  Files: {meta["filenames"][0]}')
        print(f'  Rows: ~{meta["n_rows_approx"]:,}   Features: {meta["n_features"]}')
        if 'download_url' in meta:
            print(f'  Download: {meta["download_url"]}')
        print(f'  Reference: {meta["reference"]}')
    print()


if __name__ == '__main__':
    print_dataset_info()
