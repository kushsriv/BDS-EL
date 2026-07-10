"""
Spark-Distributed DP Pipeline
==============================

Implements the full privacy-preserving IDS pipeline on Apache Spark:

  Raw Logs (HDFS / local CSV)
        ↓
  Spark ETL  (feature casting, null-drop, label normalisation)
        ↓
  compute_mi_spark()          ← MI per feature via Spark groupBy + RDD reduce
        ↓
  allocate_budget()           ← per-feature ε array  (driver-side, tiny)
        ↓
  apply_noise_spark()         ← Laplace noise injection via Spark UDF / vectorised
        ↓
  Noisy Spark DataFrame  →  toPandas()  →  train ML model

All heavy computation (MI, noise) runs distributed across Spark workers.
Falls back to weighted_budget.py single-machine functions when Spark is absent.

Usage
-----
    from spark_pipeline import run_spark_dp_pipeline

    result = run_spark_dp_pipeline(
        spark,
        data_path   = 'hdfs:///data/network_logs.csv',   # or local path
        label_col   = 'attack_cat',
        total_eps   = 1.0,
        method      = 'mi',          # 'mi' | 'variance' | 'snr' | 'uniform'
        mechanism   = 'laplace',     # 'laplace' | 'gaussian'
        n_bins      = 20,
        clip_pct    = 99.0,
        delta       = 1e-5,
    )
    # result['X_noisy']   — pandas DataFrame with DP noise applied
    # result['weights']   — per-feature importance weights
    # result['epsilons']  — per-feature ε allocations
    # result['features']  — feature column names
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spark MI computation
# ---------------------------------------------------------------------------

def compute_mi_spark(df_spark, feature_cols, label_col, n_bins=20):
    """
    Compute Mutual Information I(X_j; Y) for each feature using Spark.

    Strategy
    --------
    For each feature column:
      1. Cast to double, drop nulls
      2. Discretise into n_bins equal-width bins using Spark SQL
      3. GroupBy (bin, label) → joint count table  P(X=b, Y=c)
      4. Collect the small count table to the driver
      5. Compute MI sum locally (O(n_bins × n_classes) — tiny)

    This scales to billions of rows because the heavy groupBy runs
    distributed across Spark workers. Only the n_bins×n_classes count
    table (at most 20×10 = 200 numbers) is sent to the driver.

    Parameters
    ----------
    df_spark    : Spark DataFrame
    feature_cols: list of feature column names
    label_col   : name of the label column (string labels)
    n_bins      : number of discretisation bins per feature

    Returns
    -------
    mi_scores : dict  {feature_name: MI_value_in_nats}
    """
    try:
        from pyspark.sql import functions as F
    except ImportError:
        raise RuntimeError('PySpark not available — use compute_feature_importance() from weighted_budget.py')

    mi_scores = {}
    n_total = df_spark.count()

    for feat in feature_cols:
        try:
            # Cast feature to double, drop rows where feature or label is null
            df_f = (df_spark
                    .select(F.col(feat).cast('double').alias('x'),
                            F.col(label_col).alias('y'))
                    .dropna())

            n_f = df_f.count()
            if n_f == 0:
                mi_scores[feat] = 0.0
                continue

            # Compute min/max on Spark for bin edges
            stats = df_f.agg(F.min('x').alias('lo'), F.max('x').alias('hi')).collect()[0]
            lo, hi = float(stats['lo']), float(stats['hi'])
            if hi == lo:
                mi_scores[feat] = 0.0
                continue

            bin_width = (hi - lo) / n_bins

            # Assign each row to a bin index (0 to n_bins-1)
            df_binned = df_f.withColumn(
                'bin',
                F.least(
                    F.lit(n_bins - 1),
                    F.floor((F.col('x') - F.lit(lo)) / F.lit(bin_width)).cast('int')
                )
            )

            # Joint count: P(bin=b, label=c) ∝ count(b, c)
            joint_counts = (df_binned
                            .groupBy('bin', 'y')
                            .count()
                            .collect())

            # Build count table on driver (tiny: n_bins × n_classes)
            from collections import defaultdict
            bin_label_count = defaultdict(int)
            bin_count = defaultdict(int)
            label_count = defaultdict(int)

            for row in joint_counts:
                b, c, cnt = int(row['bin']), str(row['y']), int(row['count'])
                bin_label_count[(b, c)] += cnt
                bin_count[b] += cnt
                label_count[c] += cnt

            n_rows = sum(bin_count.values())
            if n_rows == 0:
                mi_scores[feat] = 0.0
                continue

            # MI = Σ P(b,c) log( P(b,c) / (P(b) P(c)) )
            mi = 0.0
            for (b, c), n_bc in bin_label_count.items():
                p_bc = n_bc / n_rows
                p_b  = bin_count[b] / n_rows
                p_c  = label_count[c] / n_rows
                if p_bc > 0 and p_b > 0 and p_c > 0:
                    mi += p_bc * np.log(p_bc / (p_b * p_c))

            mi_scores[feat] = max(float(mi), 0.0)

        except Exception as e:
            logger.warning(f'MI computation failed for {feat}: {e}')
            mi_scores[feat] = 0.0

    return mi_scores


def compute_variance_spark(df_spark, feature_cols):
    """
    Compute variance of each feature column using Spark's built-in stddev.

    Returns dict {feature_name: variance}
    """
    try:
        from pyspark.sql import functions as F
    except ImportError:
        raise RuntimeError('PySpark not available')

    agg_exprs = [F.variance(F.col(c).cast('double')).alias(c) for c in feature_cols]
    row = df_spark.agg(*agg_exprs).collect()[0]
    return {c: float(row[c] or 0.0) for c in feature_cols}


def compute_snr_spark(df_spark, feature_cols):
    """
    Compute SNR heuristic weight: std(X_j) / GS_j  for each feature.

    GS_j = (max - min) / n  computed on Spark.
    Returns dict {feature_name: snr_weight}
    """
    try:
        from pyspark.sql import functions as F
    except ImportError:
        raise RuntimeError('PySpark not available')

    n = df_spark.count()
    agg_exprs = []
    for c in feature_cols:
        col = F.col(c).cast('double')
        agg_exprs += [
            F.stddev(col).alias(f'{c}__std'),
            F.min(col).alias(f'{c}__min'),
            F.max(col).alias(f'{c}__max'),
        ]

    row = df_spark.agg(*agg_exprs).collect()[0]
    snr = {}
    for c in feature_cols:
        std = float(row[f'{c}__std'] or 0.0)
        lo  = float(row[f'{c}__min'] or 0.0)
        hi  = float(row[f'{c}__max'] or 0.0)
        gs  = (hi - lo) / n + 1e-10
        snr[c] = std / gs
    return snr


# ---------------------------------------------------------------------------
# Spark noise injection
# ---------------------------------------------------------------------------

def apply_noise_spark(df_spark, feature_cols, eps_per_feature,
                      mechanism='laplace', delta=1e-5,
                      clip_percentiles=None, seed=42):
    """
    Add DP noise to each feature column in a Spark DataFrame.

    Uses Spark vectorised UDFs (pandas UDFs) for efficiency — runs
    noise injection in parallel across partitions.

    Parameters
    ----------
    df_spark        : input Spark DataFrame (original, unmodified)
    feature_cols    : list of feature column names to perturb
    eps_per_feature : dict {feature_name: ε_j}  (from allocate_budget)
    mechanism       : 'laplace' | 'gaussian'
    delta           : δ for Gaussian mechanism
    clip_percentiles: dict {feature_name: threshold} — clip before noise
                      (from clipped sensitivity). None = no clipping.
    seed            : random seed for reproducibility

    Returns
    -------
    df_noisy : Spark DataFrame with noise added to each feature column
    meta     : dict with per-feature sensitivity and noise scale
    """
    try:
        from pyspark.sql import functions as F
        from pyspark.sql.types import DoubleType
    except ImportError:
        raise RuntimeError('PySpark not available')

    # Compute per-feature sensitivity on Spark (min, max, count)
    n = df_spark.count()
    agg_exprs = []
    for c in feature_cols:
        col = F.col(c).cast('double')
        agg_exprs += [F.min(col).alias(f'{c}__min'), F.max(col).alias(f'{c}__max')]

    stats_row = df_spark.agg(*agg_exprs).collect()[0]

    meta = {}
    df_noisy = df_spark

    for i, feat in enumerate(feature_cols):
        eps_j = float(eps_per_feature.get(feat, 1e-6))
        if eps_j <= 0:
            eps_j = 1e-6

        lo = float(stats_row[f'{feat}__min'] or 0.0)
        hi = float(stats_row[f'{feat}__max'] or 0.0)

        # Apply clip threshold if provided
        if clip_percentiles and feat in clip_percentiles:
            clip_hi = float(clip_percentiles[feat])
        else:
            clip_hi = hi

        gs = (clip_hi - lo) / n if n > 0 else 1.0
        gs = max(gs, 1e-12)

        if mechanism == 'laplace':
            scale = gs / eps_j
            # Use Spark SQL expression: col + laplace_noise
            # Laplace via inverse CDF: -scale * sign(u) * log(1 - 2|u - 0.5|)
            # where u ~ Uniform(0,1)
            # Implemented as a deterministic SQL expression using rand(seed+i)
            u_expr = F.rand(seed=seed + i)
            sign_expr = F.when(u_expr < 0.5, F.lit(-1.0)).otherwise(F.lit(1.0))
            lap_expr  = F.lit(-scale) * sign_expr * F.log(
                F.lit(1.0) - F.lit(2.0) * F.abs(u_expr - F.lit(0.5))
            )
            noise_col = F.col(feat).cast('double') + lap_expr
            noise_scale = scale

        else:  # gaussian
            sigma = np.sqrt(2.0 * np.log(1.25 / delta)) * gs / eps_j
            # Gaussian via Box-Muller: sqrt(-2 ln u1) * cos(2π u2)
            u1_expr = F.greatest(F.rand(seed=seed + i), F.lit(1e-10))
            u2_expr = F.rand(seed=seed + i + len(feature_cols))
            gauss_expr = (
                F.sqrt(F.lit(-2.0) * F.log(u1_expr))
                * F.cos(F.lit(2.0 * np.pi) * u2_expr)
                * F.lit(sigma)
            )
            noise_col = F.col(feat).cast('double') + gauss_expr
            noise_scale = sigma

        # Clip to [lo, clip_hi] then add noise
        if clip_percentiles and feat in clip_percentiles:
            noise_col = F.when(
                F.col(feat).cast('double') > F.lit(clip_hi),
                F.lit(clip_hi)
            ).otherwise(F.col(feat).cast('double')) + (noise_col - F.col(feat).cast('double'))

        df_noisy = df_noisy.withColumn(feat, noise_col.cast(DoubleType()))

        meta[feat] = {
            'epsilon':     eps_j,
            'sensitivity': gs,
            'noise_scale': noise_scale,
            'lo':          lo,
            'hi':          hi,
            'clip_hi':     clip_hi,
        }

    return df_noisy, meta


# ---------------------------------------------------------------------------
# Clip percentile computation on Spark
# ---------------------------------------------------------------------------

def compute_clip_thresholds_spark(df_spark, feature_cols, percentile=99.0):
    """
    Compute the p-th percentile for each feature column using Spark's
    approxQuantile (single distributed pass over the data).

    Parameters
    ----------
    df_spark     : Spark DataFrame
    feature_cols : list of feature column names
    percentile   : clip percentile (default 99.0)

    Returns
    -------
    dict {feature_name: threshold}
    """
    q = percentile / 100.0
    thresholds = {}
    for feat in feature_cols:
        try:
            result = df_spark.approxQuantile(feat, [q], relativeError=0.01)
            thresholds[feat] = float(result[0]) if result else None
        except Exception as e:
            logger.warning(f'Could not compute percentile for {feat}: {e}')
            thresholds[feat] = None
    return thresholds


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_spark_dp_pipeline(spark, data_path, label_col,
                          total_eps=1.0,
                          method='mi',
                          mechanism='laplace',
                          n_bins=20,
                          clip_pct=99.0,
                          delta=1e-5,
                          feature_cols=None,
                          seed=42):
    """
    Full distributed DP pipeline on Spark.

    Raw CSV/HDFS  →  MI weights  →  budget allocation  →  noise injection
                                                        →  pandas DataFrame

    Parameters
    ----------
    spark       : active SparkSession
    data_path   : path to CSV (local, HDFS, S3, etc.)
    label_col   : name of the label column
    total_eps   : total privacy budget ε
    method      : importance method — 'mi' | 'variance' | 'snr' | 'uniform'
    mechanism   : 'laplace' | 'gaussian'
    n_bins      : discretisation bins for MI
    clip_pct    : percentile for clipped sensitivity (0 = no clipping)
    delta       : δ for Gaussian mechanism
    feature_cols: explicit list of features (None = auto-detect numeric cols)
    seed        : random seed

    Returns
    -------
    dict with keys:
        X_noisy   — pandas DataFrame, noise applied, ready for ML
        weights   — dict {feature: importance weight}
        epsilons  — dict {feature: ε_j allocated}
        features  — list of feature column names used
        meta      — per-feature noise metadata
        df_spark  — noisy Spark DataFrame (if further Spark processing needed)
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import NumericType, StringType

    logger.info(f'[SparkDP] Loading data from {data_path}')
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    df = df.withColumn(label_col,
                       F.col(label_col).cast(StringType()))

    # Auto-detect numeric feature columns
    if feature_cols is None:
        feature_cols = [
            f.name for f in df.schema.fields
            if isinstance(f.dataType, NumericType) and f.name != label_col
        ]
    logger.info(f'[SparkDP] {len(feature_cols)} feature columns detected')

    n_total = df.count()
    logger.info(f'[SparkDP] Dataset: {n_total:,} rows × {len(feature_cols)} features')

    # ── Step 1: Clipped sensitivity thresholds (one distributed pass) ──
    clip_thresholds = None
    if clip_pct and clip_pct > 0:
        logger.info(f'[SparkDP] Computing {clip_pct}th-percentile clip thresholds ...')
        clip_thresholds = compute_clip_thresholds_spark(df, feature_cols, clip_pct)
        logger.info(f'[SparkDP] Clip thresholds computed for {len(clip_thresholds)} features')

    # ── Step 2: Feature importance (distributed MI / variance / SNR) ──
    logger.info(f'[SparkDP] Computing feature importance via method={method} ...')

    if method == 'mi':
        raw_scores = compute_mi_spark(df, feature_cols, label_col, n_bins)
    elif method == 'variance':
        raw_scores = compute_variance_spark(df, feature_cols)
    elif method == 'snr':
        raw_scores = compute_snr_spark(df, feature_cols)
    else:  # uniform
        raw_scores = {f: 1.0 for f in feature_cols}

    # ── Step 3: Normalise to weights, allocate budget (driver-side) ──
    raw_arr = np.array([raw_scores.get(f, 0.0) for f in feature_cols])
    total = raw_arr.sum()
    if total <= 0:
        raw_arr = np.ones(len(feature_cols))
        total = float(len(feature_cols))

    weights_arr = raw_arr / total
    # Floor: no feature gets less than 1% of uniform share
    floor = 1.0 / (100.0 * len(feature_cols))
    weights_arr = np.maximum(weights_arr, floor)
    weights_arr /= weights_arr.sum()

    eps_arr = total_eps * weights_arr
    weights_dict  = {f: float(w) for f, w in zip(feature_cols, weights_arr)}
    epsilons_dict = {f: float(e) for f, e in zip(feature_cols, eps_arr)}

    logger.info(f'[SparkDP] Top-5 features by ε allocation:')
    top5 = sorted(epsilons_dict.items(), key=lambda x: -x[1])[:5]
    for feat, eps_j in top5:
        logger.info(f'  {feat:35s}  ε={eps_j:.5f}  weight={weights_dict[feat]:.4f}')

    # ── Step 4: Noise injection (distributed Spark UDF / SQL expressions) ──
    logger.info(f'[SparkDP] Injecting {mechanism} noise across {len(feature_cols)} features ...')
    df_noisy, meta = apply_noise_spark(
        df, feature_cols, epsilons_dict,
        mechanism=mechanism, delta=delta,
        clip_percentiles=clip_thresholds, seed=seed
    )

    # ── Step 5: Collect to pandas for ML training ──
    logger.info('[SparkDP] Collecting noisy DataFrame to pandas ...')
    df_pandas = df_noisy.toPandas()

    logger.info('[SparkDP] Pipeline complete.')
    logger.info(f'  Total ε spent : {sum(epsilons_dict.values()):.6f}  (target={total_eps})')
    logger.info(f'  Output shape  : {df_pandas.shape}')

    return {
        'X_noisy':   df_pandas[feature_cols],
        'y':         df_pandas[label_col],
        'weights':   weights_dict,
        'epsilons':  epsilons_dict,
        'features':  feature_cols,
        'meta':      meta,
        'df_spark':  df_noisy,
    }


# ---------------------------------------------------------------------------
# Single-machine fallback (when Spark not available)
# ---------------------------------------------------------------------------

def run_pandas_dp_pipeline(df_pandas, label_col, feature_cols,
                           total_eps=1.0, method='mi',
                           mechanism='laplace', clip_pct=99.0, delta=1e-5):
    """
    Identical pipeline running on pandas/numpy when Spark is unavailable.
    Calls the existing weighted_budget.py functions directly.
    """
    from weighted_budget import (compute_feature_importance, allocate_budget,
                                  apply_weighted_dp_noise, clipped_sensitivity)

    X = df_pandas[feature_cols].values.astype(float)
    y = df_pandas[label_col].values

    weights, raw = compute_feature_importance(X, y, method=method)
    eps_arr = allocate_budget(total_eps, weights)

    clip_thresholds = None
    if clip_pct and clip_pct > 0:
        import numpy as np
        clip_thresholds_arr = np.percentile(X, clip_pct, axis=0)
        clip_thresholds = {f: float(clip_thresholds_arr[i])
                           for i, f in enumerate(feature_cols)}

    X_noisy, meta = apply_weighted_dp_noise(
        X, total_eps, y, method=method, mechanism=mechanism,
        delta=delta, clip_percentile=clip_pct if clip_pct else None
    )

    return {
        'X_noisy':  X_noisy,
        'y':        y,
        'weights':  {f: float(weights[i]) for i, f in enumerate(feature_cols)},
        'epsilons': {f: float(eps_arr[i]) for i, f in enumerate(feature_cols)},
        'features': feature_cols,
        'meta':     meta,
    }
