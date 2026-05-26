# BDS-EL: Differential Privacy Research Framework

An empirical research framework for studying **privacy–utility tradeoffs** in differential privacy (DP), validated on the NSL-KDD network intrusion detection dataset (125,973 records).

The project goes beyond a simple DP demo: it implements data-driven sensitivity calibration, Rényi DP composition, subsampling amplification, membership inference attack validation, local DP mechanisms, and DP-ML — each as a reproducible, configurable experiment module.

---

## What's Inside

```
BDS-EL/
├── main.py                   # Experiment orchestrator (6 modules)
├── dp_utils.py               # Core DP noise + amplification utilities
├── sensitivity.py            # Global / local / smooth sensitivity
├── rdp_accountant.py         # Rényi DP accountant + composition bounds
├── membership_inference.py   # Likelihood-ratio MIA simulation
├── local_dp.py               # Laplace-LDP, Duchi (2013), Wang (2019)
├── dp_ml.py                  # DP-noised intrusion detection classifier
├── txt2csv.py                # KDD .txt → .csv converter
├── requirements.txt
├── results/                  # Generated CSVs + plots
└── frontend/                 # React research dashboard
    ├── src/
    │   ├── App.jsx           # All 7 sections + charts
    │   ├── dpmath.js         # DP math in JS (runs in-browser)
    │   └── index.css         # Custom dark theme
    └── package.json
```

---

## Dataset

**NSL-KDD** (KDD Cup 1999, improved version)

| Property | Value |
|---|---|
| Training records | 125,973 |
| Features | 41 network traffic attributes |
| Label | Binary: `normal` vs. attack |
| Attack types | neptune, ipsweep, portsweep, satan, … |

The raw `.txt` file ships with the repo. Convert with:
```bash
python txt2csv.py        # produces KDDTrain+.csv
```

---

## Experiment Modules

### Module 1 — DP Aggregates with Data-Driven Sensitivity

Replaces the broken `sensitivity=1.0` default with the correct **global sensitivity** `GS = (max − min) / n` computed from the actual data distribution. Compares three aggregation models across both mechanisms.

| Model | Description |
|---|---|
| **POST** | Noise added to the scalar aggregate |
| **AGG** | Noise added to the sum, then divided by n |
| **SHUFFLE** | Per-record noise, then averaged (local model) |

Key result: at ε = 1.0 on `src_bytes`, POST error ≈ 10,399 (calibrated) vs. the old `sensitivity=1.0` which would produce noise 10,000× too small.

---

### Module 2 — RDP Composition Analysis

Tracks cumulative privacy budget across multiple queries using **Rényi Differential Privacy** (Mironov 2017). Compares three composition bounds:

| Bound | Method |
|---|---|
| Basic | Σ εᵢ (worst case) |
| Advanced | √(2k · log 1/δ) · ε + k · ε · (eᵉ − 1) — Dwork et al. 2010 |
| **RDP** | Sum RDP at each order α, convert via ε(δ) = RDP(α) + log(1/δ)/(α−1) |

RDP gives the tightest bound and grows more slowly than basic composition, enabling more queries within a fixed budget.

---

### Module 3 — Privacy Amplification by Subsampling

Implements the **Poisson subsampling amplification lemma** (Balle et al. 2018):

```
ε_amp = log(1 + q · (exp(ε) − 1))
```

At q = 0.1 (sample 10% of records), a nominal ε = 1.0 mechanism achieves effective ε ≈ 0.159 — a **6× amplification** with no algorithmic changes.

---

### Module 4 — Membership Inference Attack Validation

Empirically validates DP guarantees using the **optimal likelihood-ratio attack** (Neyman-Pearson).

The attacker observes a DP query output and tests: *was a target record in the dataset?*

DP theory guarantees (Yeom et al. 2018):
```
Pr[attacker correct] ≤ exp(ε) / (1 + exp(ε))
```

Results confirm the implementation: empirical attack accuracy stays well below the theoretical bound across all ε values, with near-random accuracy (≈ 50%) at ε = 0.1.

---

### Module 5 — Local DP vs. Central DP

Compares three **ε-LDP mechanisms** against central DP on the same features:

| Mechanism | Description |
|---|---|
| Laplace-LDP | User adds Lap(0, s/ε) locally |
| **Duchi (2013)** | MSE-optimal for real-valued LDP; outputs ±C |
| **Wang piecewise (2019)** | Beats Duchi for ε > 0.61; used in Apple's DP deployments |

The cost of local DP: error is **√n ≈ 355× larger** than central DP at the same ε (for n = 125,973). This quantifies the privacy model trade-off — LDP requires no trusted curator but at significant utility cost.

---

### Module 6 — DP-ML: Intrusion Detection

Trains a binary classifier on **DP-noised training features** and evaluates on a clean test set. Measures how classification utility degrades with ε.

- No-DP baseline accuracy: **80.8%**  
- At ε = 5.0 (Laplace): **80.0%** (< 1% loss)  
- At ε ≤ 2.0: significant degradation due to per-feature budget splitting across 6 features

Budget allocation: total ε is split equally across d features (basic composition), so each feature receives ε/d.

---

## Quick Start

### Requirements

```bash
pip install numpy pandas matplotlib
# PySpark optional — framework auto-detects and falls back to pandas
pip install pyspark==3.5.1   # optional, for distributed mode
```

### Run All Experiments

```bash
# Without Spark (pandas mode — works anywhere)
python main.py --dataset KDDTrain+.csv --no_spark

# With Spark (distributed mode)
python main.py --dataset KDDTrain+.csv

# Custom parameters
python main.py \
  --dataset KDDTrain+.csv \
  --features src_bytes count srv_count dst_host_count \
  --epsilons 0.1 0.3 0.5 1.0 2.0 5.0 \
  --mechanisms laplace gaussian \
  --runs 30 \
  --results_dir results \
  --no_spark
```

### CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `KDDTrain+.csv` | Path to dataset |
| `--features` | 6 default features | Features to analyse |
| `--epsilons` | `0.1 0.3 0.5 1.0 2.0 5.0` | Privacy budget values |
| `--mechanisms` | `laplace gaussian` | DP mechanisms |
| `--runs` | `30` | Repetitions per experiment |
| `--delta` | `1e-5` | δ for (ε, δ)-DP (Gaussian) |
| `--results_dir` | `results/` | Output directory |
| `--no_spark` | False | Use pandas instead of Spark |
| `--skip_ml` | False | Skip DP-ML module (faster) |

### Outputs

All results are saved to `results/`:

| File | Contents |
|---|---|
| `aggregate_results.csv` | POST/AGG/SHUFFLE errors per feature × mechanism × ε |
| `sensitivity_report.csv` | GS, LS, smooth sensitivity per feature |
| `rdp_composition.csv` | Basic, advanced, RDP bounds vs. query count |
| `amplification_results.csv` | Effective ε after subsampling amplification |
| `mia_results.csv` | Empirical attack accuracy vs. theoretical bound |
| `ldp_results.csv` | Local DP mechanism errors |
| `central_dp_results.csv` | Central DP errors for comparison |
| `dp_ml_results.csv` | Accuracy / F1 vs. ε for intrusion detection |
| `*.png` | Privacy-utility plots for each experiment |

---

## Research Dashboard (React)

An interactive single-page dashboard that visualises all experiment results. All DP math (RDP, amplification, MIA bounds) runs in-browser — no backend required.

```bash
cd frontend
npm install
npm run dev       # opens at http://localhost:5173
```

**Sections:**

1. **Privacy–Utility Tradeoff** — log-scale error curves, feature selector
2. **Sensitivity Analysis** — GS vs. LS per feature (bar chart)
3. **RDP Composition** — live query-count slider, budget comparison boxes
4. **Privacy Amplification** — multi-rate subsampling curves
5. **MIA Validation** — empirical accuracy vs. DP bound, safe-zone shading
6. **Local vs. Central DP** — √n penalty visualised
7. **DP-ML Results** — accuracy/F1 vs. ε with error bars

> To display real experiment data: copy CSVs from `results/` into `frontend/public/data/`. The app loads them automatically; otherwise it uses in-browser computed sample data.

---

## Architecture

```
KDDTrain+.csv
      │
      ▼
  main.py  ──────────────────────────────────────────────────────┐
      │                                                           │
      ├─► sensitivity.py        GS / LS / smooth sensitivity      │
      ├─► dp_utils.py           Laplace, Gaussian, amplification  │
      ├─► rdp_accountant.py     RDP composition + DP conversion   │
      ├─► membership_inference  LR attack + Yeom bound            │
      ├─► local_dp.py           Duchi, Piecewise, RR              │
      └─► dp_ml.py              Logistic regression on DP data    │
                                                                   │
      results/*.csv + results/*.png  ◄──────────────────────────-─┘
            │
            ▼
      frontend/  (React dashboard)
```

---

## References

| Paper | Used in |
|---|---|
| Dwork et al. "Calibrating noise to sensitivity in private data analysis" (2006) | Laplace mechanism |
| Mironov "Rényi Differential Privacy of the Gaussian Mechanism" (2017) | RDP accountant |
| Dwork, Rothblum & Vadhan "Boosting and differential privacy" (2010) | Advanced composition |
| Nissim, Raskhodnikova & Smith "Smooth sensitivity and sampling in private data analysis" (2007) | Smooth sensitivity |
| Balle et al. "Privacy amplification by subsampling" (2018) | Amplification module |
| Yeom et al. "Privacy risk in machine learning" (2018) | MIA theoretical bound |
| Duchi, Jordan & Wainwright "Local privacy and statistical minimax rates" (2013) | Duchi mechanism |
| Wang et al. "Collecting and analyzing multidimensional data with local differential privacy" (2019) | Piecewise mechanism |
| Abadi et al. "Deep learning with differential privacy" (2016) | DP-ML background |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data processing | PySpark 3.5.1 (optional) / pandas |
| Numerical computing | NumPy |
| Visualisation (Python) | Matplotlib |
| Frontend | React 18 + Vite |
| Charts | Recharts |
| Frontend styling | Custom CSS (dark theme, no UI kit) |
