# BDS-EL: Differential Privacy Research Framework

A reproducible **differential privacy research framework** targeting Q1/A* publication.
Validated on **NSL-KDD** (125,973 records, 37 features) with 11 experiment modules,
3 novel algorithmic contributions, and a statistical analysis pipeline.

---

## Novel Contributions

| Contribution | File | Key Result |
|---|---|---|
| **Clipped sensitivity** for heavy-tailed network traffic | `sensitivity.py`, `weighted_budget.py` | Up to **593M× noise reduction** (su_attempted feature); 15/37 features get >1000× at p99 |
| **Importance-weighted DP budget allocation** (MI / variance / SNR) | `weighted_budget.py` | Formally ε_total-DP by composition; concentrates budget on high-MI features |
| **PRV accountant** vs. RDP (Gopi et al. 2021) | `prv_accountant.py` | **42–59% tighter** than RDP at k=100 queries, ε₀=0.1 |

---

## Key Results (30 runs, 95% CI, NSL-KDD)

### Binary IDS (Normal vs. Attack), 37 features

| ε | Input-Uniform | Input-MI-Weighted | **DP-SGD** |
|---|---|---|---|
| ∞ (no DP) | **94.77%** | **94.77%** | **94.77%** |
| 0.1 | 54.28% | 53.29% | **92.80%** |
| 0.5 | 54.29% | 53.29% | **94.45%** |
| 1.0 | 54.29% | 53.29% | **94.70%** |
| 2.0 | 54.29% | 53.29% | **94.59%** |
| 5.0 | 54.29% | 67.09% | — |
| 10.0 | 54.29% | 70.89% | — |

**DP-SGD at ε=1.0 loses only 0.07pp** vs. the no-DP baseline.
Input perturbation collapses (predicts all-majority) for ε ≤ 5 with 37 features.

### Composition Tightness (Laplace, ε₀=0.1)

| k queries | Basic | RDP | **PRV** |
|---|---|---|---|
| 1 | 0.100 | 0.272 | **0.110** |
| 10 | 1.000 | 1.074 | **0.681** |
| 50 | 5.000 | 3.429 | **1.864** |
| 100 | 10.000 | 5.076 | **2.904** |

PRV is up to **59.6% tighter than RDP** at k=1, **42.8% tighter at k=100**.

### Clipped Sensitivity (NSL-KDD, p=99%)

| Feature | Raw GS | Clipped GS | Noise Reduction |
|---|---|---|---|
| `su_attempted` | very large | tiny | **592,825,447×** |
| `num_root` | large | small | **3,413,430×** |
| `num_shells` | large | small | **714,439×** |
| `src_bytes` | 10,399 | 0.20 | **51,332×** |
| `count` | 0.004 | 0.004 | 1× (well-behaved) |

### MIA Validation

- 24 attack configurations tested (2 mechanisms × 6 ε × 2 features)
- **0/24 DP bound violations** — all empirical attack success ≤ Yeom et al. bound

---

## Repository Structure

```
BDS-EL/
├── main.py                   # Orchestrator: 11 experiment modules
├── dp_utils.py               # Core DP noise + amplification
├── sensitivity.py            # GS / LS / smooth / clipped sensitivity
├── rdp_accountant.py         # Rényi DP accountant + composition bounds
├── prv_accountant.py         # PRV accountant (Gopi et al. 2021) — Novel
├── weighted_budget.py        # MI/variance/SNR budget allocation — Novel
├── dp_sgd.py                 # DP-SGD (Abadi et al. 2016) + RDP accounting
├── dp_ml.py                  # Input perturbation + multi-class IDS
├── membership_inference.py   # Likelihood-ratio MIA validation
├── local_dp.py               # Laplace-LDP, Duchi (2013), Wang (2019)
├── statistical_analysis.py   # Statistical tables + master summary
├── multi_dataset.py          # Adapters: NSL-KDD, UNSW-NB15, CIC-IDS-2017
├── txt2csv.py                # KDD .txt → .csv converter
├── requirements.txt
├── results/                  # CSVs, plots, statistical tables
└── frontend/                 # React research dashboard (self-contained)
    ├── src/App.jsx            # 7 interactive sections, Recharts
    ├── src/dpmath.js          # Full DP math in browser (no backend)
    └── src/index.css          # Custom dark theme
```

---

## Quick Start

```bash
# 1. Convert raw dataset (only once)
python txt2csv.py

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run all 11 experiment modules + stats (30 runs, ~10 min)
python main.py --no_spark --ml_runs 30

# 4. Run only specific modules (e.g., skip DP-ML for speed)
python main.py --no_spark --skip_ml --skip_dpsgd --skip_multiclass

# 5. Launch React dashboard
cd frontend && npm install && npm run dev
```

---

## Experiment Modules

### Module 1 — DP Aggregates with Data-Driven Sensitivity

Compares POST / AGG / SHUFFLE aggregation models with **data-driven GS** (replaces broken `sensitivity=1.0`).

### Module 2 — RDP Composition

Tracks cumulative budget with Rényi DP. RDP is 1.97× tighter than basic composition at k=100 queries.

### Module 3 — Privacy Amplification by Subsampling

Demonstrates Poisson subsampling amplification: at q=0.1, effective ε drops to ε_nom/10.5.

### Module 4 — Membership Inference Attack Validation

Optimal likelihood-ratio MIA; validates empirical attack success ≤ Yeom et al. 2018 bound at all ε.

### Module 5 — Local DP vs. Central DP

Compares Laplace-LDP, Duchi (2013), and Wang piecewise (2019) vs. central DP. Quantifies the √n penalty of the local model.

### Module 6 — DP-ML: Binary Intrusion Detection

Input perturbation (uniform and MI-weighted) sweep across ε ∈ {0.1, …, 10}. 30 runs, 95% CI.

### Module 7 — Budget Allocation Comparison

Head-to-head: uniform vs. MI-weighted vs. variance-weighted vs. SNR-optimal at fixed total ε.

### Module 8 — Multi-Class IDS (5 Categories)

5-class softmax classifier: Normal / DoS / Probe / R2L / U2R. Baseline: 94.38%.

### Module 9 — DP-SGD (Abadi et al. 2016)

Gradient clipping + Gaussian noise. Binary search for σ(ε). 94.59% at ε=2.0 (0.18pp loss).

### Module 10 — Clipped Sensitivity Analysis *(Novel)*

Computes noise reduction from clipping at p ∈ {90, 95, 99, 99.5}% across all 37 features. Saves `results/clipped_sensitivity.csv` and bar chart.

### Module 11 — PRV vs. RDP Composition *(Novel)*

FFT-based numerical PRV composition (Gopi et al. 2021). 42–59% tighter than RDP at k=100.

---

## CLI Reference

```
python main.py [OPTIONS]

--dataset PATH          Input CSV (default: KDDTrain+.csv)
--features F1 F2 ...    Explicit feature list (default: auto-detect all 37)
--epsilons E1 E2 ...    ε sweep values (default: 0.1 0.5 1.0 2.0 5.0 10.0)
--mechanisms M1 M2      laplace gaussian (default: both)
--runs N                Repetitions for aggregate experiments (default: 30)
--ml_runs N             Repetitions for DP-ML / CI (default: 30)
--max_ml_rows N         Subsample for DP-ML speed (default: 20000)
--delta D               δ for (ε,δ)-DP (default: 1e-5)
--results_dir DIR       Output directory (default: results)
--no_spark              Use pandas (no PySpark needed)
--skip_ml               Skip Module 6
--skip_budget_compare   Skip Module 7
--skip_multiclass       Skip Module 8
--skip_dpsgd            Skip Module 9
--skip_prv              Skip Module 11
--skip_stats            Skip statistical analysis
```

---

## Output Files

| File | Contents |
|---|---|
| `results/aggregate_results.csv` | POST/AGG/SHUFFLE error per feature, ε, mechanism |
| `results/sensitivity_report.csv` | GS, LS, smooth sens, clipped GS per feature |
| `results/clipped_sensitivity.csv` | Noise reduction by clip percentile × feature |
| `results/rdp_composition.csv` | Basic/advanced/RDP ε at k ∈ {1…100} |
| `results/prv_composition.csv` | PRV vs. RDP tightness |
| `results/amplification_results.csv` | Amplified ε by sampling rate |
| `results/mia_results.csv` | Empirical vs. theoretical MIA accuracy |
| `results/ldp_results.csv` | LDP mechanism error per ε |
| `results/dp_ml_results.csv` | Accuracy/F1 ± CI for input perturbation |
| `results/dpsgd_results.csv` | Accuracy/F1 ± CI for DP-SGD |
| `results/budget_comparison.csv` | Uniform vs. MI/variance/SNR accuracy |
| `results/multiclass_results.csv` | Per-class accuracy for 5-class IDS |
| `results/stat_table1_dpml.csv` | DP-ML summary with Wilcoxon p-values |
| `results/stat_table2_composition.csv` | Composition tightness table |
| `results/stat_table3a_*.csv` | Clipped sensitivity aggregate summary |
| `results/stat_table3b_*.csv` | Top-5 features by noise reduction |
| `results/stat_table4_mia.csv` | MIA bound validation table |
| `results/stat_table5_ldp_penalty.csv` | LDP vs. central MSE penalty |
| `results/stat_table6_budget_significance.csv` | Budget method significance tests |
| `results/stat_table7_multiclass.csv` | Multi-class per-category accuracy |
| `results/master_summary.txt` | One-page results digest |

---

## Multi-Dataset Support

Drop a dataset CSV and run — the framework auto-detects format:

```bash
# UNSW-NB15 (82K records, 49 features, 9 attack categories)
# Download: https://research.unsw.edu.au/projects/unsw-nb15-dataset
python main.py --dataset UNSW_NB15_training-set.csv --no_spark

# CIC-IDS-2017 (2.8M records, 78 features)
# Download: https://www.unb.ca/cic/datasets/ids-2017.html
python main.py --dataset path/to/MachineLearningCVE/ --no_spark
```

Adapter code: `multi_dataset.py` — handles column normalisation, label encoding, NaN/Inf cleaning.

---

## React Dashboard

Self-contained — all DP math runs in-browser (no server needed).

```bash
cd frontend
npm install
npm run dev       # → http://localhost:5173
```

**7 interactive sections:**
1. Privacy–Utility tradeoff curves (POST / AGG / SHUFFLE)
2. Sensitivity analysis (GS vs. LS vs. clipped)
3. RDP composition bounds (live slider)
4. Privacy amplification by subsampling
5. MIA validation (safe-zone shading, DP bound)
6. Local vs. central DP (√n penalty)
7. DP-ML accuracy vs. ε with error bars

---

## Architecture

```
NSL-KDD CSV
    │
    ├─ sensitivity.py ──────── GS / LS / smooth / clipped sensitivity
    │
    ├─ dp_utils.py ─────────── Laplace / Gaussian noise primitives
    │
    ├─ rdp_accountant.py ───── RDP tracking + basic/advanced/RDP comparison
    ├─ prv_accountant.py ───── PRV composition (FFT convolution)  ← Novel
    │
    ├─ weighted_budget.py ──── MI / variance / SNR budget allocation ← Novel
    │
    ├─ dp_ml.py ────────────── Input perturbation (uniform + weighted)
    ├─ dp_sgd.py ───────────── DP-SGD (Abadi et al. 2016)
    │
    ├─ membership_inference.py ─ Likelihood-ratio MIA
    ├─ local_dp.py ─────────── Duchi (2013) + Wang piecewise (2019)
    │
    ├─ statistical_analysis.py ─ 7 stat tables + master summary
    └─ main.py ─────────────── Orchestrates all 11 modules
```

---

## References

1. Dwork et al. (2006). *Calibrating noise to sensitivity in private data analysis.* TCC.
2. Nissim, Raskhodnikova & Smith (2007). *Smooth sensitivity and sampling in private data analysis.* STOC.
3. Dwork, Rothblum & Vadhan (2010). *Boosting and differential privacy.* FOCS.
4. Duchi, Jordan & Wainwright (2013). *Local privacy and statistical minimax rates.* FOCS.
5. Mironov (2017). *Rényi differential privacy.* CSF.
6. Wang et al. (2019). *Collecting and analyzing multidimensional data with local differential privacy.* ICDE.
7. Yeom et al. (2018). *Privacy risk in machine learning: analyzing the connection to overfitting.* CSF.
8. Abadi et al. (2016). *Deep learning with differential privacy.* CCS.
9. Balle et al. (2018). *Privacy amplification by subsampling.* ICML.
10. Gopi, Lee & Wajc (2021). *Numerical composition of differential privacy.* NeurIPS.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Data | pandas, NumPy |
| Distributed (optional) | PySpark |
| Plots | matplotlib |
| Dashboard | React 18 + Vite + Recharts |
| Stats | Pure NumPy (no scipy/sklearn) |

---

## Target Publication Venues

| Venue | Type | Fit |
|---|---|---|
| **IEEE TIFS** (IF 6.8) | Q1 Journal | DP + IDS, empirical benchmark |
| **IEEE TDSC** (IF 7.3) | Q1 Journal | Dependable/secure computing |
| **Computers & Security** (IF 4.8) | Q1 Journal | Applied security benchmark |
| **USENIX Security** | A* Conference | Strong MIA + DP-SGD results |
| **ACM CCS** | A* Conference | PRV accountant + novel theory |
