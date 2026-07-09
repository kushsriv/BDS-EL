# BDS-EL Setup & Run Guide

## Prerequisites — Windows (PowerShell)

### Step 1 — Install Java 17 (required for Spark)

Open PowerShell as Administrator and run:

```powershell
winget install Microsoft.OpenJDK.17
```

After install, **close and reopen PowerShell**, then verify:

```powershell
java -version
# Should print: openjdk version "17.x.x"
```

If `winget` is not available, download Java 17 manually from:
https://adoptium.net/temurin/releases/?version=17
Install it, then add it to PATH:
```powershell
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.x.x-hotspot"
$env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
```

---

### Step 2 — Install Python dependencies

```powershell
cd C:\Users\sriva\OneDrive\Desktop\bdsel6sem\BDS-EL

pip install -r requirements.txt
```

This installs: numpy, pandas, matplotlib, scikit-learn, scipy, pyspark, torch, opacus

> Note: `torch` and `opacus` are large (~2GB). If download is slow, install separately:
> ```powershell
> pip install numpy pandas matplotlib scikit-learn scipy pyspark
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install opacus
> ```

---

### Step 3 — Set JAVA_HOME environment variable

Spark needs JAVA_HOME set. In PowerShell:

```powershell
# Find where Java was installed
Get-Command java | Select-Object -ExpandProperty Source

# Set JAVA_HOME (replace path with what you found above)
[System.Environment]::SetEnvironmentVariable("JAVA_HOME", "C:\Program Files\Eclipse Adoptium\jdk-17.0.x.x-hotspot", "User")
```

**Restart PowerShell** after this step.

---

### Step 4 — Verify Spark works

```powershell
python -c "from pyspark.sql import SparkSession; s = SparkSession.builder.appName('test').getOrCreate(); print('Spark OK:', s.version); s.stop()"
```

Should print: `Spark OK: 3.x.x`

---

## Running the Project

### Full pipeline with Spark (Big Data mode)

```powershell
cd C:\Users\sriva\OneDrive\Desktop\bdsel6sem\BDS-EL

python main.py --dataset KDDTrain+.csv --ml_runs 5
```

You will see in the terminal:
```
INFO: Detected dataset type: NSL-KDD
INFO: Loading dataset via Spark: KDDTrain+.csv
INFO: === Module: Spark Distributed DP Pipeline ===
INFO:   [Spark] method=mi ε=1.0 — distributed run
INFO:   [Spark] method=variance ε=1.0 — distributed run
INFO: Saved results/spark_dp_pipeline.csv
```

---

### Quick test run (2 minutes, skips slow modules)

```powershell
python main.py --dataset KDDTrain+.csv --ml_runs 2 --skip_multiclass --skip_budget_compare --skip_dpsgd --skip_prv --skip_stats
```

---

### Without Spark (pandas-only, always works)

```powershell
python main.py --dataset KDDTrain+.csv --no_spark --ml_runs 5
```

---

### Run frontend dashboard

```powershell
cd frontend
npm install
npm run dev
```

Open browser at: http://localhost:5173

---

## What Each Run Produces

| Output file | What it contains |
|---|---|
| `results/spark_dp_pipeline.csv` | Per-feature ε allocations from Spark distributed MI |
| `results/dp_ml_results.csv` | Budget collapse: 53.3% accuracy at all ε |
| `results/dpsgd_results.csv` | DP-SGD fix: 94.70% at ε=1.0 |
| `results/clipped_sensitivity.csv` | Noise reduction per feature (592M× for su_attempted) |
| `results/prv_composition.csv` | PRV vs RDP: 15.34 vs 31.54 at k=100 |
| `results/mia_results.csv` | MIA validation: 0/24 bound violations |
| `results/sensitivity_report.csv` | GS / LS / Clipped-GS for all 37 features |

---

## Project Structure

```
BDS-EL/
├── main.py                  ← Run this to start everything
├── spark_pipeline.py        ← Spark distributed DP pipeline
├── sensitivity.py           ← Novel: Clipped Sensitivity (Lemma 1 + 2)
├── weighted_budget.py       ← Novel: Importance-Weighted Budget Allocation
├── prv_accountant.py        ← Novel: PRV via FFT composition
├── dp_sgd.py                ← DP-SGD (the fix to budget collapse)
├── dp_ml.py                 ← Input perturbation (shows the problem)
├── multi_dataset.py         ← NSL-KDD / UNSW-NB15 / CIC-IDS-2017 loaders
├── membership_inference.py  ← MIA attack validation
├── local_dp.py              ← Local DP vs Central DP
├── statistical_analysis.py  ← 30-run Wilcoxon tests
├── requirements.txt         ← pip install -r requirements.txt
├── KDDTrain+.csv            ← NSL-KDD dataset (125,974 rows)
├── UNSW_NB15_training-set.csv ← Synthetic UNSW-NB15 (82,332 rows)
├── frontend/                ← React dashboard (12 sections)
│   └── src/
│       ├── App.jsx
│       └── dpmath.js
└── results/                 ← Auto-generated CSVs and plots
```

---

## Troubleshooting

**"Java not found" error:**
```powershell
# Check if Java is on PATH
java -version
# If not found, add manually:
$env:PATH = "C:\Program Files\Eclipse Adoptium\jdk-17.0.x.x-hotspot\bin;$env:PATH"
```

**"JAVA_HOME not set" error:**
```powershell
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.x.x-hotspot"
```

**Spark takes too long to start:**
Normal — Spark JVM startup takes 15-30 seconds the first time.

**torch/opacus install fails:**
The DP-SGD module needs these. If install fails, skip DP-SGD:
```powershell
python main.py --dataset KDDTrain+.csv --skip_dpsgd
```

**Out of memory during Spark run:**
```powershell
# Reduce Spark memory in main.py line 895:
# Change '4g' to '2g'
python main.py --dataset KDDTrain+.csv --ml_runs 2
```
