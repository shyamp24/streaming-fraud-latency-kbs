# Streaming Fraud Detection under Verification Latency

Reproduction code and verified result files for the paper:
**"When Does History Help? The Vulnerability of Longitudinal Account Features to Verification Latency in Streaming Fraud Detection"** (*Knowledge-Based Systems*, Elsevier).

## Overview
Evaluates the impact of ground-truth verification delay ($D$) and positional lag ($k$) on longitudinal account representations in high-throughput transaction streams using the IEEE-CIS fraud detection benchmark ($590,540$ chronological transactions; $70\%$ train, $10\%$ validation, $20\%$ test).

## Repository Structure
```text
├── README.md
├── requirements.txt
├── run_pipeline.py
├── src/                                   # Modular reusable components
│   ├── __init__.py
│   ├── cost_eval.py                       # Canonical 7-tier cost schedule & validation tau* tuning
│   ├── data_loader.py                     # Chronological 70/10/20 partition loader
│   ├── dual_channel.py                    # Dual-channel mixture delay helper
│   ├── feature_engineering.py             # Positional lag (k) and verification delay (D)
│   └── label_free.py                      # Delay-invariant behavioral feature helper
├── code/                                  # Full experimental, test, and figure scripts
│   ├── cost_eval.py                       # Canonical 7-tier cost & evaluation module
│   ├── run_unified_393_pipeline.py        # Table 1: Unified 393-feature 3-seed benchmark
│   ├── run_fine_grained_latency_sweep.py  # Table 3A: Continuous delay sweep (0s to 30d)
│   ├── run_fine_grained_bootstrap.py      # Table 3B: 1,000-resample clustered bootstrap
│   ├── run_step3_clustered_bootstrap.py   # Clustered bootstrap grid verification
│   ├── run_density_sweep_393.py           # Table 4: Account density sweep (25%, 50%, 100%)
│   ├── run_constructive_features.py       # Table 5: Constructive label-free features
│   ├── run_composite_account_check.py     # Composite account identifier sensitivity
│   ├── run_canonical_experiments.py       # Tables 3C & 6: Cost sensitivity & mixture delay
│   ├── run_label_propagation_diagnostics.py # Diagnostics A & B (intra-burst linkage & profile)
│   ├── test_leakage_unit_test.py          # Zero-leakage assertion suite
│   ├── test_synthetic_boundary_builder.py # Synthetic boundary verification
│   ├── regenerate_all_tables.py           # Canonical table regeneration
│   └── generate_manuscript_figures.py     # Publication figure generation
└── out/
    └── results/                           # Saved CSV, JSON, and log outputs for all tables
```

## Setup & Requirements
Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

## Data Access
Raw datasets and intermediate prediction caches are intentionally excluded from this repository. Download the public benchmark from Kaggle:
- **Kaggle IEEE-CIS Fraud Detection:** https://www.kaggle.com/c/ieee-fraud-detection

Place `train_transaction.csv` and `train_identity.csv` in `data/`, or set environment variables before running any script:
```bash
export IEEE_CIS_TX_PATH="/path/to/train_transaction.csv"
export IEEE_CIS_ID_PATH="/path/to/train_identity.csv"
```

## Reproducing Experiments & Tests
Run all commands from the repository root:

1. **Zero-Leakage Unit Tests:**
   ```bash
   python code/test_synthetic_boundary_builder.py
   python code/test_leakage_unit_test.py
   ```
2. **Main 393-Feature Benchmark (Table 1):**
   ```bash
   python code/run_unified_393_pipeline.py
   ```
3. **Continuous Verification Latency Sweep & Clustered Bootstrap (Tables 3A & 3B):**
   ```bash
   python code/run_fine_grained_latency_sweep.py
   python code/run_fine_grained_bootstrap.py
   ```
4. **Cost-Schedule Sensitivity & Dual-Channel Mixture Delay (Tables 3C & 6):**
   ```bash
   python code/run_canonical_experiments.py
   ```
5. **Account Density Sweep & Constructive Label-Free Features (Tables 4 & 5):**
   ```bash
   python code/run_density_sweep_393.py
   python code/run_constructive_features.py
   python code/run_composite_account_check.py
   ```
6. **Intra-Burst Diagnostics (Diagnostics A & B):**
   ```bash
   python code/run_label_propagation_diagnostics.py
   ```
7. **Regenerate Tables & Figures from Saved Results:**
   ```bash
   python code/regenerate_all_tables.py
   python code/generate_manuscript_figures.py
   ```
