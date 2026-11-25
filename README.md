# When Does History Help? The Vulnerability of Longitudinal Account Features to Verification Latency in Streaming Fraud Detection

Reproduction code, zero-leakage unit tests, and verified empirical result files for the paper:
**"When Does History Help? The Vulnerability of Longitudinal Account Features to Verification Latency in Streaming Fraud Detection"** (*Knowledge-Based Systems*, Elsevier).

## Overview
Evaluates the impact of ground-truth verification delay ($D$) and positional lag ($k$) on longitudinal account representations in high-throughput transaction streams using the IEEE-CIS fraud detection benchmark ($590,540$ chronological transactions; $70\%$ train, $10\%$ validation, $20\%$ test).

## Environment & Exact Package Versions
All experiments reported in the paper were executed under:
- **Python:** `3.12.4`
- **Core packages (`requirements.txt`):** `numpy==1.26.4`, `pandas==2.1.4`, `scikit-learn==1.4.2`, `xgboost==2.1.0`, `scipy==1.11.4`, `matplotlib==3.7.5`, `joblib==1.3.2`

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Repository Structure
```text
├── LICENSE                                  # MIT License
├── README.md
├── requirements.txt                         # Exact pinned package versions
├── code/
│   ├── cost_eval.py                         # Canonical 7-tier cost schedule & validation tau* tuning
│   ├── run_unified_393_pipeline.py          # Table 1: Unified 393-feature 3-seed benchmark
│   ├── run_fine_grained_latency_sweep.py    # Table 3A: Continuous delay sweep (0s to 30d)
│   ├── run_fine_grained_bootstrap.py        # 1,000-replicate clustered bootstrap helper
│   ├── run_bootstrap_10k_unified.py         # Table 3B: 10,000-replicate clustered bootstrap
│   ├── run_step3_clustered_bootstrap.py     # Clustered bootstrap grid verification
│   ├── run_density_sweep_393.py             # Table 4: Account density sweep (25%, 50%, 100%; 1,000 replicates)
│   ├── run_constructive_features.py         # Table 5: Constructive label-free features (1,000 replicates)
│   ├── run_composite_account_check.py       # Composite account identifier sensitivity
│   ├── run_canonical_experiments.py         # Tables 3C & 6: Cost sensitivity & mixture delay
│   ├── run_label_propagation_diagnostics.py # Diagnostics A & B: Intra-burst metadata & behavioral profile
│   ├── test_leakage_unit_test.py            # Zero-leakage assertion suite
│   ├── test_synthetic_boundary_builder.py   # Synthetic boundary verification
│   ├── regenerate_all_tables.py             # Canonical table regeneration
│   └── generate_manuscript_figures.py       # Publication figure generation
└── out/
    └── results/                             # Saved CSV, JSON, and log outputs for all tables
```

## Data Access
Raw datasets and intermediate prediction caches (`out/predictions/*.npz`) are intentionally excluded from this repository. Download the public benchmark from Kaggle:
- **Kaggle IEEE-CIS Fraud Detection:** https://www.kaggle.com/c/ieee-fraud-detection

Place `train_transaction.csv` and `train_identity.csv` in `data/`, or set environment variables before running any script:
```bash
export IEEE_CIS_TX_PATH="/path/to/train_transaction.csv"
export IEEE_CIS_ID_PATH="/path/to/train_identity.csv"
```

## Mapping: Paper Tables to Scripts & Saved Result Files

> **Runtime Note:** Running the full end-to-end training, sweep, and 10,000-replicate clustered bootstrap pipeline from raw CSVs takes several hours on a standard multi-core laptop. All saved output files in `out/results/` can be inspected directly or used by `code/generate_manuscript_figures.py`.

| Paper Table / Analysis | Script | Saved Output File(s) in `out/results/` |
| :--- | :--- | :--- |
| **Zero-Leakage Verification** | `code/test_synthetic_boundary_builder.py`, `code/test_leakage_unit_test.py` | `step2_synthetic_boundary_test.log`, `step2_leakage_unit_test.log` |
| **Table 1 (Unified 3-Seed Benchmark)** | `code/run_unified_393_pipeline.py` | `unified_393_seed_summary.csv`, `unified_393_seed_variance.csv`, `unified_393_clustered_bootstrap.csv` |
| **Table 3A (Continuous Delay Sweep)** | `code/run_fine_grained_latency_sweep.py` | `fine_grained_latency_summary.csv`, `fine_grained_latency_results.csv`, `fine_grained_latency_summary_unified.csv` |
| **Table 3B (10,000-Replicate Clustered Bootstrap)** | `code/run_bootstrap_10k_unified.py` | `fine_grained_latency_bootstrap_10k_unified.csv` |
| **Table 3C (Cost-Schedule Sensitivity)** | `code/run_canonical_experiments.py` | `cost_schedule_sensitivity.csv` |
| **Table 4 (Account Density Sweep, 1,000 replicates)** | `code/run_density_sweep_393.py` | `density_sweep_393_summary.csv`, `density_sweep_393_results.csv` |
| **Table 5 (Constructive Label-Free Features, 1,000 replicates)** | `code/run_constructive_features.py` | `constructive_features_summary_unified.csv`, `constructive_features_results_unified.csv` |
| **Table 6 (Dual-Channel Mixture Delay)** | `code/run_canonical_experiments.py` | `mixture_delay_experiment.csv` |
| **Diagnostics A & B (Intra-Burst Linkage)** | `code/run_label_propagation_diagnostics.py` | `diagnostic_a_metadata_linkage.csv`, `diagnostic_b_behavioral_profile.csv` |
| **Composite Account Robustness** | `code/run_composite_account_check.py` | `composite_account_summary_unified.csv`, `composite_account_metrics.json` |

## Execution Commands
Run all commands from the repository root:

```bash
# 1. Zero-leakage unit tests
python code/test_synthetic_boundary_builder.py
python code/test_leakage_unit_test.py

# 2. Main 393-feature benchmark (Table 1) & fine-grained delay sweep (Table 3A)
python code/run_unified_393_pipeline.py
python code/run_fine_grained_latency_sweep.py

# 3. 10,000-replicate clustered bootstrap (Table 3B)
python code/run_bootstrap_10k_unified.py

# 4. Cost-schedule sensitivity & dual-channel mixture delay (Tables 3C & 6)
python code/run_canonical_experiments.py

# 5. Density sweep, constructive label-free features & composite account check (Tables 4 & 5)
python code/run_density_sweep_393.py
python code/run_constructive_features.py
python code/run_composite_account_check.py

# 6. Intra-burst diagnostics (Diagnostics A & B)
python code/run_label_propagation_diagnostics.py

# 7. Regenerate figures from saved results
python code/generate_manuscript_figures.py
```

## License
Released under the [MIT License](LICENSE).
