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
│   ├── run_fine_grained_latency_sweep.py    # Table 3: Continuous delay sweep (0s to 30d)
│   ├── run_bootstrap_10k_unified.py         # Tables 3 & 3B: 10,000-replicate clustered bootstrap
│   ├── run_fine_grained_bootstrap.py        # 1,000-replicate clustered bootstrap helper
│   ├── run_clustered_bootstrap_1k.py        # 1,000-replicate clustered bootstrap on the Table 1 predictions
│   ├── run_canonical_experiments.py         # Tables 3C & 3D: Cost sensitivity & dual-channel mixture
│   ├── run_constructive_features.py         # Table 4: Constructive label-free features (1,000 replicates)
│   ├── run_composite_account_check.py       # Table 5: Composite account robustness (1,000 replicates)
│   ├── run_density_sweep_393.py             # Table 6: Account linkability & density sweep (25%, 50%, 100%)
│   ├── run_label_propagation_diagnostics.py # Diagnostics A & B: Intra-burst metadata & behavioral profile
│   ├── test_leakage_unit_test.py            # Zero-leakage assertion suite
│   ├── test_synthetic_boundary_builder.py   # Synthetic boundary verification
│   ├── regenerate_all_tables.py             # Canonical table regeneration
│   └── generate_manuscript_figures.py       # Publication figure & Table 2 burst-interval generation
└── out/
    └── results/                             # Saved CSV, JSON, and log outputs for all tables
```

## Data Access & Prediction Caches
Raw datasets and intermediate prediction caches (`out/predictions/*.npz`) are intentionally excluded from this repository (`.gitignore`). Download the public benchmark from Kaggle:
- **Kaggle IEEE-CIS Fraud Detection:** https://www.kaggle.com/c/ieee-fraud-detection

Place `train_transaction.csv` and `train_identity.csv` in `data/`, or set environment variables before running any script:
```bash
export IEEE_CIS_TX_PATH="/path/to/train_transaction.csv"
export IEEE_CIS_ID_PATH="/path/to/train_identity.csv"
```

> **Important Note on Prediction Files (`out/predictions/*.npz`):** `code/run_bootstrap_10k_unified.py` loads `out/predictions/unified_393_test_predictions.npz` and `out/predictions/fine_grained_predictions.npz`, which are not stored in this GitHub repository. To re-run the 10,000-replicate bootstrap from scratch, you must first run `code/run_unified_393_pipeline.py` and `code/run_fine_grained_latency_sweep.py` to generate those `.npz` files (several hours on a standard multi-core laptop).

## Mapping: Paper Tables to Scripts & Saved Result Files

> **Runtime Note:** Running the full end-to-end training, sweep, and 10,000-replicate clustered bootstrap pipeline from raw CSVs takes several hours on a standard multi-core laptop. All saved output files in `out/results/` can be inspected directly or used by `code/generate_manuscript_figures.py`.

| Paper Table / Analysis | Script | Saved Output File(s) in `out/results/` |
| :--- | :--- | :--- |
| **Zero-Leakage Verification** | `code/test_synthetic_boundary_builder.py`, `code/test_leakage_unit_test.py` | `synthetic_boundary_test.log`, `leakage_unit_test.log` |
| **Table 1 (Master Unified 3-Seed Benchmark)** | `code/run_unified_393_pipeline.py` | `unified_393_seed_summary.csv`, `unified_393_seed_variance.csv`, `unified_393_clustered_bootstrap.csv` |
| **Table 2 (Intra-Burst Clustering Statistics)** | `code/generate_manuscript_figures.py` | Computed directly from `train_transaction.csv` (`Card1` inter-burst intervals) |
| **Table 3 (Continuous Latency Decay & 10,000-Replicate Bootstrap vs. Baseline)** | `code/run_fine_grained_latency_sweep.py`, `code/run_bootstrap_10k_unified.py` | `fine_grained_latency_summary_unified.csv`, `fine_grained_latency_bootstrap_10k_unified.csv` |
| **Table 3B (10,000-Replicate Bootstrap vs. `No-Fraud-Hist`)** | `code/run_bootstrap_10k_unified.py` | `fine_grained_latency_bootstrap_10k_unified.csv` |
| **Table 3C (Cost-Schedule Sensitivity)** | `code/run_canonical_experiments.py` | `cost_schedule_sensitivity.csv` |
| **Table 3D (Dual-Channel Mixture Verification Delay)** | `code/run_canonical_experiments.py` | `mixture_delay_experiment.csv` |
| **Table 4 (Constructive Label-Free Features, 1,000 replicates)** | `code/run_constructive_features.py`, `code/regenerate_all_tables.py` | `constructive_features_summary_unified.csv`, `constructive_features_results_unified.csv` |
| **Table 5 (Composite Account Robustness, 1,000 replicates)** | `code/run_composite_account_check.py`, `code/regenerate_all_tables.py` | `composite_account_summary_unified.csv`, `composite_account_metrics.json` |
| **Table 6 (Account Linkability & Density Sweep)** | `code/run_density_sweep_393.py` | `density_sweep_393_summary.csv`, `density_sweep_393_results.csv` |
| **Diagnostics A & B (Intra-Burst Linkage & Profile)** | `code/run_label_propagation_diagnostics.py` | `diagnostic_a_metadata_linkage.csv`, `diagnostic_b_behavioral_profile.csv` |

## Execution Commands
Run all commands from the repository root:

```bash
# 1. Zero-leakage unit tests
python code/test_synthetic_boundary_builder.py
python code/test_leakage_unit_test.py

# 2. Main 393-feature benchmark (Table 1) & fine-grained delay sweep (Table 3)
#    (Generates out/predictions/*.npz required by the bootstrap in step 3)
python code/run_unified_393_pipeline.py
python code/run_fine_grained_latency_sweep.py

# 3. 10,000-replicate clustered bootstrap (Tables 3 & 3B; requires .npz files from step 2 above)
python code/run_bootstrap_10k_unified.py

# 4. Cost-schedule sensitivity & dual-channel mixture delay (Tables 3C & 3D)
python code/run_canonical_experiments.py

# 5. Constructive label-free features, composite account check & density sweep (Tables 4, 5 & 6)
python code/run_constructive_features.py
python code/run_composite_account_check.py
python code/regenerate_all_tables.py
python code/run_density_sweep_393.py

# 6. Intra-burst diagnostics (Diagnostics A & B)
python code/run_label_propagation_diagnostics.py

# 7. Regenerate figures & Table 2 burst intervals
python code/generate_manuscript_figures.py
```

## License
Released under the [MIT License](LICENSE).
