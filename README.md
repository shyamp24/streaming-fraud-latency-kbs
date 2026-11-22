# Streaming Fraud Detection under Verification Latency

This repository provides reproduction code for the paper:
**"When Does History Help? The Vulnerability of Longitudinal Account Features to Verification Latency in Streaming Fraud Detection"** (*Knowledge-Based Systems*, Elsevier).

## Overview
Evaluates the impact of ground-truth verification delay ($D$) and positional lag ($k$) on longitudinal account representations in high-throughput transaction streams using the IEEE-CIS fraud detection benchmark.

## Repository Structure
```
├── README.md
├── requirements.txt
├── run_pipeline.py
└── src/
    ├── __init__.py
    ├── data_loader.py         # Chronological data partition loader
    ├── feature_engineering.py # Positional lag (k) and verification delay (D)
    ├── cost_eval.py           # Canonical 7-tier operational cost model & threshold tuning
    ├── dual_channel.py        # Dual-channel mixed-delay simulation
    └── label_free.py          # Delay-invariant behavioral feature extraction
```

## Setup & Requirements
Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

## Data Access
The benchmark dataset is publicly available on Kaggle via Vesta Corporation:
- **Kaggle IEEE-CIS Fraud Detection**: https://www.kaggle.com/c/ieee-fraud-detection

Download `train_transaction.csv` and place it in the project root or provide the file path via `--data_path`.

## Execution
Run the baseline evaluation pipeline:
```bash
python run_pipeline.py --data_path train_transaction.csv --k_lag 1 --delay_hours 24
```
