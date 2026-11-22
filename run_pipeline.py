#!/usr/bin/env python3
"""Main execution script for replicating streaming fraud verification latency benchmarks."""
import argparse
import sys
from src.cost_eval import CUTPOINTS, TIER_COSTS, evaluate_cost, tune_tau_validation

def main():
    parser = argparse.ArgumentParser(description="Streaming Fraud Verification Latency Pipeline")
    parser.add_argument("--data_path", type=str, default="train_transaction.csv", help="Path to IEEE-CIS transaction CSV")
    parser.add_argument("--k_lag", type=int, default=1, help="Positional lag constraint")
    parser.add_argument("--delay_hours", type=float, default=24.0, help="Verification delay in hours")
    args = parser.parse_args()

    print("=" * 70)
    print("STREAMING FRAUD DETECTION: VERIFICATION LATENCY BENCHMARK")
    print(f"Positional Lag (k): {args.k_lag}")
    print(f"Delay (hours): {args.delay_hours}")
    print(f"Canonical Tier Cuts: {CUTPOINTS.tolist()}")
    print(f"Canonical Tier Costs: {TIER_COSTS.tolist()}")
    print("=" * 70)

if __name__ == "__main__":
    main()
