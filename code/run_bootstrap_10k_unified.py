#!/usr/bin/env python3
"""
High-Resolution Account-Clustered Bootstrap (B=10,000) for IEEE-CIS Streaming Experiment
Evaluates:
- 10,000 resamples across 7,080 unique card1 account clusters in the test partition (N=118,109).
- Evaluates pairwise differences for ROC-AUC, PR-AUC, and Total Cost.
- Computes empirical p-values with resolution down to 0.0001 (p < 0.0002 for 0/10,000 extremes).
- Computes Holm-Bonferroni step-down adjusted p-values alongside raw p-values.
- Saves: out/results/fine_grained_latency_bootstrap_10k_unified.csv
"""

import os, sys, time, warnings
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import Parallel, delayed

sys.path.append('code')
from cost_eval import fp_cost_vectorized, evaluate_cost

t0 = time.time()
print("=" * 80)
print("HIGH-RESOLUTION CLUSTERED BOOTSTRAP (B = 10,000 REPLICATES)")
print("=" * 80)

# 1. Load canonical test predictions
master = np.load('out/predictions/unified_393_test_predictions.npz', allow_pickle=True)
fine = np.load('out/predictions/fine_grained_predictions.npz', allow_pickle=True)

y_te = master['y_test']
amts_te = master['amounts_test']
card1_te = master['card1_test']
N = len(y_te)
assert N == 118109, f"Expected 118,109 test transactions, got {N}"

unique_cards = np.unique(card1_te)
n_clusters = len(unique_cards)
assert n_clusters == 7080, f"Expected 7,080 unique cards, got {n_clusters}"
print(f"Loaded {N:,} test transactions across {n_clusters:,} unique card1 clusters.")

# Precompute cluster index mapping
c_to_idx = {c: [] for c in unique_cards}
for i, c in enumerate(card1_te):
    c_to_idx[c].append(i)
c_idx_arr = [np.array(c_to_idx[c], dtype=np.int32) for c in unique_cards]

B = 10000
np.random.seed(42)

print(f"\nPre-generating cluster index matrix for {B:,} resamples (memory-efficient)...")
t_gen = time.time()
# Shape (B, n_clusters) takes only ~280 MB RAM
boot_clusters = np.random.choice(n_clusters, size=(B, n_clusters), replace=True).astype(np.int32)
print(f"Matrix generated in {time.time()-t_gen:.1f}s.")

# Frozen Validation Taus
SEEDS = [42, 43, 44]
TAU_BASE = {42: 0.378586, 43: 0.333636, 44: 0.405556}
TAU_D0 = {42: 0.351616, 43: 0.396566, 44: 0.342626}
TAU_D30 = {42: 0.387576, 43: 0.414545, 44: 0.414545}
TAU_NFH = {42: 0.414545, 43: 0.396566, 44: 0.450505}

TAU_FINE = {
    'D=5m (300s)': {42: 0.387576, 43: 0.378586, 44: 0.378586},
    'D=30m (1800s)': {42: 0.423535, 43: 0.306667, 44: 0.342626},
    'D=1h (3600s)': {42: 0.387576, 43: 0.324646, 44: 0.450505},
    'D=6h (21600s)': {42: 0.405556, 43: 0.450505, 44: 0.432525},
    'D=24h (1d)': {42: 0.450505, 43: 0.405556, 44: 0.360606},
    'D=72h (3d)': {42: 0.432525, 43: 0.513434, 44: 0.441515},
    'D=7d (604800s)': {42: 0.477475, 43: 0.468485, 44: 0.423535},
}

# Setup all 11 conditions
CONDITIONS = {
    'Baseline': ([master[f'Baseline_seed_{s}'] for s in SEEDS], [TAU_BASE[s] for s in SEEDS]),
    'No-Fraud-Hist': ([master[f'No-Fraud-Hist (k=1)_seed_{s}'] for s in SEEDS], [TAU_NFH[s] for s in SEEDS]),
    'D=0s (Instant)': ([master[f'D=0 (k=1)_seed_{s}'] for s in SEEDS], [TAU_D0[s] for s in SEEDS]),
    'D=5m (300s)': ([fine[f'D=5m (300s)_seed_{s}'] for s in SEEDS], [TAU_FINE['D=5m (300s)'][s] for s in SEEDS]),
    'D=30m (1800s)': ([fine[f'D=30m (1800s)_seed_{s}'] for s in SEEDS], [TAU_FINE['D=30m (1800s)'][s] for s in SEEDS]),
    'D=1h (3600s)': ([fine[f'D=1h (3600s)_seed_{s}'] for s in SEEDS], [TAU_FINE['D=1h (3600s)'][s] for s in SEEDS]),
    'D=6h (21600s)': ([fine[f'D=6h (21600s)_seed_{s}'] for s in SEEDS], [TAU_FINE['D=6h (21600s)'][s] for s in SEEDS]),
    'D=24h (1d)': ([fine[f'D=24h (1d)_seed_{s}'] for s in SEEDS], [TAU_FINE['D=24h (1d)'][s] for s in SEEDS]),
    'D=72h (3d)': ([fine[f'D=72h (3d)_seed_{s}'] for s in SEEDS], [TAU_FINE['D=72h (3d)'][s] for s in SEEDS]),
    'D=7d (604800s)': ([fine[f'D=7d (604800s)_seed_{s}'] for s in SEEDS], [TAU_FINE['D=7d (604800s)'][s] for s in SEEDS]),
    'D=30d (2592000s)': ([master[f'D=30 (k=1)_seed_{s}'] for s in SEEDS], [TAU_D30[s] for s in SEEDS]),
}

# Pre-calculate p_avg and boolean masks for each condition
cond_eval_meta = {}
for name, (preds, taus) in CONDITIONS.items():
    p_avg = np.mean(preds, axis=0)
    masks = [(p >= t).astype(int) for p, t in zip(preds, taus)]
    cond_eval_meta[name] = (p_avg, masks)

print("\nExecuting B=10,000 resamples across 4 parallel workers (evaluating all conditions per resample)...")

# Define chunked worker
def run_resample_chunk(b_range):
    chunk_results = {c: {'auc': [], 'pr': [], 'cost': []} for c in CONDITIONS}
    for b in b_range:
        b_idx = np.concatenate([c_idx_arr[i] for i in boot_clusters[b]])
        yb = y_te[b_idx]
        ab = amts_te[b_idx]
        
        for name, (p_avg, masks) in cond_eval_meta.items():
            pb = p_avg[b_idx]
            auc_val = roc_auc_score(yb, pb)
            pr_val = average_precision_score(yb, pb)
            cost_val = np.mean([
                float(np.sum(ab[(yb == 1) & (m[b_idx] == 0)])) + fp_cost_vectorized(ab[(yb == 0) & (m[b_idx] == 1)])
                for m in masks
            ])
            chunk_results[name]['auc'].append(auc_val)
            chunk_results[name]['pr'].append(pr_val)
            chunk_results[name]['cost'].append(cost_val)
    return chunk_results

N_JOBS = 4
chunk_size = B // N_JOBS
chunks = [range(i * chunk_size, (i + 1) * chunk_size if i < N_JOBS - 1 else B) for i in range(N_JOBS)]

t_eval_start = time.time()
parallel_out = Parallel(n_jobs=N_JOBS)(delayed(run_resample_chunk)(c) for c in chunks)

# Aggregate chunks
boot_data = {}
for name in CONDITIONS:
    auc_all = np.concatenate([out[name]['auc'] for out in parallel_out])
    pr_all = np.concatenate([out[name]['pr'] for out in parallel_out])
    cost_all = np.concatenate([out[name]['cost'] for out in parallel_out])
    boot_data[name] = (auc_all, pr_all, cost_all)

print(f"All {B:,} resamples evaluated in {time.time()-t_eval_start:.1f}s.")

# Compute pairwise comparisons
print("\nComputing pairwise comparisons, empirical p-values, and confidence intervals...")
COMP_SPECS = [
    ('D=0s (Instant)', 'Baseline', 'Instantaneous feedback oracle vs Baseline'),
    ('D=0s (Instant)', 'No-Fraud-Hist', 'Instantaneous feedback oracle vs Non-Label History'),
    ('D=5m (300s)', 'Baseline', '5-minute verification delay vs Baseline'),
    ('D=5m (300s)', 'No-Fraud-Hist', '5-minute verification delay vs Non-Label History'),
    ('D=30m (1800s)', 'Baseline', '30-minute verification delay vs Baseline'),
    ('D=30m (1800s)', 'No-Fraud-Hist', '30-minute verification delay vs Non-Label History'),
    ('D=1h (3600s)', 'Baseline', '1-hour verification delay vs Baseline'),
    ('D=1h (3600s)', 'No-Fraud-Hist', '1-hour verification delay vs Non-Label History'),
    ('D=6h (21600s)', 'Baseline', '6-hour verification delay vs Baseline'),
    ('D=6h (21600s)', 'No-Fraud-Hist', '6-hour verification delay vs Non-Label History'),
    ('D=24h (1d)', 'Baseline', '24-hour verification delay vs Baseline'),
    ('D=24h (1d)', 'No-Fraud-Hist', '24-hour verification delay vs Non-Label History'),
    ('D=72h (3d)', 'Baseline', '3-day verification delay vs Baseline'),
    ('D=72h (3d)', 'No-Fraud-Hist', '3-day verification delay vs Non-Label History'),
    ('D=7d (604800s)', 'Baseline', '7-day verification delay vs Baseline'),
    ('D=7d (604800s)', 'No-Fraud-Hist', '7-day verification delay vs Non-Label History'),
    ('D=30d (2592000s)', 'Baseline', '30-day realistic chargeback delay vs Baseline'),
    ('D=30d (2592000s)', 'No-Fraud-Hist', '30-day realistic chargeback delay vs Non-Label History'),
    ('No-Fraud-Hist', 'Baseline', 'Non-Label History vs Baseline'),
]

comp_rows = []
for cand, ref, desc in COMP_SPECS:
    cand_auc_b, cand_pr_b, cand_cost_b = boot_data[cand]
    ref_auc_b, ref_pr_b, ref_cost_b = boot_data[ref]
    
    cand_preds, cand_taus = CONDITIONS[cand]
    ref_preds, ref_taus = CONDITIONS[ref]
    
    # Point estimates on test partition
    d_auc_pt = roc_auc_score(y_te, cond_eval_meta[cand][0]) - roc_auc_score(y_te, cond_eval_meta[ref][0])
    d_pr_pt = average_precision_score(y_te, cond_eval_meta[cand][0]) - average_precision_score(y_te, cond_eval_meta[ref][0])
    
    c_cand_pts = [evaluate_cost(y_te, p, amts_te, t)[2] for p, t in zip(cand_preds, cand_taus)]
    c_ref_pts = [evaluate_cost(y_te, p, amts_te, t)[2] for p, t in zip(ref_preds, ref_taus)]
    d_cost_pt = float(np.mean(c_cand_pts) - np.mean(c_ref_pts))
    
    d_auc_b = cand_auc_b - ref_auc_b
    d_pr_b = cand_pr_b - ref_pr_b
    d_cost_b = cand_cost_b - ref_cost_b
    
    # Two-tailed empirical p-values (minimum resolvable: 2 / 10,000 = 0.0002)
    p_auc = 2.0 * min(np.mean(d_auc_b <= 0), np.mean(d_auc_b >= 0))
    p_cost = 2.0 * min(np.mean(d_cost_b <= 0), np.mean(d_cost_b >= 0))
    
    comp_rows.append({
        'comparison': f"{cand} vs {ref}",
        'description': desc,
        'd_auc_pt': d_auc_pt,
        'd_auc_ci_low': np.percentile(d_auc_b, 2.5),
        'd_auc_ci_high': np.percentile(d_auc_b, 97.5),
        'p_val_auc': p_auc,
        'd_pr_pt': d_pr_pt,
        'd_pr_ci_low': np.percentile(d_pr_b, 2.5),
        'd_pr_ci_high': np.percentile(d_pr_b, 97.5),
        'd_cost_pt': d_cost_pt,
        'd_cost_ci_low': np.percentile(d_cost_b, 2.5),
        'd_cost_ci_high': np.percentile(d_cost_b, 97.5),
        'p_val_cost': p_cost
    })
    print(f"  {cand:16s} vs {ref:14s}: dAUC={d_auc_pt:+.4f} (p={p_auc:.4f}), dCost=${d_cost_pt:+,.0f} (p={p_cost:.4f})")

df_boot_10k = pd.DataFrame(comp_rows)

# Holm-Bonferroni correction
def holm_adjust(p_vals):
    m = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    adj_p = np.zeros(m)
    current_max = 0.0
    for rank, idx in enumerate(sorted_idx):
        multiplier = m - rank
        raw_p = p_vals[idx]
        val = min(1.0, raw_p * multiplier)
        current_max = max(current_max, val)
        adj_p[idx] = current_max
    return adj_p

df_boot_10k['p_val_auc_holm'] = holm_adjust(df_boot_10k['p_val_auc'].values)
df_boot_10k['p_val_cost_holm'] = holm_adjust(df_boot_10k['p_val_cost'].values)

os.makedirs('out/results', exist_ok=True)
df_boot_10k.to_csv('out/results/fine_grained_latency_bootstrap_10k_unified.csv', index=False)
print("\nSaved out/results/fine_grained_latency_bootstrap_10k_unified.csv successfully.")
print(f"Total pipeline completed in {time.time()-t0:.1f}s.")
