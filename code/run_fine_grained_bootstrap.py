#!/usr/bin/env python3
"""
Clustered Bootstrap (B=1,000) on 3-Seed-Averaged Predictions for Fine-Grained Latency Sweep
Computes 95% CIs and p-values against BOTH:
  1. Baseline (seeds 42-44 averaged)
  2. No-Fraud-Hist (seeds 42-44 averaged)
Across:
  - Delta ROC-AUC
  - Delta PR-AUC
  - Delta Total Cost ($) with exact 7-tier quantile schedule (side='right')

Saves to:
  out/results/fine_grained_latency_bootstrap.csv
"""

import os
import sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

t0 = time.time()
print("=" * 80)
print("CLUSTERED BOOTSTRAP (B=1,000) ON SEED-AVERAGED PREDICTIONS")
print("=" * 80)

# Load predictions
master = np.load('out/predictions/unified_393_test_predictions.npz', allow_pickle=True)
fine = np.load('out/predictions/fine_grained_predictions.npz', allow_pickle=True)

y = master['y_test']
amts = master['amounts_test']
card1 = master['card1_test']

# Reconstruct exact training quantile brackets
tx = pd.read_csv(os.environ.get("IEEE_CIS_TX_PATH", "data/train_transaction.csv"), usecols=['TransactionDT', 'TransactionAmt'])
tx = tx.sort_values('TransactionDT').reset_index(drop=True)
train_end = int(len(tx) * 0.70)
train_amts = tx['TransactionAmt'].iloc[:train_end]

BASE_UNIT_USD = 10.0
ESCALATION = (1, 2, 3, 4, 6, 10, 20)
QUANTILE_CUTS = (0.50, 0.75, 0.90, 0.97, 0.995, 0.999)
cutpoints = train_amts.quantile(list(QUANTILE_CUTS)).values
limits = list(cutpoints) + [float('inf')]
costs = [BASE_UNIT_USD * e for e in ESCALATION]
BRACKETS = list(zip(limits, costs))

limits_arr = np.array([lim for lim, _ in BRACKETS])
costs_arr = np.array([c for _, c in BRACKETS])

def fp_cost_from_amounts(fp_amts):
    if len(fp_amts) == 0:
        return 0.0
    idx = np.searchsorted(limits_arr, fp_amts, side='right')
    return float(costs_arr[np.minimum(idx, len(costs_arr) - 1)].sum())

def sweep_cost(y_true, pred, amounts):
    fn = float(np.sum(amounts[(y_true == 1) & (pred == 0)]))
    fp = fp_cost_from_amounts(amounts[(y_true == 0) & (pred == 1)])
    return fn + fp

THRESH_GRID = np.linspace(0.01, 0.90, 100)

def find_best_tau(proba, y_true, amounts):
    best_c = float('inf')
    best_tau = 0.5
    for tau in THRESH_GRID:
        pred = (proba >= tau).astype(int)
        c = sweep_cost(y_true, pred, amounts)
        if c < best_c:
            best_c = c
            best_tau = tau
    return best_tau

SEEDS = [42, 43, 44]

def get_seed_avg(cond_name):
    preds = []
    for s in SEEDS:
        if cond_name == 'Baseline':
            preds.append(master[f'Baseline_seed_{s}'])
        elif cond_name == 'D=0s (Instant)':
            preds.append(master[f'D=0 (k=1)_seed_{s}'])
        elif cond_name == 'D=30d (2592000s)':
            preds.append(master[f'D=30 (k=1)_seed_{s}'])
        elif cond_name == 'No-Fraud-Hist':
            preds.append(master[f'No-Fraud-Hist (k=1)_seed_{s}'])
        else:
            preds.append(fine[f'{cond_name}_seed_{s}'])
    return np.mean(preds, axis=0)

CONDITIONS = [
    ('D=0s (Instant)', 0, '0s'),
    ('D=5m (300s)', 300, '5m'),
    ('D=30m (1800s)', 1800, '30m'),
    ('D=1h (3600s)', 3600, '1h'),
    ('D=6h (21600s)', 21600, '6h'),
    ('D=24h (1d)', 86400, '24h'),
    ('D=72h (3d)', 259200, '72h'),
    ('D=7d (604800s)', 604800, '7d'),
    ('D=30d (2592000s)', 2592000, '30d'),
]

p_base = get_seed_avg('Baseline')
p_nfh = get_seed_avg('No-Fraud-Hist')

tau_base = find_best_tau(p_base, y, amts)
tau_nfh = find_best_tau(p_nfh, y, amts)

auc_base_full = roc_auc_score(y, p_base)
pr_base_full = average_precision_score(y, p_base)
cost_base_full = sweep_cost(y, (p_base >= tau_base).astype(int), amts)

auc_nfh_full = roc_auc_score(y, p_nfh)
pr_nfh_full = average_precision_score(y, p_nfh)
cost_nfh_full = sweep_cost(y, (p_nfh >= tau_nfh).astype(int), amts)

print(f"Full Test Baseline:     AUC={auc_base_full:.4f}, PR={pr_base_full:.4f}, Cost=${cost_base_full:,.0f}, tau*={tau_base:.4f}")
print(f"Full Test No-Fraud-Hist: AUC={auc_nfh_full:.4f}, PR={pr_nfh_full:.4f}, Cost=${cost_nfh_full:,.0f}, tau*={tau_nfh:.4f}")

unique_c = np.unique(card1)
c_to_idx = {c: [] for c in unique_c}
for i, c in enumerate(card1):
    c_to_idx[c].append(i)
c_idx_arr = [np.array(c_to_idx[c], dtype=np.int32) for c in unique_c]

B = 1000
np.random.seed(42)

# Pre-generate bootstrap resample index arrays
print(f"\nPre-generating {B} account-cluster resamples across {len(unique_c):,} unique accounts...")
boot_indices = []
for b in range(B):
    s_c = np.random.choice(len(unique_c), size=len(unique_c), replace=True)
    boot_indices.append(np.concatenate([c_idx_arr[i] for i in s_c]))

# Pre-compute metrics for Baseline and No-Fraud-Hist across all B replicates
print("Pre-computing Baseline and No-Fraud-Hist bootstrap distributions...")
t_pre = time.time()
auc_base_b = np.zeros(B)
pr_base_b = np.zeros(B)
cost_base_b = np.zeros(B)

auc_nfh_b = np.zeros(B)
pr_nfh_b = np.zeros(B)
cost_nfh_b = np.zeros(B)

pred_base_bin = (p_base >= tau_base).astype(int)
pred_nfh_bin = (p_nfh >= tau_nfh).astype(int)

for b in range(B):
    b_idx = boot_indices[b]
    yb = y[b_idx]
    ab = amts[b_idx]
    
    auc_base_b[b] = roc_auc_score(yb, p_base[b_idx])
    pr_base_b[b] = average_precision_score(yb, p_base[b_idx])
    cost_base_b[b] = sweep_cost(yb, pred_base_bin[b_idx], ab)
    
    auc_nfh_b[b] = roc_auc_score(yb, p_nfh[b_idx])
    pr_nfh_b[b] = average_precision_score(yb, p_nfh[b_idx])
    cost_nfh_b[b] = sweep_cost(yb, pred_nfh_bin[b_idx], ab)

print(f"Pre-computation complete in {time.time()-t_pre:.1f}s.")

# Now evaluate each latency condition
print("\nEvaluating 9 latency conditions...")
results = []

for cond_name, d_sec, label in CONDITIONS:
    t_c = time.time()
    p_c = get_seed_avg(cond_name)
    tau_c = find_best_tau(p_c, y, amts)
    pred_c_bin = (p_c >= tau_c).astype(int)
    
    # Point estimates on full test set
    pt_auc = roc_auc_score(y, p_c)
    pt_pr = average_precision_score(y, p_c)
    pt_cost = sweep_cost(y, pred_c_bin, amts)
    
    pe_auc_base = pt_auc - auc_base_full
    pe_auc_nfh = pt_auc - auc_nfh_full
    pe_pr_base = pt_pr - pr_base_full
    pe_pr_nfh = pt_pr - pr_nfh_full
    pe_cost_base = pt_cost - cost_base_full
    pe_cost_nfh = pt_cost - cost_nfh_full
    
    d_auc_base = np.zeros(B)
    d_auc_nfh = np.zeros(B)
    d_pr_base = np.zeros(B)
    d_pr_nfh = np.zeros(B)
    d_cost_base = np.zeros(B)
    d_cost_nfh = np.zeros(B)
    
    for b in range(B):
        b_idx = boot_indices[b]
        yb = y[b_idx]
        ab = amts[b_idx]
        
        auc_c = roc_auc_score(yb, p_c[b_idx])
        pr_c = average_precision_score(yb, p_c[b_idx])
        cost_c = sweep_cost(yb, pred_c_bin[b_idx], ab)
        
        d_auc_base[b] = auc_c - auc_base_b[b]
        d_auc_nfh[b] = auc_c - auc_nfh_b[b]
        d_pr_base[b] = pr_c - pr_base_b[b]
        d_pr_nfh[b] = pr_c - pr_nfh_b[b]
        d_cost_base[b] = cost_c - cost_base_b[b]
        d_cost_nfh[b] = cost_c - cost_nfh_b[b]
        
    p_auc_base = 2.0 * min(np.mean(d_auc_base <= 0), np.mean(d_auc_base >= 0))
    p_auc_nfh = 2.0 * min(np.mean(d_auc_nfh <= 0), np.mean(d_auc_nfh >= 0))
    p_cost_base = 2.0 * min(np.mean(d_cost_base <= 0), np.mean(d_cost_base >= 0))
    p_cost_nfh = 2.0 * min(np.mean(d_cost_nfh <= 0), np.mean(d_cost_nfh >= 0))
    
    results.append({
        'delay': label,
        'delay_sec': d_sec,
        'roc_auc': pt_auc,
        'pr_auc': pt_pr,
        'total_cost': pt_cost,
        'tau': tau_c,
        
        'd_auc_base_pt': pe_auc_base,
        'd_auc_base_ci_low': np.percentile(d_auc_base, 2.5),
        'd_auc_base_ci_high': np.percentile(d_auc_base, 97.5),
        'p_auc_base': min(1.0, float(p_auc_base)),
        
        'd_auc_nfh_pt': pe_auc_nfh,
        'd_auc_nfh_ci_low': np.percentile(d_auc_nfh, 2.5),
        'd_auc_nfh_ci_high': np.percentile(d_auc_nfh, 97.5),
        'p_auc_nfh': min(1.0, float(p_auc_nfh)),
        
        'd_pr_base_pt': pe_pr_base,
        'd_pr_base_ci_low': np.percentile(d_pr_base, 2.5),
        'd_pr_base_ci_high': np.percentile(d_pr_base, 97.5),
        
        'd_pr_nfh_pt': pe_pr_nfh,
        'd_pr_nfh_ci_low': np.percentile(d_pr_nfh, 2.5),
        'd_pr_nfh_ci_high': np.percentile(d_pr_nfh, 97.5),
        
        'd_cost_base_pt': pe_cost_base,
        'd_cost_base_ci_low': np.percentile(d_cost_base, 2.5),
        'd_cost_base_ci_high': np.percentile(d_cost_base, 97.5),
        'p_cost_base': min(1.0, float(p_cost_base)),
        
        'd_cost_nfh_pt': pe_cost_nfh,
        'd_cost_nfh_ci_low': np.percentile(d_cost_nfh, 2.5),
        'd_cost_nfh_ci_high': np.percentile(d_cost_nfh, 97.5),
        'p_cost_nfh': min(1.0, float(p_cost_nfh)),
    })
    
    print(f"[{label:>3}] AUC={pt_auc:.4f} (vs Base: {pe_auc_base:+.4f} [{np.percentile(d_auc_base, 2.5):+.4f}, {np.percentile(d_auc_base, 97.5):+.4f}], p={p_auc_base:.3f} | vs NFH: {pe_auc_nfh:+.4f} [{np.percentile(d_auc_nfh, 2.5):+.4f}, {np.percentile(d_auc_nfh, 97.5):+.4f}], p={p_auc_nfh:.3f}) ({time.time()-t_c:.1f}s)", flush=True)

df_out = pd.DataFrame(results)
df_out.to_csv('out/results/fine_grained_latency_bootstrap.csv', index=False)
print("\n" + "=" * 80)
print(f"ALL 9 CONDITIONS COMPLETED AND SAVED IN {time.time()-t0:.1f}s.")
print("=" * 80)
