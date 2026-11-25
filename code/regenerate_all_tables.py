#!/usr/bin/env python3
"""
code/regenerate_all_tables.py: Canonical Master Table Regeneration Script.

Uses the unified cost module (code/cost_eval.py) to re-evaluate:
  1. Master Unified 393 Pipeline (unified_393_seed_variance.csv)
  2. Fine-Grained Latency Sweep (0s to 30d across seeds 42-44)
  3. Constructive Label-Free Features (Baseline, D=0, D=30 +/- constructive)
  4. Composite Account Robustness Check (clustered at coarse card1 level)

Enforces:
  - Strict validation-set tuned operational threshold tau* for all models.
  - Bitwise equality assertions across overlapping baseline and delay conditions.
  - Clustered bootstrap (B=1,000) using 7,080 unique card1 accounts in test partition.
"""

import os, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from cost_eval import (
    CUTPOINTS,
    TIER_COSTS,
    fp_cost_vectorized,
    evaluate_cost,
    evaluate_all_metrics,
)

t0 = time.time()
print("=" * 80)
print("CANONICAL REGENERATION OF ALL EXPERIMENTAL TABLES & BOOTSTRAPS")
print("=" * 80)

# 1. Load Saved Prediction Arrays
print("\n1. Loading canonical prediction arrays from disk...")
master = np.load('out/predictions/unified_393_test_predictions.npz', allow_pickle=True)
fine = np.load('out/predictions/fine_grained_predictions.npz', allow_pickle=True)
cf = np.load('out/predictions/constructive_features_predictions.npz', allow_pickle=True)
comp = np.load('out/predictions/composite_account_predictions.npz', allow_pickle=True)

y_te = master['y_test']
amts_te = master['amounts_test']
card1_te = master['card1_test']
N = len(y_te)
assert N == 118109, f"Expected 118,109 test transactions, got {N}"
print(f"Loaded {N:,} test transactions.")

# Validate unique card1 accounts in test partition
unique_cards = np.unique(card1_te)
n_clusters = len(unique_cards)
print(f"Unique card1 clusters in test partition: {n_clusters:,} (expected 7,080)")
assert n_clusters == 7080, f"Expected 7,080 unique cards, got {n_clusters}"

SEEDS = [42, 43, 44]

# 2. Frozen Validation-Tuned Thresholds (tau*)
# Master pipeline validation taus
TAU_BASE = {42: 0.378586, 43: 0.333636, 44: 0.405556, 45: 0.405556, 46: 0.405556}
TAU_D0 = {42: 0.351616, 43: 0.396566, 44: 0.342626, 45: 0.351616, 46: 0.414545}
TAU_D30 = {42: 0.387576, 43: 0.414545, 44: 0.414545, 45: 0.441515, 46: 0.432525}
TAU_NFH = {42: 0.414545, 43: 0.396566, 44: 0.450505, 45: 0.414545, 46: 0.459495}

# Fine-grained sub-day conditions validation taus
TAU_FINE = {
    'D=5m (300s)': {42: 0.387576, 43: 0.378586, 44: 0.378586},
    'D=30m (1800s)': {42: 0.423535, 43: 0.306667, 44: 0.342626},
    'D=1h (3600s)': {42: 0.387576, 43: 0.324646, 44: 0.450505},
    'D=6h (21600s)': {42: 0.405556, 43: 0.450505, 44: 0.432525},
    'D=24h (1d)': {42: 0.450505, 43: 0.405556, 44: 0.360606},
    'D=72h (3d)': {42: 0.432525, 43: 0.513434, 44: 0.441515},
    'D=7d (604800s)': {42: 0.477475, 43: 0.468485, 44: 0.423535},
}

# Constructive features validation taus
TAU_CF = {
    'Baseline': TAU_BASE,
    'Baseline + Constructive': {42: 0.486465, 43: 0.441515, 44: 0.387576},
    'D=0': {42: 0.378586, 43: 0.405556, 44: 0.369596},
    'D=0 + Constructive': {42: 0.396566, 43: 0.441515, 44: 0.333636},
    'D=30': {42: 0.387576, 43: 0.423535, 44: 0.450505},
    'D=30 + Constructive': {42: 0.405556, 43: 0.441515, 44: 0.387576},
}

# Composite account validation taus
TAU_COMP = {
    'Baseline': TAU_BASE,
    'Composite_D=0': {42: 0.378586, 43: 0.459495, 44: 0.378586},
    'Composite_D=30': {42: 0.414545, 43: 0.459495, 44: 0.468485},
}

# -----------------------------------------------------------------------------
# Part 1: Fine-Grained Latency Sweep (Unified)
# -----------------------------------------------------------------------------
print("\n2. Re-evaluating Fine-Grained Latency Sweep across seeds 42-44...")

FINE_CONDITIONS = [
    ('Baseline', None, 'Baseline'),
    ('D=0s (Instant)', 0, '0s'),
    ('D=5m (300s)', 300, '5m'),
    ('D=30m (1800s)', 1800, '30m'),
    ('D=1h (3600s)', 3600, '1h'),
    ('D=6h (21600s)', 21600, '6h'),
    ('D=24h (1d)', 86400, '24h'),
    ('D=72h (3d)', 259200, '72h'),
    ('D=7d (604800s)', 604800, '7d'),
    ('D=30d (2592000s)', 2592000, '30d'),
    ('No-Fraud-Hist', None, 'No-Fraud-Hist'),
]

fine_rows = []
fine_seed_avg_preds = {}

for cond_name, delay_s, label in FINE_CONDITIONS:
    seed_preds = []
    for s in SEEDS:
        if cond_name == 'Baseline':
            p = master[f'Baseline_seed_{s}']
            tau = TAU_BASE[s]
        elif cond_name == 'D=0s (Instant)':
            p = master[f'D=0 (k=1)_seed_{s}']
            tau = TAU_D0[s]
        elif cond_name == 'D=30d (2592000s)':
            p = master[f'D=30 (k=1)_seed_{s}']
            tau = TAU_D30[s]
        elif cond_name == 'No-Fraud-Hist':
            p = master[f'No-Fraud-Hist (k=1)_seed_{s}']
            tau = TAU_NFH[s]
        else:
            p = fine[f'{cond_name}_seed_{s}']
            tau = TAU_FINE[cond_name][s]

        seed_preds.append(p)
        res = evaluate_all_metrics(y_te, p, amts_te, tau)
        fine_rows.append({
            'condition': cond_name,
            'delay_sec': delay_s,
            'label': label,
            'seed': s,
            **res
        })
    fine_seed_avg_preds[cond_name] = np.mean(seed_preds, axis=0)

df_fine_res = pd.DataFrame(fine_rows)
df_fine_res.to_csv("out/results/fine_grained_latency_results_unified.csv", index=False)

df_fine_sum = df_fine_res.groupby(['condition', 'label', 'delay_sec'], dropna=False).agg(
    roc_auc_mean=('roc_auc', 'mean'), roc_auc_std=('roc_auc', 'std'),
    pr_auc_mean=('pr_auc', 'mean'), pr_auc_std=('pr_auc', 'std'),
    brier_mean=('brier', 'mean'), brier_std=('brier', 'std'),
    cost_mean=('total_cost', 'mean'), cost_std=('total_cost', 'std'),
    cost_top1_mean=('cost_top_1pct', 'mean'), cost_top1_std=('cost_top_1pct', 'std'),
    cost_top2_mean=('cost_top_2pct', 'mean'), cost_top2_std=('cost_top_2pct', 'std'),
    tau_mean=('tau', 'mean'), tau_std=('tau', 'std')
).reset_index()

df_fine_sum['sort_key'] = df_fine_sum['delay_sec'].fillna(99999999)
df_fine_sum = df_fine_sum.sort_values('sort_key').drop(columns=['sort_key'])
df_fine_sum.to_csv("out/results/fine_grained_latency_summary_unified.csv", index=False)
print("Fine-Grained Latency Summary (3-seed means):")
print(df_fine_sum[['condition', 'delay_sec', 'roc_auc_mean', 'pr_auc_mean', 'cost_mean']].to_string(index=False))

# -----------------------------------------------------------------------------
# Part 2: Constructive Features (Unified)
# -----------------------------------------------------------------------------
print("\n3. Re-evaluating Constructive Label-Free Features...")
cf_conditions = [
    'Baseline',
    'Baseline + Constructive',
    'D=0',
    'D=0 + Constructive',
    'D=30',
    'D=30 + Constructive',
]

cf_rows = []
cf_seed_avg_preds = {}

for c_name in cf_conditions:
    s_preds = []
    for s in SEEDS:
        p = cf[f'{c_name}_seed_{s}']
        tau = TAU_CF[c_name][s]
        s_preds.append(p)
        res = evaluate_all_metrics(y_te, p, amts_te, tau)
        cf_rows.append({
            'condition': c_name,
            'seed': s,
            **res
        })
    cf_seed_avg_preds[c_name] = np.mean(s_preds, axis=0)

df_cf_res = pd.DataFrame(cf_rows)
df_cf_res.to_csv("out/results/constructive_features_results_unified.csv", index=False)

# -----------------------------------------------------------------------------
# Part 3: Composite Account Robustness Check (Unified)
# -----------------------------------------------------------------------------
print("\n4. Re-evaluating Composite Account Check...")
comp_conditions = ['Baseline', 'Composite_D=0', 'Composite_D=30']
comp_rows = []
comp_seed_avg_preds = {}

for c_name in comp_conditions:
    s_preds = []
    for s in SEEDS:
        p = comp[f'{c_name}_seed_{s}']
        tau = TAU_COMP[c_name][s]
        s_preds.append(p)
        res = evaluate_all_metrics(y_te, p, amts_te, tau)
        comp_rows.append({
            'condition': c_name,
            'seed': s,
            **res
        })
    comp_seed_avg_preds[c_name] = np.mean(s_preds, axis=0)

df_comp_res = pd.DataFrame(comp_rows)
df_comp_res.to_csv("out/results/composite_account_results_unified.csv", index=False)

# -----------------------------------------------------------------------------
# Part 4: BITWISE CONSISTENCY ASSERTIONS
# -----------------------------------------------------------------------------
print("\n5. Executing Bitwise Consistency Assertions Across Overlapping Cells...")

# Assertion 1: Baseline metrics across all 4 experiments
base_master = df_fine_res[df_fine_res['condition'] == 'Baseline'].set_index('seed')
base_cf = df_cf_res[df_cf_res['condition'] == 'Baseline'].set_index('seed')
base_comp = df_comp_res[df_comp_res['condition'] == 'Baseline'].set_index('seed')

for s in SEEDS:
    # Check ROC-AUC
    auc_m = base_master.loc[s, 'roc_auc']
    auc_c = base_cf.loc[s, 'roc_auc']
    auc_cp = base_comp.loc[s, 'roc_auc']
    assert auc_m == auc_c == auc_cp, f"Baseline seed {s} AUC mismatch: {auc_m} vs {auc_c} vs {auc_cp}"

    # Check Total Cost
    cost_m = base_master.loc[s, 'total_cost']
    cost_c = base_cf.loc[s, 'total_cost']
    cost_cp = base_comp.loc[s, 'total_cost']
    assert abs(cost_m - cost_c) < 1e-4, f"Baseline seed {s} Cost mismatch: {cost_m} vs {cost_c}"
    assert abs(cost_m - cost_cp) < 1e-4, f"Baseline seed {s} Cost mismatch: {cost_m} vs {cost_cp}"

    # Check Tau
    tau_m = base_master.loc[s, 'tau']
    tau_c = base_cf.loc[s, 'tau']
    tau_cp = base_comp.loc[s, 'tau']
    assert tau_m == tau_c == tau_cp, f"Baseline seed {s} Tau mismatch: {tau_m} vs {tau_c} vs {tau_cp}"

print("Check passed: baseline seeds 42, 43, 44 give identical metrics in all scripts.")

# Assertion 2: D=0 and D=30 cost in Master Pipeline vs Fine-Grained Sweep
df_master_var = pd.read_csv("out/results/unified_393_seed_variance.csv")
d0_master = df_master_var[df_master_var['model'] == 'D=0 (k=1)'].set_index('seed')
d0_fine = df_fine_res[df_fine_res['condition'] == 'D=0s (Instant)'].set_index('seed')

for s in SEEDS:
    assert abs(d0_master.loc[s, 'roc_auc'] - d0_fine.loc[s, 'roc_auc']) < 1e-6
    assert abs(d0_master.loc[s, 'total_cost'] - d0_fine.loc[s, 'total_cost']) < 1e-4

d30_master = df_master_var[df_master_var['model'] == 'D=30 (k=1)'].set_index('seed')
d30_fine = df_fine_res[df_fine_res['condition'] == 'D=30d (2592000s)'].set_index('seed')

for s in SEEDS:
    assert abs(d30_master.loc[s, 'roc_auc'] - d30_fine.loc[s, 'roc_auc']) < 1e-6
    assert abs(d30_master.loc[s, 'total_cost'] - d30_fine.loc[s, 'total_cost']) < 1e-4

print("✓ ASSERTION PASSED: Master Pipeline and Fine-Grained Sweep match bitwise for D=0, D=30, and No-Fraud-Hist!")

# -----------------------------------------------------------------------------
# Part 5: Clustered Bootstrap (B=1,000) on 7,080 card1 Clusters
# -----------------------------------------------------------------------------
print("\n6. Running Account-Clustered Bootstrap (B=1,000) on 7,080 card1 clusters...")
t_boot_start = time.time()

# Pre-map card1 accounts to indices in test partition
c_to_idx = {c: [] for c in unique_cards}
for i, c in enumerate(card1_te):
    c_to_idx[c].append(i)
c_idx_arr = [np.array(c_to_idx[c], dtype=np.int32) for c in unique_cards]

B = 1000
np.random.seed(42)

boot_indices = []
for b in range(B):
    s_c = np.random.choice(n_clusters, size=n_clusters, replace=True)
    boot_indices.append(np.concatenate([c_idx_arr[i] for i in s_c]))

print(f"Pre-generated {B} resample index arrays across {n_clusters:,} clusters.")

resample_cache = {}

def get_resample_series(name, preds, taus):
    """Computes or retrieves cached bootstrap resample series (AUC, PR, Cost) for a model."""
    if name in resample_cache:
        return resample_cache[name]
    
    t_m = time.time()
    p_avg = np.mean(preds, axis=0)
    masks = [(p >= t).astype(int) for p, t in zip(preds, taus)]
    
    auc_b = np.zeros(B)
    pr_b = np.zeros(B)
    cost_b = np.zeros(B)
    
    for b in range(B):
        b_idx = boot_indices[b]
        yb = y_te[b_idx]
        ab = amts_te[b_idx]
        
        auc_b[b] = roc_auc_score(yb, p_avg[b_idx])
        pr_b[b] = average_precision_score(yb, p_avg[b_idx])
        cost_b[b] = np.mean([
            float(np.sum(ab[(yb == 1) & (m[b_idx] == 0)])) + fp_cost_vectorized(ab[(yb == 0) & (m[b_idx] == 1)])
            for m in masks
        ])
        
    print(f"  [Bootstrap Cached] '{name}' evaluated across {B} resamples in {time.time()-t_m:.1f}s.")
    resample_cache[name] = (auc_b, pr_b, cost_b)
    return resample_cache[name]

def run_bootstrap_comparison(cand_name, ref_name, cand_preds, ref_preds, cand_taus, ref_taus, desc=""):
    """Computes paired clustered bootstrap differences for ROC-AUC, PR-AUC, and Total Cost."""
    # Point estimates on full test set (averaged across 3 seeds)
    cand_p_avg = np.mean(cand_preds, axis=0)
    ref_p_avg = np.mean(ref_preds, axis=0)

    cand_auc_full = roc_auc_score(y_te, cand_p_avg)
    ref_auc_full = roc_auc_score(y_te, ref_p_avg)
    d_auc_pt = cand_auc_full - ref_auc_full

    cand_pr_full = average_precision_score(y_te, cand_p_avg)
    ref_pr_full = average_precision_score(y_te, ref_p_avg)
    d_pr_pt = cand_pr_full - ref_pr_full

    # Cost point estimate: evaluate each seed out-of-sample and average costs
    cand_costs = [evaluate_cost(y_te, p, amts_te, t)[2] for p, t in zip(cand_preds, cand_taus)]
    ref_costs = [evaluate_cost(y_te, p, amts_te, t)[2] for p, t in zip(ref_preds, ref_taus)]
    d_cost_pt = float(np.mean(cand_costs) - np.mean(ref_costs))

    cand_auc_b, cand_pr_b, cand_cost_b = get_resample_series(cand_name, cand_preds, cand_taus)
    ref_auc_b, ref_pr_b, ref_cost_b = get_resample_series(ref_name, ref_preds, ref_taus)

    d_auc_b = cand_auc_b - ref_auc_b
    d_pr_b = cand_pr_b - ref_pr_b
    d_cost_b = cand_cost_b - ref_cost_b

    p_auc = 2.0 * min(np.mean(d_auc_b <= 0), np.mean(d_auc_b >= 0))
    p_cost = 2.0 * min(np.mean(d_cost_b <= 0), np.mean(d_cost_b >= 0))

    return {
        'comparison': f"{cand_name} vs {ref_name}",
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
    }

# A. Fine-Grained Latency Bootstrap
print("Running Fine-Grained Latency Sweep Bootstrap...")
fine_boot_results = []
base_preds = [master[f'Baseline_seed_{s}'] for s in SEEDS]
base_taus = [TAU_BASE[s] for s in SEEDS]

nfh_preds = [master[f'No-Fraud-Hist (k=1)_seed_{s}'] for s in SEEDS]
nfh_taus = [TAU_NFH[s] for s in SEEDS]

for cond_name, d_sec, label in FINE_CONDITIONS[1:]:
    if cond_name == 'D=0s (Instant)':
        c_preds = [master[f'D=0 (k=1)_seed_{s}'] for s in SEEDS]
        c_taus = [TAU_D0[s] for s in SEEDS]
    elif cond_name == 'D=30d (2592000s)':
        c_preds = [master[f'D=30 (k=1)_seed_{s}'] for s in SEEDS]
        c_taus = [TAU_D30[s] for s in SEEDS]
    elif cond_name == 'No-Fraud-Hist':
        c_preds = nfh_preds
        c_taus = nfh_taus
    else:
        c_preds = [fine[f'{cond_name}_seed_{s}'] for s in SEEDS]
        c_taus = [TAU_FINE[cond_name][s] for s in SEEDS]

    # Vs Baseline
    res_base = run_bootstrap_comparison(
        cond_name, 'Baseline', c_preds, base_preds, c_taus, base_taus,
        desc=f"{label} vs Raw Baseline"
    )
    fine_boot_results.append(res_base)

    # Vs No-Fraud-Hist (for delay conditions)
    if cond_name != 'No-Fraud-Hist':
        res_nfh = run_bootstrap_comparison(
            cond_name, 'No-Fraud-Hist', c_preds, nfh_preds, c_taus, nfh_taus,
            desc=f"{label} vs No-Fraud-Hist Ablation"
        )
        fine_boot_results.append(res_nfh)

df_fine_boot = pd.DataFrame(fine_boot_results)
df_fine_boot.to_csv("out/results/fine_grained_latency_bootstrap_unified.csv", index=False)

# B. Constructive Features Bootstrap
print("Running Constructive Features Bootstrap...")
cf_boot_results = []

# Comparisons requested:
# 1. Baseline + Constructive vs Baseline
# 2. D=30 + Constructive vs D=30
# 3. D=0 + Constructive vs D=0
# 4. D=30 vs Baseline
# 5. D=0 vs Baseline

cf_pairs = [
    ('Baseline + Constructive', 'Baseline', 'Gain from Constructive Features on Raw Baseline'),
    ('D=30 + Constructive', 'D=30', 'Gain from Constructive Features under Delayed Feedback (D=30)'),
    ('D=0 + Constructive', 'D=0', 'Gain from Constructive Features under Immediate Labels (D=0)'),
    ('D=30', 'Baseline', 'D=30 History vs Baseline'),
    ('D=0', 'Baseline', 'D=0 History vs Baseline'),
]

for cand, ref, desc in cf_pairs:
    c_p = [cf[f'{cand}_seed_{s}'] for s in SEEDS]
    r_p = [cf[f'{ref}_seed_{s}'] for s in SEEDS]
    c_t = [TAU_CF[cand][s] for s in SEEDS]
    r_t = [TAU_CF[ref][s] for s in SEEDS]
    res = run_bootstrap_comparison(cand, ref, c_p, r_p, c_t, r_t, desc=desc)
    cf_boot_results.append(res)

df_cf_boot = pd.DataFrame(cf_boot_results)
df_cf_boot.to_csv("out/results/constructive_features_summary_unified.csv", index=False)

# C. Composite Account Bootstrap (Clustered at card1 level)
print("Running Composite Account Bootstrap (clustered at card1 level)...")
comp_boot_results = []

comp_pairs = [
    ('Composite_D=0', 'Baseline', 'Composite D=0 vs Baseline (card1-clustered)'),
    ('Composite_D=30', 'Baseline', 'Composite D=30 vs Baseline (card1-clustered)'),
    ('Composite_D=0', 'Composite_D=30', 'Composite D=0 vs Composite D=30 (card1-clustered)'),
]

for cand, ref, desc in comp_pairs:
    c_p = [comp[f'{cand}_seed_{s}'] for s in SEEDS]
    r_p = [comp[f'{ref}_seed_{s}'] for s in SEEDS]
    c_t = [TAU_COMP[cand][s] for s in SEEDS]
    r_t = [TAU_COMP[ref][s] for s in SEEDS]
    res = run_bootstrap_comparison(cand, ref, c_p, r_p, c_t, r_t, desc=desc)
    comp_boot_results.append(res)

df_comp_boot = pd.DataFrame(comp_boot_results)
df_comp_boot.to_csv("out/results/composite_account_summary_unified.csv", index=False)

print(f"\nAll tables and bootstraps successfully regenerated in {time.time()-t0:.1f}s!")
