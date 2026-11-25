#!/usr/bin/env python3
"""
Unified 393-Feature Pipeline:
Standardized on the original ACXIS_IEEE_CIS_Prediction_FULL.ipynb specification.
Runs:
1. Baseline Detection (393 features, ROC-AUC = 0.8767)
2. Label Delay Grid (k=1 for D in {0, 7, 14, 30, 60} and No-Fraud-Hist; k in {3, 5, 10, 20} for D in {0, 30})
3. Fixed alert rate costs (top 1%, 2%, 5%) and threshold-free Brier score / PR-AUC / ROC-AUC
4. 5-Seed Training Variance evaluation (seeds 42, 43, 44, 45, 46 with subsample=0.8, colsample_bytree=0.8)
5. Save all test predictions to out/predictions/unified_393_test_predictions.npz with SHA256 checksums
6. Account-Clustered Bootstrap (B=1,000 replicates) centered on the exact saved predictions
"""

import sys
import os
import gc
import time
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, precision_score, recall_score, f1_score)

warnings.filterwarnings('ignore')

print("=" * 80)
print("UNIFIED 393-FEATURE PIPELINE (STANDARDIZED NOTEBOOK SPECIFICATION)")
print("=" * 80)

t_start = time.time()

TRANSACTION_PATH = os.environ.get("IEEE_CIS_TX_PATH", "data/train_transaction.csv")

# Hyperparameters matching original notebook
XGB_PARAMS = dict(n_estimators=500, max_depth=5, learning_rate=0.05,
                  scale_pos_weight=10, random_state=42, eval_metric='logloss',
                  enable_categorical=True, tree_method='hist')

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.10, 0.20
BASE_UNIT_USD = 10.0
ESCALATION = (1, 2, 3, 4, 6, 10, 20)
QUANTILE_CUTS = (0.50, 0.75, 0.90, 0.97, 0.995, 0.999)
THRESH_GRID = np.linspace(0.01, 0.90, 100)

def build_quantile_brackets(train_amounts, base_unit=BASE_UNIT_USD,
                            escalation=ESCALATION, quantile_cuts=QUANTILE_CUTS):
    cutpoints = train_amounts.quantile(list(quantile_cuts)).values
    limits = list(cutpoints) + [float('inf')]
    costs = [base_unit * e for e in escalation]
    return list(zip(limits, costs))

def fp_cost_from_amounts(fp_amts, brackets):
    if len(fp_amts) == 0:
        return 0.0
    costs = np.zeros(len(fp_amts), dtype=np.float64)
    prev_lim = -1.0
    for lim, c in brackets:
        m = (fp_amts > prev_lim) & (fp_amts <= lim)
        costs[m] = c
        prev_lim = lim
    return float(np.sum(costs))

def cost_opt(proba, y_true, amts, brackets):
    best_c = float('inf')
    best_tau = 0.5
    for tau in THRESH_GRID:
        pred = (proba >= tau).astype(int)
        fn = float(np.sum(amts[(y_true == 1) & (pred == 0)]))
        fp = fp_cost_from_amounts(amts[(y_true == 0) & (pred == 1)], brackets)
        tot = fn + fp
        if tot < best_c:
            best_c = tot
            best_tau = tau
    return best_tau, best_c

def evaluate_predictions(y_true, proba, amts, tau, brackets):
    pred = (proba >= tau).astype(int)
    fn = float(np.sum(amts[(y_true == 1) & (pred == 0)]))
    fp = fp_cost_from_amounts(amts[(y_true == 0) & (pred == 1)], brackets)
    
    # Fixed alert budget costs: top 1%, 2%, 5% alerts
    N = len(y_true)
    fixed_costs = {}
    for pct in [1.0, 2.0, 5.0]:
        k_alerts = int(np.round(N * (pct / 100.0)))
        cutoff_p = np.partition(proba, N - k_alerts)[N - k_alerts]
        pred_fixed = (proba >= cutoff_p).astype(int)
        fn_f = float(np.sum(amts[(y_true == 1) & (pred_fixed == 0)]))
        fp_f = fp_cost_from_amounts(amts[(y_true == 0) & (pred_fixed == 1)], brackets)
        fixed_costs[f'cost_top_{int(pct)}pct'] = fn_f + fp_f

    return {
        'roc_auc': float(roc_auc_score(y_true, proba)),
        'pr_auc': float(average_precision_score(y_true, proba)),
        'brier': float(brier_score_loss(y_true, proba)),
        'tau': float(tau),
        'precision': float(precision_score(y_true, pred, zero_division=0)),
        'recall': float(recall_score(y_true, pred, zero_division=0)),
        'f1': float(f1_score(y_true, pred, zero_division=0)),
        'cost_fn': fn,
        'cost_fp': fp,
        'total_cost': fn + fp,
        **fixed_costs
    }

print("1. Loading raw transaction dataset...")
t0 = time.time()
tx = pd.read_csv(TRANSACTION_PATH)
assert len(tx) == 590540, f"Expected 590,540 rows, got {len(tx)}"
print(f"Loaded {len(tx):,} rows in {time.time()-t0:.1f}s.")

# Rename TransactionAmt -> amount
tx = tx.rename(columns={'TransactionAmt': 'amount'})

# Drop TransactionID and any identity columns
tx = tx.drop(columns=['TransactionID'])

# Numeric and string categoricals
NUM_CATEGORICAL = ['card1', 'card2', 'card3', 'card5', 'addr1', 'addr2']
STR_CATEGORICAL = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain'] + [f'M{i}' for i in range(1, 10)]
CATEGORICAL_COLS = [c for c in STR_CATEGORICAL + NUM_CATEGORICAL if c in tx.columns]

def numeric_to_int_string(series):
    rounded = series.round()
    out = rounded.astype(object)
    mask = rounded.notna()
    out[mask] = rounded[mask].astype(np.int64).astype(str)
    out[~mask] = None
    return out

for col in NUM_CATEGORICAL:
    tx[col] = numeric_to_int_string(tx[col])
for col in CATEGORICAL_COLS:
    tx[col] = tx[col].astype('category')

# Time features
tx['hour_of_day'] = ((tx['TransactionDT'] // 3600) % 24).astype('int32')
tx['day_of_week_index'] = ((tx['TransactionDT'] // 86400) % 7).astype('int32')

# Downcast numerics
for col in tx.columns:
    if col not in CATEGORICAL_COLS and tx[col].dtype == 'float64':
        tx[col] = tx[col].astype('float32')
    elif col not in CATEGORICAL_COLS and tx[col].dtype == 'int64':
        tx[col] = tx[col].astype('int32')

# Chronological sort
tx = tx.sort_values('TransactionDT', kind='stable').reset_index(drop=True)

# Account ID from card1
tx['account_id'] = tx['card1'].cat.codes.astype('int32')

N = len(tx)
train_end = int(N * TRAIN_FRAC)
val_end = int(N * (TRAIN_FRAC + VAL_FRAC))

Y = tx['isFraud'].astype('int8').values
amounts = tx['amount'].values
tx_dts = tx['TransactionDT'].values.astype('float64')
acc_ids = tx['account_id'].values
card1_raw = tx['card1'].astype(str).values

BRACKETS = build_quantile_brackets(tx['amount'].iloc[:train_end])

BASE_FEATURES = [c for c in tx.columns
                 if c not in ('isFraud', 'TransactionDT', 'account_id')
                 and not c.startswith('hist_')]

print(f"Base feature count: {len(BASE_FEATURES)} features (expected 393).")
assert len(BASE_FEATURES) == 393, f"Expected 393 features, got {len(BASE_FEATURES)}"

# Dictionary to hold predictions for test set
test_predictions = {}
val_predictions = {}
optimal_taus = {}
grid_results = {}

# 2. Baseline Model
print("\n--- Training Baseline Detection Model ---")
t0 = time.time()
X_tr = tx.iloc[:train_end][BASE_FEATURES]
X_va = tx.iloc[train_end:val_end][BASE_FEATURES]
X_te = tx.iloc[val_end:][BASE_FEATURES]

xgb_base = XGBClassifier(**XGB_PARAMS)
xgb_base.fit(X_tr, Y[:train_end])

p_va_base = xgb_base.predict_proba(X_va)[:, 1]
p_te_base = xgb_base.predict_proba(X_te)[:, 1]

tau_base, _ = cost_opt(p_va_base, Y[train_end:val_end], amounts[train_end:val_end], BRACKETS)
res_base = evaluate_predictions(Y[val_end:], p_te_base, amounts[val_end:], tau_base, BRACKETS)

print(f"Baseline: ROC-AUC={res_base['roc_auc']:.4f}, PR-AUC={res_base['pr_auc']:.4f}, "
      f"Brier={res_base['brier']:.4f}, Cost=${res_base['total_cost']:,.0f}, tau*={tau_base:.4f} (trained in {time.time()-t0:.1f}s)")

test_predictions['Baseline'] = p_te_base
val_predictions['Baseline'] = p_va_base
optimal_taus['Baseline'] = tau_base
grid_results['Baseline'] = {'k': 0, 'delay_days': 'N/A', **res_base}

# Precompute structures for fast temporal feature extraction
print("\nIndexing fraud events per account for exact delay computation...")
account_frauds = {}
for ac, dt, is_fr in zip(acc_ids, tx_dts, Y):
    if is_fr == 1:
        account_frauds.setdefault(ac, []).append(dt)
account_frauds = {ac: np.array(v, dtype=np.float64) for ac, v in account_frauds.items()}

acc_indices = {}
for idx, ac in enumerate(acc_ids):
    acc_indices.setdefault(ac, []).append(idx)

# Precompute base account history (volume and value)
tmp = pd.DataFrame({
    'account_id': acc_ids,
    'amount': amounts,
    'TransactionDT': tx_dts,
    'isFraud': Y
})
g = tmp.groupby('account_id', sort=False)
tmp['_cnt'] = g.cumcount()
tmp['_sum'] = g['amount'].cumsum().astype('float32')
tmp['_max'] = g['amount'].cummax().astype('float32')
tmp['_amt2'] = amounts ** 2
tmp['_sumsq'] = tmp.groupby('account_id', sort=False)['_amt2'].cumsum().astype('float64')

first_dt = g['TransactionDT'].transform('first')
account_age_days = ((tx_dts - first_dt) / 86400.0).astype('float32')

def build_features_for_cell(k, delay_days):
    """Generates the 12 account-history features for given k and delay D."""
    g2 = tmp.groupby('account_id', sort=False)
    hist_cnt = g2['_cnt'].shift(k).fillna(0.0)
    hist_sum = g2['_sum'].shift(k).fillna(0.0)
    hist_sumsq = g2['_sumsq'].shift(k).fillna(0.0)
    hist_max = g2['_max'].shift(k)
    dt_last = g['TransactionDT'].shift(k)
    amt_last = g['amount'].shift(k)

    mean = (hist_sum / hist_cnt.clip(lower=1)).fillna(0.0)
    n = hist_cnt.clip(lower=0)
    var = (hist_sumsq - n * mean.astype('float64') ** 2) / (n - 1).clip(lower=1)

    hf = pd.DataFrame(index=tx.index)
    hf['hist_tx_count'] = hist_cnt.astype('float32')
    hf['hist_amt_sum'] = hist_sum.astype('float32')
    hf['hist_amt_mean'] = mean.astype('float32')
    hf['hist_amt_std'] = pd.Series(np.where(n > 1, np.sqrt(var.clip(lower=0)), np.nan), index=tx.index).astype('float32')
    hf['hist_amt_max'] = hist_max.astype('float32')
    hf['hist_amt_last'] = amt_last.astype('float32')
    hf['hist_time_since_last'] = (tx_dts - dt_last).astype('float32')
    hf['hist_account_age_days'] = account_age_days
    hf['hist_tx_per_day'] = (hf['hist_tx_count'] / (account_age_days + 1e-6)).astype('float32')

    if delay_days is None:
        # No fraud history ablation
        hf['hist_fraud_count'] = 0.0
        hf['hist_fraud_rate'] = 0.0
        hf['hist_fraud_time_since'] = np.nan
    elif delay_days == 0:
        # Unrealistic zero-delay (leaked)
        tmp['_fr'] = g['isFraud'].cumsum().astype('float32')
        tmp['_fr_dt'] = np.where(Y == 1, tx_dts, np.nan)
        g2_fr = tmp.groupby('account_id', sort=False)
        hist_fr = g2_fr['_fr'].shift(k).fillna(0.0)
        last_fr = g2_fr['_fr_dt'].shift(k).groupby(acc_ids).ffill()
        hf['hist_fraud_count'] = hist_fr.astype('float32')
        hf['hist_fraud_rate'] = (hf['hist_fraud_count'] / hf['hist_tx_count'].clip(lower=1)).astype('float32')
        hf['hist_fraud_time_since'] = (tx_dts - last_fr).astype('float32')
    else:
        # Realistic verification latency
        delay_sec = float(delay_days) * 86400.0
        fr_cnt_arr = np.zeros(N, dtype=np.float32)
        fr_time_arr = np.full(N, np.nan, dtype=np.float32)

        for ac, idxs in acc_indices.items():
            if ac not in account_frauds:
                continue
            fr_arr = account_frauds[ac]
            L = len(idxs)
            if L <= k:
                continue
            for pos in range(k, L):
                row_idx = idxs[pos]
                eval_dt = tx_dts[row_idx]
                cutoff_dt = eval_dt - delay_sec
                pos_k_idx = idxs[pos - k]
                max_allowed_dt = tx_dts[pos_k_idx]
                effective_cutoff = min(cutoff_dt, max_allowed_dt)
                
                n_fr = np.searchsorted(fr_arr, effective_cutoff, side='right')
                fr_cnt_arr[row_idx] = float(n_fr)
                if n_fr > 0:
                    fr_time_arr[row_idx] = float(eval_dt - fr_arr[n_fr - 1])

        hf['hist_fraud_count'] = fr_cnt_arr
        hf['hist_fraud_rate'] = (hf['hist_fraud_count'] / hf['hist_tx_count'].clip(lower=1)).astype('float32')
        hf['hist_fraud_time_since'] = fr_time_arr

    # Combine with base features
    full_X = pd.concat([tx[BASE_FEATURES], hf], axis=1)
    return full_X

# Define Grid of experiments
GRID_RUNS = [
    # k=1 primary sweep
    (1, 0, 'D=0 (k=1)'),
    (1, 7, 'D=7 (k=1)'),
    (1, 14, 'D=14 (k=1)'),
    (1, 30, 'D=30 (k=1)'),
    (1, 60, 'D=60 (k=1)'),
    (1, None, 'No-Fraud-Hist (k=1)'),
    # Cross-lead-time comparisons
    (3, 0, 'D=0 (k=3)'),
    (3, 30, 'D=30 (k=3)'),
    (5, 0, 'D=0 (k=5)'),
    (5, 30, 'D=30 (k=5)'),
    (10, 0, 'D=0 (k=10)'),
    (10, 30, 'D=30 (k=10)'),
    (20, 0, 'D=0 (k=20)'),
    (20, 30, 'D=30 (k=20)'),
]

print(f"\n--- Running Full k x D Grid ({len(GRID_RUNS)} models) ---")
for k, d, label in GRID_RUNS:
    t_cell = time.time()
    X_full = build_features_for_cell(k, d)
    
    X_tr_cell = X_full.iloc[:train_end]
    X_va_cell = X_full.iloc[train_end:val_end]
    X_te_cell = X_full.iloc[val_end:]
    
    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X_tr_cell, Y[:train_end])
    
    p_va = xgb.predict_proba(X_va_cell)[:, 1]
    p_te = xgb.predict_proba(X_te_cell)[:, 1]
    
    tau, _ = cost_opt(p_va, Y[train_end:val_end], amounts[train_end:val_end], BRACKETS)
    res = evaluate_predictions(Y[val_end:], p_te, amounts[val_end:], tau, BRACKETS)
    
    test_predictions[label] = p_te
    val_predictions[label] = p_va
    optimal_taus[label] = tau
    grid_results[label] = {'k': k, 'delay_days': str(d), **res}
    
    d_auc = res['roc_auc'] - res_base['roc_auc']
    d_cost = res['total_cost'] - res_base['total_cost']
    print(f"  [{label}] ROC-AUC={res['roc_auc']:.4f} (d={d_auc:+.4f}), PR-AUC={res['pr_auc']:.4f}, "
          f"Brier={res['brier']:.4f}, Cost=${res['total_cost']:,.0f} (d={d_cost:+,.0f}), tau*={tau:.4f} ({time.time()-t_cell:.1f}s)")
    
    del X_full, X_tr_cell, X_va_cell, X_te_cell, xgb
    gc.collect()

# 3. Multi-Seed Training Variance Analysis (Item 2)
print("\n" + "=" * 80)
print("MULTI-SEED TRAINING VARIANCE ANALYSIS (5 SEEDS WITH SUBSAMPLING)")
print("=" * 80)

SEEDS = [42, 43, 44, 45, 46]
VARIANCE_MODELS = [
    ('Baseline', None, None),
    ('D=0 (k=1)', 1, 0),
    ('D=30 (k=1)', 1, 30),
    ('No-Fraud-Hist (k=1)', 1, None),
]

seed_variance_results = []
seed_predictions = {}

for m_name, k, d in VARIANCE_MODELS:
    print(f"\nEvaluating 5 seeds for {m_name}...")
    if k is None:
        X_mat = tx[BASE_FEATURES]
    else:
        X_mat = build_features_for_cell(k, d)
        
    X_tr_mat = X_mat.iloc[:train_end]
    X_va_mat = X_mat.iloc[train_end:val_end]
    X_te_mat = X_mat.iloc[val_end:]
    
    for seed in SEEDS:
        t_seed = time.time()
        # Enable subsample=0.8 and colsample_bytree=0.8 so random seeds generate true stochastic variation
        params = XGB_PARAMS.copy()
        params['random_state'] = seed
        params['subsample'] = 0.8
        params['colsample_bytree'] = 0.8
        
        xgb_seed = XGBClassifier(**params)
        xgb_seed.fit(X_tr_mat, Y[:train_end])
        
        p_va_s = xgb_seed.predict_proba(X_va_mat)[:, 1]
        p_te_s = xgb_seed.predict_proba(X_te_mat)[:, 1]
        
        tau_s, _ = cost_opt(p_va_s, Y[train_end:val_end], amounts[train_end:val_end], BRACKETS)
        res_s = evaluate_predictions(Y[val_end:], p_te_s, amounts[val_end:], tau_s, BRACKETS)
        
        seed_predictions[f"{m_name}_seed_{seed}"] = p_te_s
        seed_variance_results.append({
            'model': m_name,
            'seed': seed,
            'roc_auc': res_s['roc_auc'],
            'pr_auc': res_s['pr_auc'],
            'brier': res_s['brier'],
            'tau': res_s['tau'],
            'recall': res_s['recall'],
            'precision': res_s['precision'],
            'f1': res_s['f1'],
            'total_cost': res_s['total_cost'],
            'cost_top_1pct': res_s['cost_top_1pct'],
            'cost_top_2pct': res_s['cost_top_2pct'],
            'cost_top_5pct': res_s['cost_top_5pct'],
            'fit_time_s': time.time() - t_seed
        })
        print(f"    Seed {seed}: ROC-AUC={res_s['roc_auc']:.4f}, PR-AUC={res_s['pr_auc']:.4f}, Cost=${res_s['total_cost']:,.0f}, tau*={tau_s:.4f}")

    del X_mat, X_tr_mat, X_va_mat, X_te_mat
    gc.collect()

df_seed_var = pd.DataFrame(seed_variance_results)
df_seed_summary = df_seed_var.groupby('model').agg(
    roc_auc_mean=('roc_auc', 'mean'), roc_auc_std=('roc_auc', 'std'),
    pr_auc_mean=('pr_auc', 'mean'), pr_auc_std=('pr_auc', 'std'),
    brier_mean=('brier', 'mean'), brier_std=('brier', 'std'),
    cost_mean=('total_cost', 'mean'), cost_std=('total_cost', 'std'),
    cost_top1_mean=('cost_top_1pct', 'mean'), cost_top1_std=('cost_top_1pct', 'std'),
    tau_mean=('tau', 'mean'), tau_std=('tau', 'std')
).reset_index()

print("\n--- Seed Variance Summary (Mean +/- SD across 5 seeds) ---")
print(df_seed_summary.to_string(index=False))

# 4. Save Master Prediction Artifacts & Checksums (Item 5)
print("\n" + "=" * 80)
print("SAVING MASTER PREDICTION ARRAYS AND SHA256 CHECKSUM")
print("=" * 80)

pred_file = "out/predictions/unified_393_test_predictions.npz"
np.savez_compressed(
    pred_file,
    y_test=Y[val_end:],
    amounts_test=amounts[val_end:],
    card1_test=card1_raw[val_end:],
    **test_predictions,
    **seed_predictions
)

sha256 = hashlib.sha256()
with open(pred_file, 'rb') as f:
    while chunk := f.read(8192 * 1024):
        sha256.update(chunk)
pred_hash = sha256.hexdigest()

checksum_file = "out/predictions/checksums.sha256"
with open(checksum_file, 'w') as f:
    f.write(f"{pred_hash}  {os.path.basename(pred_file)}\n")

print(f"Saved {len(test_predictions) + len(seed_predictions)} prediction arrays to {pred_file}")
print(f"SHA256: {pred_hash}")
print(f"Checksum logged in {checksum_file}")

# 5. Account-Clustered Bootstrap (B=1,000) on Master Predictions
print("\n" + "=" * 80)
print("ACCOUNT-CLUSTERED BOOTSTRAP (B=1,000 REPLICATES)")
print("=" * 80)

y_te = Y[val_end:]
amt_te = amounts[val_end:]
clusters_raw = card1_raw[val_end:]

unique_clusters = np.unique(clusters_raw)
num_clusters = len(unique_clusters)
print(f"Building cluster map for {num_clusters:,} unique accounts in test set...")

cluster_to_indices = {}
for i, c in enumerate(clusters_raw):
    cluster_to_indices.setdefault(c, []).append(i)
cluster_index_arrays = [np.array(cluster_to_indices[c], dtype=np.int32) for c in unique_clusters]

B = 1000
np.random.seed(42)

def sweep_cost(y_true, pred, amts, brackets):
    fn = float(np.sum(amts[(y_true == 1) & (pred == 0)]))
    fp = fp_cost_from_amounts(amts[(y_true == 0) & (pred == 1)], brackets)
    return fn + fp

BOOT_COMPARISONS = [
    ('D=0 (k=1)', 'Baseline'),
    ('D=7 (k=1)', 'Baseline'),
    ('D=14 (k=1)', 'Baseline'),
    ('D=30 (k=1)', 'Baseline'),
    ('D=60 (k=1)', 'Baseline'),
    ('No-Fraud-Hist (k=1)', 'Baseline'),
    ('D=30 (k=1)', 'D=0 (k=1)'),
    ('D=30 (k=3)', 'Baseline'),
    ('D=30 (k=5)', 'Baseline'),
]

boot_data = {comp: {'diff_auc': [], 'diff_prauc': [], 'diff_brier': [], 'diff_cost': []} for comp in BOOT_COMPARISONS}

t_boot = time.time()
for b in range(B):
    sampled_cluster_indices = np.random.choice(num_clusters, size=num_clusters, replace=True)
    idx_list = [cluster_index_arrays[c_idx] for c_idx in sampled_cluster_indices]
    boot_idx = np.concatenate(idx_list)
    
    y_b = y_te[boot_idx]
    amt_b = amt_te[boot_idx]
    
    for m1, m2 in BOOT_COMPARISONS:
        p1 = test_predictions[m1][boot_idx]
        p2 = test_predictions[m2][boot_idx]
        
        auc1 = roc_auc_score(y_b, p1)
        auc2 = roc_auc_score(y_b, p2)
        pr1 = average_precision_score(y_b, p1)
        pr2 = average_precision_score(y_b, p2)
        br1 = brier_score_loss(y_b, p1)
        br2 = brier_score_loss(y_b, p2)
        
        # Cost at respective optimal thresholds
        cost1 = sweep_cost(y_b, (p1 >= optimal_taus[m1]).astype(int), amt_b, BRACKETS)
        cost2 = sweep_cost(y_b, (p2 >= optimal_taus[m2]).astype(int), amt_b, BRACKETS)
        
        boot_data[(m1, m2)]['diff_auc'].append(auc1 - auc2)
        boot_data[(m1, m2)]['diff_prauc'].append(pr1 - pr2)
        boot_data[(m1, m2)]['diff_brier'].append(br1 - br2)
        boot_data[(m1, m2)]['diff_cost'].append(cost1 - cost2)

    if (b + 1) % 200 == 0:
        print(f"  Bootstrap progress: {b+1}/{B} iterations ({time.time()-t_boot:.1f}s)...")

bootstrap_summary_rows = []
for comp in BOOT_COMPARISONS:
    m1, m2 = comp
    d_dict = boot_data[comp]
    
    # Point estimates from main table
    pe_auc = grid_results[m1]['roc_auc'] - grid_results[m2]['roc_auc']
    pe_pr = grid_results[m1]['pr_auc'] - grid_results[m2]['pr_auc']
    pe_br = grid_results[m1]['brier'] - grid_results[m2]['brier']
    pe_cost = grid_results[m1]['total_cost'] - grid_results[m2]['total_cost']
    
    d_auc = np.array(d_dict['diff_auc'])
    d_pr = np.array(d_dict['diff_prauc'])
    d_br = np.array(d_dict['diff_brier'])
    d_cost = np.array(d_dict['diff_cost'])
    
    # Two-sided empirical p-values
    p_auc = 2.0 * min(np.mean(d_auc <= 0), np.mean(d_auc >= 0))
    p_pr = 2.0 * min(np.mean(d_pr <= 0), np.mean(d_pr >= 0))
    p_cost = 2.0 * min(np.mean(d_cost <= 0), np.mean(d_cost >= 0))
    
    bootstrap_summary_rows.append({
        'comparison': f"{m1} vs {m2}",
        'point_diff_auc': pe_auc,
        'boot_mean_diff_auc': float(np.mean(d_auc)),
        'auc_ci_lower': float(np.percentile(d_auc, 2.5)),
        'auc_ci_upper': float(np.percentile(d_auc, 97.5)),
        'p_val_auc': min(1.0, float(p_auc)),
        
        'point_diff_prauc': pe_pr,
        'boot_mean_diff_prauc': float(np.mean(d_pr)),
        'prauc_ci_lower': float(np.percentile(d_pr, 2.5)),
        'prauc_ci_upper': float(np.percentile(d_pr, 97.5)),
        'p_val_prauc': min(1.0, float(p_pr)),
        
        'point_diff_brier': pe_br,
        'boot_mean_diff_brier': float(np.mean(d_br)),
        'brier_ci_lower': float(np.percentile(d_br, 2.5)),
        'brier_ci_upper': float(np.percentile(d_br, 97.5)),
        
        'point_diff_cost': pe_cost,
        'boot_mean_diff_cost': float(np.mean(d_cost)),
        'cost_ci_lower': float(np.percentile(d_cost, 2.5)),
        'cost_ci_upper': float(np.percentile(d_cost, 97.5)),
        'p_val_cost': min(1.0, float(p_cost)),
    })

df_boot_summary = pd.DataFrame(bootstrap_summary_rows)

# 6. Save Tables & Reports
print("\n" + "=" * 80)
print("EXPORTING TABLES AND FINAL SUMMARY")
print("=" * 80)

df_grid = pd.DataFrame.from_dict(grid_results, orient='index').reset_index().rename(columns={'index': 'model'})
df_grid.to_csv("out/results/unified_393_grid_results.csv", index=False)
df_seed_var.to_csv("out/results/unified_393_seed_variance.csv", index=False)
df_seed_summary.to_csv("out/results/unified_393_seed_summary.csv", index=False)
df_boot_summary.to_csv("out/results/unified_393_clustered_bootstrap.csv", index=False)

print("Saved files:")
print(" - out/results/unified_393_grid_results.csv")
print(" - out/results/unified_393_seed_variance.csv")
print(" - out/results/unified_393_seed_summary.csv")
print(" - out/results/unified_393_clustered_bootstrap.csv")
print(f"Total pipeline completed in {time.time()-t_start:.1f}s")
