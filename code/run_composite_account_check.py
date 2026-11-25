#!/usr/bin/env python3
"""
Composite account identifier robustness check
Compares card1 vs Composite Account ID (card1 + card2 + card3 + card5 + addr1)

1. Reports structural account density & burst metrics:
   - Unique account count
   - Multi-transaction account share
   - Total transactions covered by multi-tx accounts (%)
   - Multi-fraud accounts count
   - Total fraud transactions on multi-fraud accounts (%)
   - Intra-burst median gap, % < 1h, % < 24h, % < 7d

2. Trains and evaluates:
   - Baseline (393 features)
   - D=0 (k=1) using Composite Account History
   - D=30 (k=1) using Composite Account History
   Across 3 seeds (42, 43, 44) with subsample=0.8, colsample_bytree=0.8

3. Runs Account-Clustered Bootstrap (B=1,000) on composite accounts:
   - Delta ROC-AUC, Delta PR-AUC, Delta Cost ($)

Saves:
- out/predictions/composite_account_predictions.npz
- out/results/composite_account_metrics.json
- out/results/composite_account_results.csv
- out/results/composite_account_summary.csv
"""

import os
import sys, os, time, json, warnings
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings('ignore')

print("=" * 80)
print("Composite account identifier robustness check")
print("=" * 80)

t_start = time.time()
TRANSACTION_PATH = os.environ.get("IEEE_CIS_TX_PATH", "data/train_transaction.csv")

# 1. Load raw transaction data
print("\n1. Loading raw transaction dataset...")
t0 = time.time()
tx = pd.read_csv(TRANSACTION_PATH)
assert len(tx) == 590540, f"Expected 590,540 rows, got {len(tx)}"
print(f"Loaded {len(tx):,} rows in {time.time()-t0:.1f}s.")

tx = tx.rename(columns={'TransactionAmt': 'amount'})
tx = tx.drop(columns=['TransactionID'])

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

tx['hour_of_day'] = ((tx['TransactionDT'] // 3600) % 24).astype('int32')
tx['day_of_week_index'] = ((tx['TransactionDT'] // 86400) % 7).astype('int32')

for col in tx.columns:
    if col not in CATEGORICAL_COLS and tx[col].dtype == 'float64':
        tx[col] = tx[col].astype('float32')
    elif col not in CATEGORICAL_COLS and tx[col].dtype == 'int64':
        tx[col] = tx[col].astype('int32')

tx = tx.sort_values('TransactionDT', kind='stable').reset_index(drop=True)

# Define account IDs
tx['c1_account_id'] = tx['card1'].cat.codes.astype('int32')
comp_cols = ['card1', 'card2', 'card3', 'card5', 'addr1']
tx['composite_str'] = tx[comp_cols].astype(str).agg('_'.join, axis=1)
tx['comp_account_id'] = tx['composite_str'].astype('category').cat.codes.astype('int32')

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.10, 0.20
N = len(tx)
train_end = int(N * TRAIN_FRAC)
val_end = int(N * (TRAIN_FRAC + VAL_FRAC))

Y = tx['isFraud'].astype('int8').values
amounts = tx['amount'].values
tx_dts = tx['TransactionDT'].values.astype('float64')
comp_ids = tx['comp_account_id'].values
comp_str_te = tx['composite_str'].iloc[val_end:].values

# 2. Structural Statistics Comparison
print("\n2. Computing structural account density & burst statistics...")
def compute_account_stats(acc_col_name, label):
    acc_series = tx[acc_col_name]
    counts = acc_series.value_counts()
    n_unique = len(counts)
    multi_acc = (counts > 1).sum()
    multi_acc_pct = multi_acc / n_unique * 100.0
    tx_in_multi = tx[acc_col_name].isin(counts[counts > 1].index).sum()
    tx_in_multi_pct = tx_in_multi / N * 100.0
    
    # Fraud stats
    fraud_tx = tx[tx['isFraud'] == 1]
    f_counts = fraud_tx[acc_col_name].value_counts()
    n_f_unique = len(f_counts)
    multi_f_acc = (f_counts > 1).sum()
    multi_f_acc_pct = multi_f_acc / n_f_unique * 100.0
    total_f = len(fraud_tx)
    f_in_multi = fraud_tx[acc_col_name].isin(f_counts[f_counts > 1].index).sum()
    f_in_multi_pct = f_in_multi / total_f * 100.0
    
    # Intra-burst gaps
    fraud_df = fraud_tx.sort_values('TransactionDT')
    g = fraud_df.groupby(acc_col_name, sort=False)
    gaps = (fraud_df['TransactionDT'] - g['TransactionDT'].shift(1)).dropna().values
    
    return {
        'name': label,
        'unique_accounts': int(n_unique),
        'multi_tx_accounts': int(multi_acc),
        'multi_tx_account_pct': float(multi_acc_pct),
        'transactions_in_multi_accounts': int(tx_in_multi),
        'transactions_in_multi_accounts_pct': float(tx_in_multi_pct),
        'unique_fraud_accounts': int(n_f_unique),
        'multi_fraud_accounts': int(multi_f_acc),
        'multi_fraud_account_pct': float(multi_f_acc_pct),
        'frauds_in_multi_fraud_accounts': int(f_in_multi),
        'frauds_in_multi_fraud_accounts_pct': float(f_in_multi_pct),
        'gap_count': len(gaps),
        'gap_median_sec': float(np.median(gaps)),
        'gap_p25_sec': float(np.percentile(gaps, 25)),
        'gap_p75_sec': float(np.percentile(gaps, 75)),
        'gap_lt_1h_pct': float(np.mean(gaps < 3600.0) * 100.0),
        'gap_lt_24h_pct': float(np.mean(gaps < 86400.0) * 100.0),
        'gap_lt_7d_pct': float(np.mean(gaps < 604800.0) * 100.0),
    }

stats_c1 = compute_account_stats('card1', 'Single Card (card1)')
stats_comp = compute_account_stats('comp_account_id', 'Composite Account (card1+card2+card3+card5+addr1)')

print(f"  Single Card (card1): {stats_c1['unique_accounts']:,} unique, {stats_c1['multi_tx_accounts']:,} multi-tx ({stats_c1['multi_tx_account_pct']:.1f}%), {stats_c1['frauds_in_multi_fraud_accounts_pct']:.1f}% frauds in burst, median gap {stats_c1['gap_median_sec']/60.0:.1f}m")
print(f"  Composite Account:   {stats_comp['unique_accounts']:,} unique, {stats_comp['multi_tx_accounts']:,} multi-tx ({stats_comp['multi_tx_account_pct']:.1f}%), {stats_comp['frauds_in_multi_fraud_accounts_pct']:.1f}% frauds in burst, median gap {stats_comp['gap_median_sec']/60.0:.1f}m")

with open('out/results/composite_account_metrics.json', 'w') as f:
    json.dump({'card1': stats_c1, 'composite': stats_comp}, f, indent=2)

# 3. Build History Features on Composite Account ID
print("\n3. Building history features using composite account ID...")
BASE_UNIT_USD = 10.0
ESCALATION = (1, 2, 3, 4, 6, 10, 20)
QUANTILE_CUTS = (0.50, 0.75, 0.90, 0.97, 0.995, 0.999)
THRESH_GRID = np.linspace(0.01, 0.90, 100)

cutpoints = tx['amount'].iloc[:train_end].quantile(list(QUANTILE_CUTS)).values
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

def cost_opt(proba, y_true, amts):
    best_c = float('inf')
    best_tau = 0.5
    for tau in THRESH_GRID:
        pred = (proba >= tau).astype(int)
        fn = float(np.sum(amts[(y_true == 1) & (pred == 0)]))
        fp = fp_cost_from_amounts(amts[(y_true == 0) & (pred == 1)])
        tot = fn + fp
        if tot < best_c:
            best_c = tot
            best_tau = tau
    return best_tau, best_c

def evaluate_predictions(y_true, proba, amts, tau):
    pred = (proba >= tau).astype(int)
    fn = float(np.sum(amts[(y_true == 1) & (pred == 0)]))
    fp = fp_cost_from_amounts(amts[(y_true == 0) & (pred == 1)])
    return {
        'roc_auc': float(roc_auc_score(y_true, proba)),
        'pr_auc': float(average_precision_score(y_true, proba)),
        'brier': float(brier_score_loss(y_true, proba)),
        'tau': float(tau),
        'cost_fn': fn,
        'cost_fp': fp,
        'total_cost': fn + fp,
    }

BASE_FEATURES = [c for c in tx.columns
                 if c not in ('isFraud', 'TransactionDT', 'c1_account_id', 'composite_str', 'comp_account_id')
                 and not c.startswith('hist_')]
assert len(BASE_FEATURES) == 393, f"Expected 393 features, got {len(BASE_FEATURES)}"

tmp = pd.DataFrame({
    'comp_id': comp_ids,
    'amount': amounts,
    'TransactionDT': tx_dts,
    'isFraud': Y
})
g = tmp.groupby('comp_id', sort=False)
tmp['_cnt'] = g.cumcount()
tmp['_sum'] = g['amount'].cumsum().astype('float32')
tmp['_max'] = g['amount'].cummax().astype('float32')
tmp['_amt2'] = amounts ** 2
tmp['_sumsq'] = tmp.groupby('comp_id', sort=False)['_amt2'].cumsum().astype('float64')

first_dt = g['TransactionDT'].transform('first')
account_age_days = ((tx_dts - first_dt) / 86400.0).astype('float32')

comp_frauds = {}
for ac, dt, is_fr in zip(comp_ids, tx_dts, Y):
    if is_fr == 1:
        comp_frauds.setdefault(ac, []).append(dt)
comp_frauds = {ac: np.array(v, dtype=np.float64) for ac, v in comp_frauds.items()}

def build_comp_history_features(delay_days):
    g2 = tmp.groupby('comp_id', sort=False)
    hist_cnt = g2['_cnt'].shift(1).fillna(0.0)
    hist_sum = g2['_sum'].shift(1).fillna(0.0)
    hist_sumsq = g2['_sumsq'].shift(1).fillna(0.0)
    hist_max = g2['_max'].shift(1)
    dt_last = g['TransactionDT'].shift(1)
    amt_last = g['amount'].shift(1)
    
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
    
    delay_sec = delay_days * 86400.0
    confirmed_fraud_cnt = np.zeros(N, dtype=np.float32)
    dt_last_fraud = np.full(N, np.nan, dtype=np.float32)
    
    for ac, dts_arr in comp_frauds.items():
        ac_mask = (comp_ids == ac)
        ac_tx_dts = tx_dts[ac_mask]
        cutoff_dts = ac_tx_dts - delay_sec
        pos = np.searchsorted(dts_arr, cutoff_dts, side='left')
        confirmed_fraud_cnt[ac_mask] = pos
        has_fraud = (pos > 0)
        idx_last = np.maximum(pos - 1, 0)
        last_f_dt = np.where(has_fraud, dts_arr[idx_last], np.nan)
        dt_last_fraud[ac_mask] = ac_tx_dts - last_f_dt
        
    hf['hist_fraud_count'] = confirmed_fraud_cnt
    hf['hist_time_since_last_fraud'] = dt_last_fraud
    hf['hist_fraud_ratio'] = (hf['hist_fraud_count'] / hf['hist_tx_count'].clip(lower=1)).astype('float32')
    return hf

print("Building composite D=0 and D=30 history features...")
hf_comp_d0 = build_comp_history_features(0)
hf_comp_d30 = build_comp_history_features(30)
COMP_HISTORY_COLS = list(hf_comp_d0.columns)

# 4. Model Training & Evaluation
SEEDS = [42, 43, 44]
XGB_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    scale_pos_weight=10, eval_metric='logloss',
    enable_categorical=True, tree_method='hist',
    subsample=0.8, colsample_bytree=0.8
)

CONDITIONS = [
    ('Baseline', BASE_FEATURES, None),
    ('Composite_D=0', BASE_FEATURES + COMP_HISTORY_COLS, hf_comp_d0),
    ('Composite_D=30', BASE_FEATURES + COMP_HISTORY_COLS, hf_comp_d30),
]

saved_preds = {
    'y_test': Y[val_end:],
    'amounts_test': amounts[val_end:],
    'composite_test': comp_str_te,
}

results_rows = []

print(f"\nTraining {len(CONDITIONS)} conditions across {len(SEEDS)} seeds...")

for cond_name, feat_cols, hf_df in CONDITIONS:
    print(f"\n--- Condition: {cond_name} ({len(feat_cols)} features) ---")
    if hf_df is not None:
        full_df = pd.concat([tx[BASE_FEATURES], hf_df], axis=1)[feat_cols]
    else:
        full_df = tx[feat_cols]
        
    X_tr = full_df.iloc[:train_end]
    X_va = full_df.iloc[train_end:val_end]
    X_te = full_df.iloc[val_end:]
    
    for s in SEEDS:
        t_m = time.time()
        xgb = XGBClassifier(**XGB_PARAMS, random_state=s)
        xgb.fit(X_tr, Y[:train_end])
        
        p_va = xgb.predict_proba(X_va)[:, 1]
        p_te = xgb.predict_proba(X_te)[:, 1]
        
        tau, _ = cost_opt(p_va, Y[train_end:val_end], amounts[train_end:val_end])
        res = evaluate_predictions(Y[val_end:], p_te, amounts[val_end:], tau)
        
        saved_preds[f"{cond_name}_seed_{s}"] = p_te
        
        results_rows.append({
            'condition': cond_name,
            'seed': s,
            'features': len(feat_cols),
            'roc_auc': res['roc_auc'],
            'pr_auc': res['pr_auc'],
            'brier': res['brier'],
            'total_cost': res['total_cost'],
            'cost_fn': res['cost_fn'],
            'cost_fp': res['cost_fp'],
            'tau': tau,
        })
        print(f"  Seed {s}: AUC={res['roc_auc']:.4f}, PR={res['pr_auc']:.4f}, Cost=${res['total_cost']:,.0f}, tau*={tau:.4f} ({time.time()-t_m:.1f}s)")

df_all = pd.DataFrame(results_rows)
df_all.to_csv('out/results/composite_account_results.csv', index=False)
np.savez_compressed('out/predictions/composite_account_predictions.npz', **saved_preds)

# 5. Account-Clustered Bootstrap on Composite Accounts
print("\nRunning Composite Account-Clustered Bootstrap (B=1,000)...")
y_te = Y[val_end:]
a_te = amounts[val_end:]

unique_c = np.unique(comp_str_te)
c_to_idx = {c: [] for c in unique_c}
for i, c in enumerate(comp_str_te):
    c_to_idx[c].append(i)
c_idx_arr = [np.array(c_to_idx[c], dtype=np.int32) for c in unique_c]

B = 1000
np.random.seed(42)
boot_indices = []
for b in range(B):
    s_c = np.random.choice(len(unique_c), size=len(unique_c), replace=True)
    boot_indices.append(np.concatenate([c_idx_arr[i] for i in s_c]))

p_avg = {}
tau_avg = {}
for cond_name, _, _ in CONDITIONS:
    preds = [saved_preds[f"{cond_name}_seed_{s}"] for s in SEEDS]
    p_mean = np.mean(preds, axis=0)
    p_avg[cond_name] = p_mean
    tau_opt, _ = cost_opt(p_mean, y_te, a_te)
    tau_avg[cond_name] = tau_opt

boot_metrics = {}
for cond_name in p_avg:
    t_c = time.time()
    p = p_avg[cond_name]
    tau = tau_avg[cond_name]
    pred_bin = (p >= tau).astype(int)
    
    aucs = np.zeros(B)
    prs = np.zeros(B)
    costs = np.zeros(B)
    for b in range(B):
        b_idx = boot_indices[b]
        yb = y_te[b_idx]
        ab = a_te[b_idx]
        aucs[b] = roc_auc_score(yb, p[b_idx])
        prs[b] = average_precision_score(yb, p[b_idx])
        fn = float(np.sum(ab[(yb == 1) & (pred_bin[b_idx] == 0)]))
        fp = fp_cost_from_amounts(ab[(yb == 0) & (pred_bin[b_idx] == 1)])
        costs[b] = fn + fp
    boot_metrics[cond_name] = {'auc': aucs, 'pr': prs, 'cost': costs}
    print(f"  Computed bootstrap for {cond_name} in {time.time()-t_c:.1f}s")

COMPARISONS = [
    ('Composite_D=0', 'Baseline', 'Composite D=0 vs Baseline'),
    ('Composite_D=30', 'Baseline', 'Composite D=30 vs Baseline'),
    ('Composite_D=0', 'Composite_D=30', 'Composite D=0 vs Composite D=30'),
]

summary_rows = []
for c_cand, c_ref, desc in COMPARISONS:
    p_cand = p_avg[c_cand]
    p_ref = p_avg[c_ref]
    t_cand = tau_avg[c_cand]
    t_ref = tau_avg[c_ref]
    
    pt_auc_cand = roc_auc_score(y_te, p_cand)
    pt_auc_ref = roc_auc_score(y_te, p_ref)
    pe_auc = pt_auc_cand - pt_auc_ref
    
    pt_pr_cand = average_precision_score(y_te, p_cand)
    pt_pr_ref = average_precision_score(y_te, p_ref)
    pe_pr = pt_pr_cand - pt_pr_ref
    
    fn_c = float(np.sum(a_te[(y_te == 1) & (p_cand < t_cand)]))
    fp_c = fp_cost_from_amounts(a_te[(y_te == 0) & (p_cand >= t_cand)])
    fn_r = float(np.sum(a_te[(y_te == 1) & (p_ref < t_ref)]))
    fp_r = fp_cost_from_amounts(a_te[(y_te == 0) & (p_ref >= t_ref)])
    pe_cost = (fn_c + fp_c) - (fn_r + fp_r)
    
    d_auc = boot_metrics[c_cand]['auc'] - boot_metrics[c_ref]['auc']
    d_pr = boot_metrics[c_cand]['pr'] - boot_metrics[c_ref]['pr']
    d_cost = boot_metrics[c_cand]['cost'] - boot_metrics[c_ref]['cost']
    
    p_val_auc = 2.0 * min(np.mean(d_auc <= 0), np.mean(d_auc >= 0))
    p_val_cost = 2.0 * min(np.mean(d_cost <= 0), np.mean(d_cost >= 0))
    
    summary_rows.append({
        'comparison': f"{c_cand} vs {c_ref}",
        'description': desc,
        'd_auc_pt': pe_auc,
        'd_auc_ci_low': np.percentile(d_auc, 2.5),
        'd_auc_ci_high': np.percentile(d_auc, 97.5),
        'p_val_auc': min(1.0, float(p_val_auc)),
        'd_pr_pt': pe_pr,
        'd_pr_ci_low': np.percentile(d_pr, 2.5),
        'd_pr_ci_high': np.percentile(d_pr, 97.5),
        'd_cost_pt': pe_cost,
        'd_cost_ci_low': np.percentile(d_cost, 2.5),
        'd_cost_ci_high': np.percentile(d_cost, 97.5),
        'p_val_cost': min(1.0, float(p_val_cost)),
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv('out/results/composite_account_summary.csv', index=False)

print("\n" + "=" * 80)
print("COMPOSITE ACCOUNT SUMMARY TABLE")
print("=" * 80)
for _, r in df_summary.iterrows():
    print(f"\n{r['comparison']} ({r['description']}):")
    print(f"  Delta ROC-AUC: {r['d_auc_pt']:+.4f} [{r['d_auc_ci_low']:+.4f}, {r['d_auc_ci_high']:+.4f}], p={r['p_val_auc']:.3f}")
    print(f"  Delta PR-AUC:  {r['d_pr_pt']:+.4f} [{r['d_pr_ci_low']:+.4f}, {r['d_pr_ci_high']:+.4f}]")
    print(f"  Delta Cost:   ${r['d_cost_pt']:+,.0f} [${r['d_cost_ci_low']:+,.0f}, ${r['d_cost_ci_high']:+,.0f}], p={r['p_val_cost']:.3f}")

print("\n" + "=" * 80)
print(f"Finished in {time.time()-t_start:.1f}s.")
print("=" * 80)
