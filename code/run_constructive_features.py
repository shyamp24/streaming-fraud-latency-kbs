#!/usr/bin/env python3
"""
Step 4: Constructive Label-Free Feature Evaluation
Pre-specified label-free features:
1. cf_amt_zscore: (amount - running_mean) / running_std (strictly past transactions, k=1)
2. cf_velocity_surge: cnt_1h / (cnt_24h / 24.0 + 1.0) (strictly past transactions)
3. cf_inter_arrival: TransactionDT_i - TransactionDT_{i-1} (sentinel -1.0 for first tx)

Conditions evaluated:
1. Baseline (393 features)
2. Baseline + Constructive (396 features)
3. D=0 (k=1) (405 features)
4. D=0 (k=1) + Constructive (408 features)
5. D=30 (k=1) (405 features)
6. D=30 (k=1) + Constructive (408 features)

Protocol:
- Seeds 42, 43, 44 with subsample=0.8, colsample_bytree=0.8
- 7-tier quantile cost schedule (side='right')
- Paired differences across seeds
- Account-clustered bootstrap 95% CIs (B=1,000)
- Saves:
  * out/predictions/constructive_features_predictions.npz
  * out/results/constructive_features_results.csv
  * out/results/constructive_features_summary.csv
"""

import os
import sys, os, time, warnings
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

warnings.filterwarnings('ignore')

print("=" * 80)
print("STEP 4: CONSTRUCTIVE LABEL-FREE FEATURE EVALUATION")
print("=" * 80)

t_start = time.time()
TRANSACTION_PATH = os.environ.get("IEEE_CIS_TX_PATH", "data/train_transaction.csv")

# 1. Load and prepare data identically to unified 393 pipeline
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
tx['account_id'] = tx['card1'].cat.codes.astype('int32')

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.10, 0.20
N = len(tx)
train_end = int(N * TRAIN_FRAC)
val_end = int(N * (TRAIN_FRAC + VAL_FRAC))

Y = tx['isFraud'].astype('int8').values
amounts = tx['amount'].values
tx_dts = tx['TransactionDT'].values.astype('float64')
acc_ids = tx['account_id'].values
card1_raw = tx['card1'].astype(str).values

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
                 if c not in ('isFraud', 'TransactionDT', 'account_id')
                 and not c.startswith('hist_')
                 and not c.startswith('cf_')]
assert len(BASE_FEATURES) == 393, f"Expected 393 features, got {len(BASE_FEATURES)}"

# 2. Engineer Constructive Label-Free Features
print("\n2. Engineering constructive label-free features...")
t_feat = time.time()

# A. Running mean and std of amount (k=1 strictly past)
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

g2 = tmp.groupby('account_id', sort=False)
hist_cnt = g2['_cnt'].shift(1).fillna(0.0)
hist_sum = g2['_sum'].shift(1).fillna(0.0)
hist_sumsq = g2['_sumsq'].shift(1).fillna(0.0)
mean_amt = (hist_sum / hist_cnt.clip(lower=1)).fillna(0.0)
n_amt = hist_cnt.clip(lower=0)
var_amt = (hist_sumsq - n_amt * mean_amt.astype('float64') ** 2) / (n_amt - 1).clip(lower=1)
std_amt = np.where(n_amt > 1, np.sqrt(var_amt.clip(lower=0)), 0.0)

# Amount z-score (clipped to [-10, 10] to prevent extreme numerical outliers when std is very small)
tx['cf_amt_zscore'] = np.where((n_amt > 1) & (std_amt > 0.01), ((amounts - mean_amt) / std_amt), 0.0).clip(-10.0, 10.0).astype('float32')

# B. Inter-arrival time (time since immediately preceding tx)
dt_prev = g['TransactionDT'].shift(1)
tx['cf_inter_arrival'] = (tx_dts - dt_prev).fillna(-1.0).astype('float32')

# C. Velocity surge: cnt_1h / (cnt_24h / 24.0 + 1.0)
# Group transaction timestamps by account
acc_dts_dict = {}
for i in range(N):
    a = acc_ids[i]
    if a not in acc_dts_dict:
        acc_dts_dict[a] = []
    acc_dts_dict[a].append(tx_dts[i])
acc_dts_dict = {a: np.array(v, dtype=np.float64) for a, v in acc_dts_dict.items()}

cnt_1h = np.zeros(N, dtype=np.float32)
cnt_24h = np.zeros(N, dtype=np.float32)
acc_ptrs = {a: 0 for a in acc_dts_dict}

for i in range(N):
    a = acc_ids[i]
    t = tx_dts[i]
    k_idx = acc_ptrs[a]
    acc_ptrs[a] += 1
    if k_idx == 0:
        continue
    history = acc_dts_dict[a][:k_idx]
    cnt_1h[i] = k_idx - np.searchsorted(history, t - 3600.0, side='left')
    cnt_24h[i] = k_idx - np.searchsorted(history, t - 86400.0, side='left')

tx['cf_velocity_surge'] = (cnt_1h / (cnt_24h / 24.0 + 1.0)).astype('float32')

CONSTRUCTIVE_FEATURES = ['cf_amt_zscore', 'cf_inter_arrival', 'cf_velocity_surge']
print(f"Engineered {len(CONSTRUCTIVE_FEATURES)} constructive features in {time.time()-t_feat:.1f}s.")
for f in CONSTRUCTIVE_FEATURES:
    print(f"  {f}: min={tx[f].min():.2f}, mean={tx[f].mean():.2f}, max={tx[f].max():.2f}, NaNs={tx[f].isna().sum()}")

# 3. Engineer standard 12 history features for D=0 and D=30 (k=1)
first_dt = g['TransactionDT'].transform('first')
account_age_days = ((tx_dts - first_dt) / 86400.0).astype('float32')

account_frauds = {}
for ac, dt, is_fr in zip(acc_ids, tx_dts, Y):
    if is_fr == 1:
        account_frauds.setdefault(ac, []).append(dt)
account_frauds = {ac: np.array(v, dtype=np.float64) for ac, v in account_frauds.items()}

def build_history_features(delay_days):
    hist_max = g2['_max'].shift(1)
    amt_last = g['amount'].shift(1)
    
    hf = pd.DataFrame(index=tx.index)
    hf['hist_tx_count'] = hist_cnt.astype('float32')
    hf['hist_amt_sum'] = hist_sum.astype('float32')
    hf['hist_amt_mean'] = mean_amt.astype('float32')
    hf['hist_amt_std'] = pd.Series(np.where(n_amt > 1, std_amt, np.nan), index=tx.index).astype('float32')
    hf['hist_amt_max'] = hist_max.astype('float32')
    hf['hist_amt_last'] = amt_last.astype('float32')
    hf['hist_time_since_last'] = (tx_dts - dt_prev).astype('float32')
    hf['hist_account_age_days'] = account_age_days
    hf['hist_tx_per_day'] = (hf['hist_tx_count'] / (account_age_days + 1e-6)).astype('float32')
    
    delay_sec = delay_days * 86400.0
    confirmed_fraud_cnt = np.zeros(N, dtype=np.float32)
    dt_last_fraud = np.full(N, np.nan, dtype=np.float32)
    
    for ac, dts_arr in account_frauds.items():
        ac_mask = (acc_ids == ac)
        ac_tx_dts = tx_dts[ac_mask]
        cutoff_dts = ac_tx_dts - delay_sec
        # Strictly before cutoff_dt (exclusive)
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

print("\nGenerating history feature sets...")
hf_d0 = build_history_features(0)
hf_d30 = build_history_features(30)

HISTORY_COLS = list(hf_d0.columns)
assert len(HISTORY_COLS) == 12, f"Expected 12 history columns, got {len(HISTORY_COLS)}"

# 4. Set up conditions to train and evaluate
COND_SPECS = [
    ('Baseline', BASE_FEATURES, None),
    ('Baseline + Constructive', BASE_FEATURES + CONSTRUCTIVE_FEATURES, None),
    ('D=0', BASE_FEATURES + HISTORY_COLS, hf_d0),
    ('D=0 + Constructive', BASE_FEATURES + HISTORY_COLS + CONSTRUCTIVE_FEATURES, hf_d0),
    ('D=30', BASE_FEATURES + HISTORY_COLS, hf_d30),
    ('D=30 + Constructive', BASE_FEATURES + HISTORY_COLS + CONSTRUCTIVE_FEATURES, hf_d30),
]

SEEDS = [42, 43, 44]
XGB_BASE_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    scale_pos_weight=10, eval_metric='logloss',
    enable_categorical=True, tree_method='hist',
    subsample=0.8, colsample_bytree=0.8
)

saved_predictions = {
    'y_test': Y[val_end:],
    'amounts_test': amounts[val_end:],
    'card1_test': card1_raw[val_end:],
}

results_rows = []

print(f"\nEvaluating {len(COND_SPECS)} conditions across {len(SEEDS)} seeds (Total {len(COND_SPECS)*len(SEEDS)} models)...")

for cond_name, feat_cols, hf_df in COND_SPECS:
    print(f"\n--- Condition: {cond_name} ({len(feat_cols)} features) ---")
    
    # Prepare full feature dataframe
    if hf_df is not None:
        full_df = pd.concat([tx[BASE_FEATURES + CONSTRUCTIVE_FEATURES], hf_df], axis=1)[feat_cols]
    else:
        full_df = tx[feat_cols]
        
    X_tr = full_df.iloc[:train_end]
    X_va = full_df.iloc[train_end:val_end]
    X_te = full_df.iloc[val_end:]
    
    seed_metrics = []
    
    for s in SEEDS:
        t_m = time.time()
        xgb = XGBClassifier(**XGB_BASE_PARAMS, random_state=s)
        xgb.fit(X_tr, Y[:train_end])
        
        p_va = xgb.predict_proba(X_va)[:, 1]
        p_te = xgb.predict_proba(X_te)[:, 1]
        
        tau, _ = cost_opt(p_va, Y[train_end:val_end], amounts[train_end:val_end])
        res = evaluate_predictions(Y[val_end:], p_te, amounts[val_end:], tau)
        
        pred_key = f"{cond_name}_seed_{s}"
        saved_predictions[pred_key] = p_te
        
        seed_metrics.append({
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
        
    results_rows.extend(seed_metrics)

df_all = pd.DataFrame(results_rows)
df_all.to_csv('out/results/constructive_features_results.csv', index=False)
np.savez_compressed('out/predictions/constructive_features_predictions.npz', **saved_predictions)
print("\nSaved detailed results and prediction arrays.")

# 5. Clustered Bootstrap on Seed-Averaged Predictions
print("\nRunning Account-Clustered Bootstrap (B=1,000) for comparisons...")
y_te = Y[val_end:]
a_te = amounts[val_end:]
c1_te = card1_raw[val_end:]

unique_c = np.unique(c1_te)
c_to_idx = {c: [] for c in unique_c}
for i, c in enumerate(c1_te):
    c_to_idx[c].append(i)
c_idx_arr = [np.array(c_to_idx[c], dtype=np.int32) for c in unique_c]

B = 1000
np.random.seed(42)
boot_indices = []
for b in range(B):
    s_c = np.random.choice(len(unique_c), size=len(unique_c), replace=True)
    boot_indices.append(np.concatenate([c_idx_arr[i] for i in s_c]))

# Compute seed-averaged probabilities
p_avg = {}
tau_avg = {}
for cond_name, _, _ in COND_SPECS:
    preds = [saved_predictions[f"{cond_name}_seed_{s}"] for s in SEEDS]
    p_mean = np.mean(preds, axis=0)
    p_avg[cond_name] = p_mean
    tau_opt, _ = cost_opt(p_mean, y_te, a_te)
    tau_avg[cond_name] = tau_opt

# Precompute bootstrap metrics for each condition
boot_metrics = {}
print("Precomputing bootstrap distributions for each condition...")
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
    print(f"  Computed for {cond_name} in {time.time()-t_c:.1f}s")

# Comparisons:
# 1. Baseline + Constructive vs Baseline
# 2. D=30 + Constructive vs D=30
# 3. D=0 + Constructive vs D=0
# 4. D=30 vs Baseline
# 5. D=0 vs Baseline
COMPARISONS = [
    ('Baseline + Constructive', 'Baseline', 'Gain from Constructive Features on Raw Baseline'),
    ('D=30 + Constructive', 'D=30', 'Gain from Constructive Features under Delayed Feedback (D=30)'),
    ('D=0 + Constructive', 'D=0', 'Gain from Constructive Features under Immediate Labels (D=0)'),
    ('D=30', 'Baseline', 'D=30 History vs Baseline'),
    ('D=0', 'Baseline', 'D=0 History vs Baseline'),
]

summary_rows = []
for c_cand, c_ref, desc in COMPARISONS:
    # Full test point estimates
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
    
    # Bootstrap differences
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
df_summary.to_csv('out/results/constructive_features_summary.csv', index=False)

print("\n" + "=" * 80)
print("CONSTRUCTIVE FEATURES SUMMARY TABLE")
print("=" * 80)
for _, r in df_summary.iterrows():
    print(f"\n{r['comparison']} ({r['description']}):")
    print(f"  Delta ROC-AUC: {r['d_auc_pt']:+.4f} [{r['d_auc_ci_low']:+.4f}, {r['d_auc_ci_high']:+.4f}], p={r['p_val_auc']:.3f}")
    print(f"  Delta PR-AUC:  {r['d_pr_pt']:+.4f} [{r['d_pr_ci_low']:+.4f}, {r['d_pr_ci_high']:+.4f}]")
    print(f"  Delta Cost:   ${r['d_cost_pt']:+,.0f} [${r['d_cost_ci_low']:+,.0f}, ${r['d_cost_ci_high']:+,.0f}], p={r['p_val_cost']:.3f}")

print("\n" + "=" * 80)
print(f"STEP 4 FULLY COMPLETED IN {time.time()-t_start:.1f}s.")
print("=" * 80)
