#!/usr/bin/env python3
"""
Secondary Analysis: Account Density Sweep on Standardized 393-Feature Pipeline
Tiers: 25%, 50%, 100% multi-transaction account linkability
Seeds: 42, 43, 44
Conditions: D=0 (Oracle) and D=30 (Realistic Verification Delay)

Saves:
- out/results/density_sweep_393_results.csv
- out/results/density_sweep_393_summary.csv
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
print("SECONDARY ANALYSIS: ACCOUNT DENSITY SWEEP (393 PIPELINE)")
print("=" * 80)

t_start = time.time()
TRANSACTION_PATH = os.environ.get("IEEE_CIS_TX_PATH", "data/train_transaction.csv")

# 1. Load raw transaction data
print("\n1. Loading raw transaction dataset...")
t0 = time.time()
tx = pd.read_csv(TRANSACTION_PATH)
assert len(tx) == 590540, f"Expected 590,540 rows, got {len(tx)}"

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
tx['raw_acc_id'] = tx['card1'].cat.codes.astype('int32')

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.10, 0.20
N = len(tx)
train_end = int(N * TRAIN_FRAC)
val_end = int(N * (TRAIN_FRAC + VAL_FRAC))

Y = tx['isFraud'].astype('int8').values
amounts = tx['amount'].values
tx_dts = tx['TransactionDT'].values.astype('float64')
raw_acc_ids = tx['raw_acc_id'].values

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
                 if c not in ('isFraud', 'TransactionDT', 'raw_acc_id')
                 and not c.startswith('hist_')]
assert len(BASE_FEATURES) == 393, f"Expected 393 features, got {len(BASE_FEATURES)}"

# Account structure
acc_counts = pd.Series(raw_acc_ids).value_counts()
multi_accounts = acc_counts[acc_counts > 1].index.values
single_accounts = acc_counts[acc_counts == 1].index.values

print(f"Total accounts: {len(acc_counts):,} (Multi: {len(multi_accounts):,}, Single: {len(single_accounts):,})")

DENSITIES = [0.25, 0.50, 1.00]
DELAYS = [0, 30]
SEEDS = [42, 43, 44]

XGB_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    scale_pos_weight=10, eval_metric='logloss',
    enable_categorical=True, tree_method='hist',
    subsample=0.8, colsample_bytree=0.8
)

results = []

for d_frac in DENSITIES:
    print(f"\n=======================================================")
    print(f"DENSITY TIER: {int(d_frac*100)}% Multi-Account Retention")
    print(f"=======================================================")
    
    # Subsample multi-accounts deterministically
    np.random.seed(42)
    n_keep = int(round(len(multi_accounts) * d_frac))
    if d_frac < 1.0:
        kept_multi = set(np.random.choice(multi_accounts, size=n_keep, replace=False))
    else:
        kept_multi = set(multi_accounts)
        
    # Map non-retained accounts to unique singleton IDs
    effective_ids = raw_acc_ids.copy()
    mask_singleton = ~pd.Series(raw_acc_ids).isin(kept_multi).values
    max_id = int(raw_acc_ids.max()) + 1
    effective_ids[mask_singleton] = max_id + np.arange(mask_singleton.sum(), dtype=np.int32)
    
    # Precompute fraud lookup on effective IDs
    acc_frauds = {}
    for ac, dt, is_fr in zip(effective_ids, tx_dts, Y):
        if is_fr == 1:
            acc_frauds.setdefault(ac, []).append(dt)
    acc_frauds = {k: np.array(v, dtype=np.float64) for k, v in acc_frauds.items()}
    
    tmp = pd.DataFrame({
        'acc': effective_ids,
        'amount': amounts,
        'TransactionDT': tx_dts,
        'isFraud': Y
    })
    g = tmp.groupby('acc', sort=False)
    tmp['_cnt'] = g.cumcount()
    tmp['_sum'] = g['amount'].cumsum().astype('float32')
    tmp['_max'] = g['amount'].cummax().astype('float32')
    tmp['_amt2'] = amounts ** 2
    tmp['_sumsq'] = tmp.groupby('acc', sort=False)['_amt2'].cumsum().astype('float64')
    
    first_dt = g['TransactionDT'].transform('first')
    account_age_days = ((tx_dts - first_dt) / 86400.0).astype('float32')
    
    def build_density_history(delay_days):
        g2 = tmp.groupby('acc', sort=False)
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
        
        for ac, dts_arr in acc_frauds.items():
            ac_mask = (effective_ids == ac)
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
        
    for delay in DELAYS:
        print(f"\n--- Evaluating Density={int(d_frac*100)}%, Delay D={delay}d ---")
        hf = build_density_history(delay)
        full_df = pd.concat([tx[BASE_FEATURES], hf], axis=1)
        feat_cols = list(full_df.columns)
        
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
            
            results.append({
                'density_pct': int(d_frac * 100),
                'delay_days': delay,
                'seed': s,
                'roc_auc': res['roc_auc'],
                'pr_auc': res['pr_auc'],
                'brier': res['brier'],
                'total_cost': res['total_cost'],
                'tau': tau,
            })
            print(f"  Density {int(d_frac*100)}% | D={delay:>2}d | Seed {s}: AUC={res['roc_auc']:.4f}, PR={res['pr_auc']:.4f}, Cost=${res['total_cost']:,.0f} ({time.time()-t_m:.1f}s)")

df_res = pd.DataFrame(results)
df_res.to_csv('out/results/density_sweep_393_results.csv', index=False)

summary = df_res.groupby(['density_pct', 'delay_days']).agg(
    auc_mean=('roc_auc', 'mean'), auc_sd=('roc_auc', 'std'),
    pr_mean=('pr_auc', 'mean'), pr_sd=('pr_auc', 'std'),
    cost_mean=('total_cost', 'mean'), cost_sd=('total_cost', 'std')
).reset_index()
summary.to_csv('out/results/density_sweep_393_summary.csv', index=False)

print("\n" + "=" * 80)
print("DENSITY SWEEP SUMMARY TABLE (393 PIPELINE)")
print("=" * 80)
print(summary.to_string(index=False))
print(f"\nCompleted in {time.time()-t_start:.1f}s.")
