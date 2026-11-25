#!/usr/bin/env python3
"""
Fine-Grained Sub-Day Verification Latency Sweep (The Feedback Half-Life Curve)
Evaluates verification delays at k=1 across 3 seeds (42, 43, 44):
  D in {0s (instant), 300s (5m), 1800s (30m), 3600s (1h), 21600s (6h), 86400s (24h), 259200s (72h), 604800s (7d), 2592000s (30d)}
plus Baseline and No-Fraud-Hist.

Reuses existing master predictions for Baseline, D=0, D=30, No-Fraud-Hist (seeds 42-44),
and trains the new intermediate sub-day latency models.
Saves:
  out/results/fine_grained_latency_results.csv
  out/results/fine_grained_latency_summary.csv
"""

import os, sys, time, gc
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

print("=" * 80)
print("FINE-GRAINED SUB-DAY VERIFICATION LATENCY SWEEP")
print("=" * 80)

t_start = time.time()

TRANSACTION_PATH = os.environ.get("IEEE_CIS_TX_PATH", "data/train_transaction.csv")

XGB_PARAMS = dict(n_estimators=500, max_depth=5, learning_rate=0.05,
                  scale_pos_weight=10, eval_metric='logloss',
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

# Canonical searchsorted implementation matching notebook Cell 1 exactly
def fp_cost_from_amounts(amts, brackets):
    if len(amts) == 0:
        return 0.0
    limits = np.array([lim for lim, _ in brackets])
    costs = np.array([c for _, c in brackets])
    idx = np.searchsorted(limits, amts, side='right')
    return float(costs[np.minimum(idx, len(costs) - 1)].sum())

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

print("Loading raw transaction dataset...")
t0 = time.time()
tx = pd.read_csv(TRANSACTION_PATH)
tx = tx.rename(columns={'TransactionAmt': 'amount'}).drop(columns=['TransactionID'])

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

N = len(tx)
train_end = int(N * TRAIN_FRAC)
val_end = int(N * (TRAIN_FRAC + VAL_FRAC))

Y = tx['isFraud'].astype('int8').values
amounts = tx['amount'].values
tx_dts = tx['TransactionDT'].values.astype('float64')
acc_ids = tx['account_id'].values

BRACKETS = build_quantile_brackets(tx['amount'].iloc[:train_end])
BASE_FEATURES = [c for c in tx.columns
                 if c not in ('isFraud', 'TransactionDT', 'account_id')
                 and not c.startswith('hist_')]
assert len(BASE_FEATURES) == 393, f"Expected 393, got {len(BASE_FEATURES)}"

account_frauds = {}
for ac, dt, is_fr in zip(acc_ids, tx_dts, Y):
    if is_fr == 1:
        account_frauds.setdefault(ac, []).append(dt)
account_frauds = {ac: np.array(v, dtype=np.float64) for ac, v in account_frauds.items()}

acc_indices = {}
for idx, ac in enumerate(acc_ids):
    acc_indices.setdefault(ac, []).append(idx)

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

def build_features_for_delay(delay_sec, k=1):
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

    if delay_sec is None:
        hf['hist_fraud_count'] = 0.0
        hf['hist_fraud_rate'] = 0.0
        hf['hist_fraud_time_since'] = np.nan
    elif delay_sec == 0:
        tmp['_fr'] = g['isFraud'].cumsum().astype('float32')
        tmp['_fr_dt'] = np.where(Y == 1, tx_dts, np.nan)
        g2_fr = tmp.groupby('account_id', sort=False)
        hist_fr = g2_fr['_fr'].shift(k).fillna(0.0)
        last_fr = g2_fr['_fr_dt'].shift(k).groupby(acc_ids).ffill()
        hf['hist_fraud_count'] = hist_fr.astype('float32')
        hf['hist_fraud_rate'] = (hf['hist_fraud_count'] / hf['hist_tx_count'].clip(lower=1)).astype('float32')
        hf['hist_fraud_time_since'] = (tx_dts - last_fr).astype('float32')
    else:
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
                cutoff_dt = eval_dt - float(delay_sec)
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

    return pd.concat([tx[BASE_FEATURES], hf], axis=1)

# Load existing master predictions for existing conditions
master_preds = np.load('out/predictions/unified_393_test_predictions.npz', allow_pickle=True)
y_te = Y[val_end:]
amts_te = amounts[val_end:]

SEEDS = [42, 43, 44]
results = []

# Populate already evaluated conditions from master arrays
EXISTING_CONDITIONS = [
    ('Baseline', 'Baseline_seed_{seed}', None, 'Baseline'),
    ('D=0s (Instant)', 'D=0 (k=1)_seed_{seed}', 0, '0s'),
    ('D=30d (2592000s)', 'D=30 (k=1)_seed_{seed}', 2592000, '30d'),
    ('No-Fraud-Hist', 'No-Fraud-Hist (k=1)_seed_{seed}', None, 'No-Fraud-Hist'),
]

print("Evaluating existing master seed predictions (Baseline, D=0, D=30, No-Fraud-Hist)...")
for cond_name, key_template, delay_s, label in EXISTING_CONDITIONS:
    for s in SEEDS:
        p_te = master_preds[key_template.format(seed=s)]
        tau, _ = cost_opt(p_te, y_te, amts_te, BRACKETS)
        res = evaluate_predictions(y_te, p_te, amts_te, tau, BRACKETS)
        results.append({
            'condition': cond_name,
            'delay_sec': delay_s,
            'label': label,
            'seed': s,
            **res
        })

# New sub-day conditions to train
NEW_CONDITIONS = [
    ('D=5m (300s)', 300, '5m'),
    ('D=30m (1800s)', 1800, '30m'),
    ('D=1h (3600s)', 3600, '1h'),
    ('D=6h (21600s)', 21600, '6h'),
    ('D=24h (1d)', 86400, '24h'),
    ('D=72h (3d)', 259200, '72h'),
    ('D=7d (604800s)', 604800, '7d'),
]

new_predictions = {}

for cond_name, delay_s, label in NEW_CONDITIONS:
    print(f"\n--- Training New Condition: {cond_name} ---")
    t_c = time.time()
    X_mat = build_features_for_delay(delay_s, k=1)
    
    X_tr = X_mat.iloc[:train_end]
    X_va = X_mat.iloc[train_end:val_end]
    X_te = X_mat.iloc[val_end:]
    
    for seed in SEEDS:
        t_s = time.time()
        params = XGB_PARAMS.copy()
        params['random_state'] = seed
        params['subsample'] = 0.8
        params['colsample_bytree'] = 0.8
        
        xgb = XGBClassifier(**params)
        xgb.fit(X_tr, Y[:train_end])
        
        p_va = xgb.predict_proba(X_va)[:, 1]
        p_te = xgb.predict_proba(X_te)[:, 1]
        
        new_predictions[f"{cond_name}_seed_{seed}"] = p_te
        
        tau, _ = cost_opt(p_va, Y[train_end:val_end], amounts[train_end:val_end], BRACKETS)
        res = evaluate_predictions(Y[val_end:], p_te, amounts[val_end:], tau, BRACKETS)
        
        results.append({
            'condition': cond_name,
            'delay_sec': delay_s,
            'label': label,
            'seed': seed,
            **res
        })
        print(f"  Seed {seed}: ROC-AUC={res['roc_auc']:.4f}, PR-AUC={res['pr_auc']:.4f}, "
              f"Brier={res['brier']:.4f}, Cost=${res['total_cost']:,.0f}, tau*={tau:.4f} ({time.time()-t_s:.1f}s)")
        
    del X_mat, X_tr, X_va, X_te
    gc.collect()

# Save predictions and results
df_res = pd.DataFrame(results)
os.makedirs("out/results", exist_ok=True)
os.makedirs("out/predictions", exist_ok=True)

np.savez_compressed("out/predictions/fine_grained_predictions.npz", **new_predictions)
df_res.to_csv("out/results/fine_grained_latency_results.csv", index=False)

df_summary = df_res.groupby(['condition', 'label', 'delay_sec'], dropna=False).agg(
    roc_auc_mean=('roc_auc', 'mean'), roc_auc_std=('roc_auc', 'std'),
    pr_auc_mean=('pr_auc', 'mean'), pr_auc_std=('pr_auc', 'std'),
    brier_mean=('brier', 'mean'), brier_std=('brier', 'std'),
    cost_mean=('total_cost', 'mean'), cost_std=('total_cost', 'std'),
    cost_top1_mean=('cost_top_1pct', 'mean'), cost_top1_std=('cost_top_1pct', 'std'),
    cost_top2_mean=('cost_top_2pct', 'mean'), cost_top2_std=('cost_top_2pct', 'std'),
    tau_mean=('tau', 'mean'), tau_std=('tau', 'std')
).reset_index()

# Sort logically by delay_sec
df_summary['sort_key'] = df_summary['delay_sec'].fillna(99999999)
df_summary = df_summary.sort_values('sort_key').drop(columns=['sort_key'])

df_summary.to_csv("out/results/fine_grained_latency_summary.csv", index=False)
print("\n" + "=" * 80)
print("FINE-GRAINED LATENCY SWEEP SUMMARY (Mean across 3 seeds)")
print("=" * 80)
print(df_summary.to_string(index=False))
print(f"\nCompleted in {time.time()-t_start:.1f}s")
