import numpy as np
import pandas as pd
import sys

def compute_history_features_unified(df, k=1, delay_days=None):
    data = df.copy()
    acc_ids = data['account_id'].values
    amounts = data['amount'].values
    tx_dts = data['TransactionDT'].values.astype('float64')
    Y = data['isFraud'].astype('int8').values

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
    hist_cnt = g2['_cnt'].shift(k).fillna(0.0)
    hist_sum = g2['_sum'].shift(k).fillna(0.0)
    hist_sumsq = g2['_sumsq'].shift(k).fillna(0.0)
    hist_max = g2['_max'].shift(k)
    dt_last = g['TransactionDT'].shift(k)
    amt_last = g['amount'].shift(k)

    mean = (hist_sum / hist_cnt.clip(lower=1)).fillna(0.0)
    n = hist_cnt.clip(lower=0)
    var = (hist_sumsq - n * mean.astype('float64') ** 2) / (n - 1).clip(lower=1)

    hf = pd.DataFrame(index=data.index)
    hf['hist_tx_count'] = hist_cnt.astype('float32')
    hf['hist_amt_sum'] = hist_sum.astype('float32')
    hf['hist_amt_mean'] = mean.astype('float32')
    hf['hist_amt_std'] = pd.Series(np.where(n > 1, np.sqrt(var.clip(lower=0)), np.nan), index=data.index).astype('float32')
    hf['hist_amt_max'] = hist_max.astype('float32')
    hf['hist_amt_last'] = amt_last.astype('float32')
    hf['hist_time_since_last'] = (tx_dts - dt_last).astype('float32')
    
    first_dt = g['TransactionDT'].transform('first')
    account_age_days = ((tx_dts - first_dt) / 86400.0).astype('float32')
    hf['hist_account_age_days'] = account_age_days
    hf['hist_tx_per_day'] = (hf['hist_tx_count'] / (account_age_days + 1e-6)).astype('float32')

    if delay_days is None:
        hf['hist_fraud_count'] = 0.0
        hf['hist_fraud_rate'] = 0.0
        hf['hist_fraud_time_since'] = np.nan
    elif delay_days == 0:
        tmp['_fr'] = g['isFraud'].cumsum().astype('float32')
        tmp['_fr_dt'] = np.where(Y == 1, tx_dts, np.nan)
        g2 = tmp.groupby('account_id', sort=False)
        hist_fr = g2['_fr'].shift(k).fillna(0.0)
        last_fr = g2['_fr_dt'].shift(k).groupby(acc_ids).ffill()
        hf['hist_fraud_count'] = hist_fr.astype('float32')
        hf['hist_fraud_rate'] = (hf['hist_fraud_count'] / hf['hist_tx_count'].clip(lower=1)).astype('float32')
        hf['hist_fraud_time_since'] = (tx_dts - last_fr).astype('float32')
    else:
        delay_sec = float(delay_days) * 86400.0
        account_frauds = {}
        for acc, dt, is_fr in zip(acc_ids, tx_dts, Y):
            if is_fr == 1:
                account_frauds.setdefault(acc, []).append(dt)
        account_frauds = {ac: np.array(v, dtype=np.float64) for ac, v in account_frauds.items()}

        acc_indices = {}
        for idx, ac in enumerate(acc_ids):
            acc_indices.setdefault(ac, []).append(idx)
        
        fr_cnt_arr = np.zeros(len(data), dtype=np.float32)
        fr_time_arr = np.full(len(data), np.nan, dtype=np.float32)

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

    return hf

def main():
    print("=" * 80)
    print("SYNTHETIC ACCOUNT BOUNDARY & REAL FEATURE BUILDER AUDIT")
    print("=" * 80)
    
    # 1. Exact Boundary Test (D=30, k=1)
    # Account 1:
    # tx 0: t = 10,000, fraud = 1
    # tx 1: t = 10,000 + 30*86400 - 1s (boundary - 1s). Expect hist_fraud_count == 0
    # tx 2: t = 10,000 + 30*86400 (exact boundary). Expect hist_fraud_count == 1
    # tx 3: t = 10,000 + 30*86400 + 500s. Expect hist_fraud_count == 1
    t0 = 10000.0
    day = 86400.0
    df1 = pd.DataFrame({
        'account_id': [1, 1, 1, 1],
        'amount': [100.0, 50.0, 75.0, 120.0],
        'TransactionDT': [t0, t0 + 30*day - 1.0, t0 + 30*day, t0 + 30*day + 500.0],
        'isFraud': [1, 0, 0, 0]
    })
    res1 = compute_history_features_unified(df1, k=1, delay_days=30)
    print("\nTest 1: Exact Boundary at t_fraud + D and t_fraud + D - 1s (D=30, k=1)")
    print(res1[['hist_tx_count', 'hist_fraud_count', 'hist_fraud_rate', 'hist_fraud_time_since']])
    assert res1['hist_fraud_count'].iloc[1] == 0.0, "Test 1 Failed: t + D - 1s leaked fraud!"
    assert res1['hist_fraud_count'].iloc[2] == 1.0, "Test 1 Failed: t + D did not register fraud!"
    assert res1['hist_fraud_count'].iloc[3] == 1.0, "Test 1 Failed: t + D + 500s did not register fraud!"
    assert res1['hist_fraud_time_since'].iloc[2] == 30 * day, "Test 1 Failed: time since fraud mismatch!"
    print("--> Test 1 PASSED: Strict boundary enforcement verified directly in feature builder.")

    # 2. Combined Lag & Delay Interaction Tests (k=2, D=30)
    # Account 2:
    # tx 0: t = 0d, fraud = 1
    # tx 1: t = 10d, non-fraud
    # tx 2: t = 35d, non-fraud. For tx 2:
    #   Lag k=2 looks back to tx 0 (t=0d).
    #   Delay D=30d requires t >= t_fraud + 30d (0d + 30d = 30d). Since 35d >= 30d, tx 0 is visible.
    #   Expected: hist_fraud_count == 1
    # tx 3: t = 45d, fraud = 1
    # tx 4: t = 50d, non-fraud. For tx 4:
    #   Lag k=2 looks back to tx 2 (t=35d).
    #   Fraud at tx 3 (t=45d) is excluded by lag (45d > 35d).
    #   Fraud at tx 0 (t=0d) is visible (0d <= 35d and 50d >= 0d + 30d).
    #   Expected: hist_fraud_count == 1 (tx 3 fraud masked by lag!)
    df2 = pd.DataFrame({
        'account_id': [2, 2, 2, 2, 2],
        'amount': [10.0, 20.0, 30.0, 40.0, 50.0],
        'TransactionDT': [0.0, 10.0*day, 35.0*day, 45.0*day, 50.0*day],
        'isFraud': [1, 0, 0, 1, 0]
    })
    res2 = compute_history_features_unified(df2, k=2, delay_days=30)
    print("\nTest 2: Combined Lag (k=2) and Delay (D=30) Interaction")
    print(res2[['hist_tx_count', 'hist_fraud_count', 'hist_fraud_rate', 'hist_fraud_time_since']])
    assert res2['hist_fraud_count'].iloc[2] == 1.0, "Test 2 Failed at tx 2!"
    assert res2['hist_fraud_count'].iloc[4] == 1.0, "Test 2 Failed at tx 4 (lag should exclude tx 3)!"
    print("--> Test 2 PASSED: Strict k-lag and D-delay interaction verified.")

    # 3. Positive Control vs Strict Delay Comparison (D=0 vs D=30)
    # Account 3: Fraud at t=10d. Next tx at t=15d.
    # Under D=0: Fraud at t=10d is immediately visible at t=15d (k=1). Expect 1.
    # Under D=30: Fraud at t=10d is masked until t=40d. At t=15d, expect 0.
    df3 = pd.DataFrame({
        'account_id': [3, 3],
        'amount': [100.0, 200.0],
        'TransactionDT': [10.0*day, 15.0*day],
        'isFraud': [1, 0]
    })
    res3_d0 = compute_history_features_unified(df3, k=1, delay_days=0)
    res3_d30 = compute_history_features_unified(df3, k=1, delay_days=30)
    print("\nTest 3: Positive Control (D=0 vs D=30)")
    print(f"  D=0 at t=15d:  hist_fraud_count = {res3_d0['hist_fraud_count'].iloc[1]}")
    print(f"  D=30 at t=15d: hist_fraud_count = {res3_d30['hist_fraud_count'].iloc[1]}")
    assert res3_d0['hist_fraud_count'].iloc[1] == 1.0, "Test 3 Failed: D=0 failed to see prior fraud!"
    assert res3_d30['hist_fraud_count'].iloc[1] == 0.0, "Test 3 Failed: D=30 leaked unconfirmed fraud!"
    print("--> Test 3 PASSED: Positive control proves D=0 detects fraud while D=30 hides unconfirmed label.")

    # 4. Multi-Fraud Cumulative Count Verification
    # Account 4: Frauds at t=0d, t=10d, t=20d.
    # Evaluation at t=45d.
    # At t=45d, cutoff is 45 - 30 = 15d.
    # Frauds <= 15d are t=0d and t=10d (2 frauds).
    # Fraud at t=20d is masked (20d > 15d).
    df4 = pd.DataFrame({
        'account_id': [4, 4, 4, 4],
        'amount': [10.0, 20.0, 30.0, 40.0],
        'TransactionDT': [0.0, 10.0*day, 20.0*day, 45.0*day],
        'isFraud': [1, 1, 1, 0]
    })
    res4 = compute_history_features_unified(df4, k=1, delay_days=30)
    print("\nTest 4: Multi-Fraud Cumulative Delay Masking")
    print(res4[['hist_tx_count', 'hist_fraud_count', 'hist_fraud_rate', 'hist_fraud_time_since']])
    assert res4['hist_fraud_count'].iloc[3] == 2.0, f"Test 4 Failed: Expected 2 frauds, got {res4['hist_fraud_count'].iloc[3]}"
    assert res4['hist_fraud_time_since'].iloc[3] == (45.0 - 10.0) * day, "Test 4 Failed: time since last confirmed fraud mismatch!"
    print("--> Test 4 PASSED: Multi-fraud masking correctly counts 2 confirmed frauds and hides 1 unconfirmed.")

    print("\n" + "=" * 80)
    print("ALL 4 SYNTHETIC REAL-FEATURE-BUILDER AUDIT TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 80)

if __name__ == '__main__':
    main()
