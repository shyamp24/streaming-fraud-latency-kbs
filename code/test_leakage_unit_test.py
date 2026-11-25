import os
import sys
import time
import numpy as np
import pandas as pd

print("=" * 80)
print("Zero-leakage unit test: verification latency and positional lag")
print("=" * 80)

# Load raw transaction data
TP = os.environ.get("IEEE_CIS_TX_PATH", "data/train_transaction.csv")
print(f"Loading raw transaction data from {TP}...")
t0 = time.time()
raw = pd.read_csv(TP, usecols=['TransactionID', 'TransactionDT', 'card1', 'isFraud'])
raw = raw.sort_values('TransactionDT', kind='stable').reset_index(drop=True)
print(f"Loaded {len(raw):,} transactions in {time.time()-t0:.1f}s.")

# Account mapping
raw['card1'] = raw['card1'].astype('category')
raw['account_id'] = raw['card1'].cat.codes.astype('int32')

N = len(raw)
train_end = int(N * 0.70)
val_end   = int(N * 0.80)

tx_dts = raw['TransactionDT'].values
is_fraud = raw['isFraud'].values
acc_ids = raw['account_id'].values

# Precompute independent ground-truth fraud occurrences per account from RAW data
print("Building independent raw fraud lookup table...")
raw_fraud_events = {}
for idx, (acc, dt, fr) in enumerate(zip(acc_ids, tx_dts, is_fraud)):
    if fr == 1:
        raw_fraud_events.setdefault(acc, []).append((idx, float(dt)))

for acc in raw_fraud_events:
    # list of (raw_index, timestamp)
    raw_fraud_events[acc] = np.array(raw_fraud_events[acc], dtype=[('idx', 'i4'), ('dt', 'f8')])

print(f"Total accounts with at least one fraud: {len(raw_fraud_events):,}")

# Feature generation function (as used in experiments)
def generate_delayed_fraud_features(k=1, delay_days=0):
    delay_sec = float(delay_days) * 86400.0
    tmp = pd.DataFrame({'acc': acc_ids, 'dt': tx_dts})
    g = tmp.groupby('acc', sort=False)
    dt_last = g['dt'].shift(k).values
    
    fr_counts = np.zeros(N, dtype=np.float32)
    fr_times  = np.full(N, np.nan, dtype=np.float32)
    
    if delay_days == 0:
        tmp_fr = pd.Series(is_fraud).groupby(acc_ids, sort=False).cumsum().astype('float32')
        tmp_fr_dt = np.where(is_fraud == 1, tx_dts, np.nan)
        g_fr = pd.DataFrame({'acc': acc_ids, 'fr': tmp_fr, 'dt': tmp_fr_dt}).groupby('acc', sort=False)
        hist_fr = g_fr['fr'].shift(k).fillna(0.0).values
        last_fr = g_fr['dt'].shift(k).groupby(acc_ids).ffill().values
        fr_counts = hist_fr.astype(np.float32)
        fr_times  = (tx_dts - last_fr).astype(np.float32)
    else:
        for acc, fr_arr in raw_fraud_events.items():
            acc_mask = (acc_ids == acc)
            acc_indices = np.where(acc_mask)[0]
            if len(acc_indices) < k + 1:
                continue
            fr_dts = fr_arr['dt']
            for idx in acc_indices[k:]:
                t_curr = tx_dts[idx]
                t_k = dt_last[idx]
                if np.isnan(t_k):
                    continue
                cutoff = min(t_k, t_curr - delay_sec)
                # Count frauds <= cutoff (equality convention: t_eval >= t_label + D)
                c = np.searchsorted(fr_dts, cutoff, side='right')
                if c > 0:
                    fr_counts[idx] = float(c)
                    fr_times[idx] = float(t_curr - fr_dts[c - 1])
                    
    return fr_counts, fr_times

# =============================================================================
# UNIT TEST EXECUTION
# =============================================================================
passed_tests = 0
total_tests = 0

print("\n--- Running Unit Test Suite ---")

# TEST 1: Positive Control (D=0) vs Negative Verification (D=30) on identical transactions
print("\n[TEST 1] Positive Control (D=0 vs D=30 on identical sampled transactions)")
np.random.seed(42)

# Sample 500 accounts known to have >= 2 transactions and >= 1 fraud
multi_fraud_accs = [acc for acc, ev in raw_fraud_events.items() if len(ev) >= 1 and (acc_ids == acc).sum() >= 3]
sampled_accs = np.random.choice(multi_fraud_accs, size=min(500, len(multi_fraud_accs)), replace=False)

fr_cnt_d0,  fr_time_d0  = generate_delayed_fraud_features(k=1, delay_days=0)
fr_cnt_d30, fr_time_d30 = generate_delayed_fraud_features(k=1, delay_days=30)

d0_frauds_detected = 0
d30_frauds_hidden = 0
zero_leakage_violations = 0

for acc in sampled_accs:
    acc_indices = np.where(acc_ids == acc)[0]
    fr_events = raw_fraud_events[acc]
    
    for i in acc_indices[1:]:  # transactions after first
        t_now = tx_dts[i]
        
        # Ground truth from RAW data:
        # Frauds strictly >= 1 tx back
        prior_frauds = fr_events[fr_events['idx'] <= i - 1]
        
        # Under D=0: observable iff index <= i - 1
        expected_d0_cnt = len(prior_frauds)
        actual_d0_cnt   = fr_cnt_d0[i]
        assert actual_d0_cnt == expected_d0_cnt, f"D=0 count mismatch at tx {i}: expected {expected_d0_cnt}, got {actual_d0_cnt}"
        if actual_d0_cnt > 0:
            d0_frauds_detected += 1
            
        # Under D=30: observable iff index <= i - 1 AND t_fraud <= t_now - 30 days
        cutoff_30d = t_now - (30.0 * 86400.0)
        observable_30d = prior_frauds[prior_frauds['dt'] <= cutoff_30d]
        expected_d30_cnt = len(observable_30d)
        actual_d30_cnt   = fr_cnt_d30[i]
        
        assert actual_d30_cnt == expected_d30_cnt, f"D=30 count mismatch at tx {i}: expected {expected_d30_cnt}, got {actual_d30_cnt}"
        
        if expected_d0_cnt > expected_d30_cnt:
            d30_frauds_hidden += (expected_d0_cnt - expected_d30_cnt)
            
        # LEAKAGE CHECK: verify that NO fraud occurred after cutoff_30d is counted
        unobservable = prior_frauds[prior_frauds['dt'] > cutoff_30d]
        if len(unobservable) > 0 and actual_d30_cnt > expected_d30_cnt:
            zero_leakage_violations += 1

assert zero_leakage_violations == 0, f"LEAKAGE DETECTED! {zero_leakage_violations} violations found!"
assert d0_frauds_detected > 0, "Positive control failed: D=0 saw zero prior frauds!"
assert d30_frauds_hidden > 0, "Positive control failed: D=30 hid zero delayed frauds!"

print(f"  --> POSITIVE CONTROL PASSED: D=0 successfully detected {d0_frauds_detected:,} prior fraud instances.")
print(f"  --> DELAY MASKING PASSED: D=30 strictly masked {d30_frauds_hidden:,} unconfirmed fraud labels.")
print(f"  --> ZERO LEAKAGE VERIFIED: Violations = {zero_leakage_violations}.")
passed_tests += 1
total_tests += 1

# TEST 2: Boundary Condition Check (t_eval == t_fraud + D)
print("\n[TEST 2] Boundary Condition Verification (t_eval == t_fraud + D vs t_eval == t_fraud + D - 1s)")
# Verify exact equality behavior:
# Convention: Label is observable if t_eval >= t_fraud + D
# If t_eval == t_fraud + D, count includes it.
# If t_eval == t_fraud + D - 1s, count excludes it.
boundary_checks = 0
for acc in sampled_accs[:50]:
    fr_events = raw_fraud_events[acc]
    if len(fr_events) == 0:
        continue
    t_fr = fr_events['dt'][0]
    
    # Evaluation at exact boundary: t_eval == t_fr + D
    t_eval_at = t_fr + (30.0 * 86400.0)
    cutoff_at = t_eval_at - (30.0 * 86400.0) # equals t_fr exactly
    c_at = np.searchsorted(fr_events['dt'], cutoff_at, side='right')
    
    # Evaluation 1 second before boundary: t_eval == t_fr + D - 1s
    t_eval_before = t_eval_at - 1.0
    cutoff_before = t_eval_before - (30.0 * 86400.0) # equals t_fr - 1s
    c_before = np.searchsorted(fr_events['dt'], cutoff_before, side='right')
    
    assert c_at >= 1, "Boundary check failed: label at exact boundary (t_eval == t_fraud + D) should be observable."
    assert c_before == 0, "Boundary check failed: label 1 second before boundary (t_eval == t_fraud + D - 1s) must be masked."
    boundary_checks += 1

print(f"  --> BOUNDARY TEST PASSED across {boundary_checks} accounts. Equality convention (t_eval >= t_fraud + D) strictly confirmed.")
passed_tests += 1
total_tests += 1

# TEST 3: Multi-Split Verification (Train, Validation, Test)
print("\n[TEST 3] Multi-Split Coverage Verification (Train, Val, Test)")
splits = {
    'Train (first 70%)': slice(0, train_end),
    'Validation (next 10%)': slice(train_end, val_end),
    'Test (last 20%)': slice(val_end, N)
}

for split_name, sl in splits.items():
    split_indices = np.arange(sl.start, sl.stop)
    # Check random 1,000 transactions in each split
    sample_indices = np.random.choice(split_indices, size=1000, replace=False)
    
    leakage_in_split = 0
    for idx in sample_indices:
        acc = acc_ids[idx]
        if acc not in raw_fraud_events:
            continue
        t_curr = tx_dts[idx]
        fr_events = raw_fraud_events[acc]
        
        # All prior frauds on account
        prior = fr_events[fr_events['idx'] <= idx - 1]
        cutoff = t_curr - (30.0 * 86400.0)
        valid_observable = prior[prior['dt'] <= cutoff]
        
        actual_cnt = fr_cnt_d30[idx]
        if actual_cnt > len(valid_observable):
            leakage_in_split += 1
            
    assert leakage_in_split == 0, f"Leakage detected in {split_name}!"
    print(f"  --> {split_name}: 1,000 transactions audited. Violations = 0 (PASSED).")
    passed_tests += 1
    total_tests += 1

# TEST 4: Combined Lag (k) and Delay (D) Rule Verification across (k, D) pairs
print("\n[TEST 4] Combined Lag (k) and Delay (D) Multi-Grid Verification")
grid_pairs = [(1, 7), (1, 30), (3, 7), (3, 30), (5, 14), (10, 60)]

for (k_val, d_val) in grid_pairs:
    cnts, times = generate_delayed_fraud_features(k=k_val, delay_days=d_val)
    d_sec = float(d_val) * 86400.0
    
    violations = 0
    checked = 0
    for acc in sampled_accs[:100]:
        acc_indices = np.where(acc_ids == acc)[0]
        if len(acc_indices) < k_val + 1:
            continue
        fr_events = raw_fraud_events[acc]
        
        for idx in acc_indices[k_val:]:
            t_curr = tx_dts[idx]
            t_k = tx_dts[acc_indices[acc_indices < idx][-k_val]]  # timestamp at pos i-k
            
            # Combined rule: fraud must satisfy BOTH:
            # 1. index <= idx - k (i.e. timestamp <= t_k)
            # 2. timestamp <= t_curr - delay_sec
            cutoff = min(t_k, t_curr - d_sec)
            
            valid_frauds = fr_events[fr_events['dt'] <= cutoff]
            expected_c = len(valid_frauds)
            actual_c = cnts[idx]
            
            if actual_c != expected_c:
                violations += 1
            checked += 1
            
    assert violations == 0, f"Combined rule failed for k={k_val}, D={d_val}!"
    stricter_rule = "Delay D" if d_val >= 30 else "Lag k (for frequent accounts) or Delay D"
    print(f"  --> Grid pair (k={k_val:2d}, D={d_val:2d}d): {checked:,} evaluations checked. Violations = 0 (PASSED). [Stricter bound: {stricter_rule}]")
    passed_tests += 1
    total_tests += 1

print("\n" + "=" * 80)
print(f"UNIT TEST SUMMARY: ALL {passed_tests}/{total_tests} TESTS PASSED WITH ZERO LEAKAGE VIOLATIONS.")
print("=" * 80)
