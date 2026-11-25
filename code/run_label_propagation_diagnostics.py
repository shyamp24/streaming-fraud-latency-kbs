#!/usr/bin/env python3
"""
Diagnostic Script: Label Propagation vs Attacker Bursts on IEEE-CIS Fraud Benchmark
- Diagnostic A: Post-first-fraud transactions on a card, comparing subsequent FRAUD vs subsequent LEGITIMATE (Control Group)
  measuring sharing rates of addr1, P_emaildomain, and DeviceInfo (alongside reporting coverage).
- Diagnostic B: Behavioral comparison of intra-burst frauds (delta_t <= 1h) vs isolated frauds (> 1h)
  across ProductCD, card4, card6, and transaction amount distribution.
"""

import os, sys, time
import numpy as np
import pandas as pd

t0 = time.time()
print("=" * 80)
print("RUNNING DIAGNOSTICS: LABEL PROPAGATION VS ATTACKER BURSTS")
print("=" * 80)

TX_PATH = os.environ.get("IEEE_CIS_TX_PATH", "data/train_transaction.csv")
ID_PATH = os.environ.get("IEEE_CIS_ID_PATH", "data/train_identity.csv")

# 1. Load transaction table
print("\n1. Loading train_transaction.csv...")
cols_tx = ['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'card1', 'card4', 'card6', 'addr1', 'P_emaildomain', 'ProductCD']
df_tx = pd.read_csv(TX_PATH, usecols=cols_tx)
print(f"   Loaded {len(df_tx):,} transactions.")

# 2. Load identity table for DeviceInfo
print("2. Loading train_identity.csv for DeviceInfo...")
df_id = pd.read_csv(ID_PATH, usecols=['TransactionID', 'DeviceInfo'])
print(f"   Loaded {len(df_id):,} identity records (Coverage: {len(df_id)/len(df_tx)*100:.2f}% of transactions).")

# Merge DeviceInfo
df = df_tx.merge(df_id, on='TransactionID', how='left')
df = df.sort_values(['card1', 'TransactionDT']).reset_index(drop=True)

# ---------------------------------------------------------
# DIAGNOSTIC A: Post-First-Fraud Sharing Rates vs Control Group
# ---------------------------------------------------------
print("\n3. Computing Diagnostic A (Metadata Linkage with Control Group)...")

# Identify cards with at least one fraud
cards_with_fraud = set(df[df['isFraud'] == 1]['card1'].unique())
print(f"   Total cards with at least 1 fraud: {len(cards_with_fraud):,}")

# Filter to transactions on cards with fraud
df_f_cards = df[df['card1'].isin(cards_with_fraud)].copy()

# Find the first fraud timestamp and its metadata per card
first_frauds = df_f_cards[df_f_cards['isFraud'] == 1].groupby('card1').first().reset_index()
first_frauds = first_frauds[['card1', 'TransactionDT', 'addr1', 'P_emaildomain', 'DeviceInfo']].rename(columns={
    'TransactionDT': 'first_f_dt',
    'addr1': 'first_f_addr1',
    'P_emaildomain': 'first_f_email',
    'DeviceInfo': 'first_f_device'
})

df_f_cards = df_f_cards.merge(first_frauds, on='card1', how='left')

# Select transactions occurring strictly AFTER the first fraud on that card
post_first_f = df_f_cards[df_f_cards['TransactionDT'] > df_f_cards['first_f_dt']].copy()

print(f"   Transactions occurring strictly after a card's first fraud: {len(post_first_f):,}")
n_post_fraud = (post_first_f['isFraud'] == 1).sum()
n_post_legit = (post_first_f['isFraud'] == 0).sum()
print(f"     Subsequent Frauds: {n_post_fraud:,} ({n_post_fraud/len(post_first_f)*100:.2f}%)")
print(f"     Subsequent Legitimate (Control): {n_post_legit:,} ({n_post_legit/len(post_first_f)*100:.2f}%)")

# Compare sharing rates
post_f = post_first_f[post_first_f['isFraud'] == 1]
post_l = post_first_f[post_first_f['isFraud'] == 0]

def calc_match_rates(sub_df, name):
    # addr1 match
    has_both_addr = sub_df['addr1'].notna() & sub_df['first_f_addr1'].notna()
    addr_match = (sub_df.loc[has_both_addr, 'addr1'] == sub_df.loc[has_both_addr, 'first_f_addr1']).mean() * 100
    addr_cov = has_both_addr.mean() * 100
    
    # email match
    has_both_email = sub_df['P_emaildomain'].notna() & sub_df['first_f_email'].notna()
    email_match = (sub_df.loc[has_both_email, 'P_emaildomain'] == sub_df.loc[has_both_email, 'first_f_email']).mean() * 100
    email_cov = has_both_email.mean() * 100

    # device match
    has_both_dev = sub_df['DeviceInfo'].notna() & sub_df['first_f_device'].notna()
    if has_both_dev.sum() > 0:
        dev_match = (sub_df.loc[has_both_dev, 'DeviceInfo'] == sub_df.loc[has_both_dev, 'first_f_device']).mean() * 100
    else:
        dev_match = np.nan
    dev_cov = has_both_dev.mean() * 100

    return {
        'group': name,
        'count': len(sub_df),
        'addr1_cov_pct': addr_cov,
        'addr1_match_pct': addr_match,
        'email_cov_pct': email_cov,
        'email_match_pct': email_match,
        'device_cov_pct': dev_cov,
        'device_match_pct': dev_match
    }

res_a_fraud = calc_match_rates(post_f, 'Subsequent Fraud (T_f > T_first)')
res_a_legit = calc_match_rates(post_l, 'Subsequent Legitimate (Control Group)')

df_diag_a = pd.DataFrame([res_a_fraud, res_a_legit])
os.makedirs('out/results', exist_ok=True)
df_diag_a.to_csv('out/results/diagnostic_a_metadata_linkage.csv', index=False)
print("\n--- Diagnostic A Summary ---")
print(df_diag_a.to_string())

# ---------------------------------------------------------
# DIAGNOSTIC B: Intra-Burst Frauds vs Isolated Frauds
# ---------------------------------------------------------
print("\n4. Computing Diagnostic B (Behavioral Profile: Intra-Burst vs Isolated Frauds)...")

# Filter to all frauds
frauds = df[df['isFraud'] == 1].copy()
frauds['time_prev_fraud'] = frauds.groupby('card1')['TransactionDT'].diff()
frauds['is_intra_burst_1h'] = (frauds['time_prev_fraud'] <= 3600) & (frauds['time_prev_fraud'].notna())

burst_f = frauds[frauds['is_intra_burst_1h']]
iso_f = frauds[~frauds['is_intra_burst_1h']]

print(f"   Total Frauds: {len(frauds):,}")
print(f"   Intra-burst frauds (<= 1h gap): {len(burst_f):,} ({len(burst_f)/len(frauds)*100:.2f}%)")
print(f"   Isolated frauds (> 1h gap or first in card): {len(iso_f):,} ({len(iso_f)/len(frauds)*100:.2f}%)")

diag_b_stats = []

for name, sub in [('Intra-Burst Frauds (<=1h)', burst_f), ('Isolated Frauds (>1h / First)', iso_f)]:
    row = {
        'group': name,
        'count': len(sub),
        'amt_median': float(sub['TransactionAmt'].median()),
        'amt_mean': float(sub['TransactionAmt'].mean()),
        'amt_p25': float(sub['TransactionAmt'].quantile(0.25)),
        'amt_p75': float(sub['TransactionAmt'].quantile(0.75)),
        'productCD_W_pct': float((sub['ProductCD'] == 'W').mean() * 100),
        'productCD_C_pct': float((sub['ProductCD'] == 'C').mean() * 100),
        'productCD_R_pct': float((sub['ProductCD'] == 'R').mean() * 100),
        'card4_visa_pct': float((sub['card4'] == 'visa').mean() * 100),
        'card4_mc_pct': float((sub['card4'] == 'mastercard').mean() * 100),
        'card6_credit_pct': float((sub['card6'] == 'credit').mean() * 100),
        'card6_debit_pct': float((sub['card6'] == 'debit').mean() * 100),
    }
    diag_b_stats.append(row)

df_diag_b = pd.DataFrame(diag_b_stats)
df_diag_b.to_csv('out/results/diagnostic_b_behavioral_profile.csv', index=False)
print("\n--- Diagnostic B Summary ---")
print(df_diag_b.to_string())

print(f"\nAll diagnostics completed in {time.time()-t0:.1f}s. Results saved to out/results/.")
