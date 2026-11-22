"""Label-free behavioral feature engineering (spending z-scores, velocity)."""
import numpy as np
import pandas as pd

def compute_label_free_features(df, acc_col='account_id', amt_col='TransactionAmt', time_col='TransactionDT'):
    """Extracts delay-invariant unsupervised behavioral aggregations."""
    g = df.groupby(acc_col, sort=False)
    
    # Running spend mean and std
    amt_mean = g[amt_col].transform(lambda x: x.shift(1).expanding().mean())
    amt_std = g[amt_col].transform(lambda x: x.shift(1).expanding().std())
    spend_zscore = (df[amt_col] - amt_mean) / (amt_std + 1e-5)
    
    # Running transaction frequency
    tx_count = g[time_col].cumcount()
    
    return pd.DataFrame({
        'spend_zscore': spend_zscore.fillna(0.0),
        'tx_count_prior': tx_count
    }, index=df.index)
