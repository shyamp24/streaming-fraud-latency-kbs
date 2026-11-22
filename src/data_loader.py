"""Data loader module for chronological splitting and account encoding."""
import numpy as np
import pandas as pd

def load_and_split_data(csv_path, train_frac=0.70, val_frac=0.10):
    """Loads transaction data and partitions chronologically."""
    df = pd.read_csv(csv_path)
    df = df.sort_values('TransactionDT', kind='stable').reset_index(drop=True)
    
    # Account identifier mapping
    df['card1'] = df['card1'].astype('category')
    df['account_id'] = df['card1'].cat.codes.astype('int32')
    
    n_total = len(df)
    train_end = int(n_total * train_frac)
    val_end = int(n_total * (train_frac + val_frac))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df
