"""Feature engineering module enforcing structural lag (k) and delay (D)."""
import numpy as np
import pandas as pd

def build_delayed_features(acc_ids, timestamps, labels, k=1, delay_seconds=0.0):
    """Constructs longitudinal features subject to dual constraints."""
    n_samples = len(acc_ids)
    fraud_counts = np.zeros(n_samples, dtype=np.float32)
    time_since_fraud = np.full(n_samples, np.nan, dtype=np.float32)
    
    # Pre-index positive instances per account
    fraud_events = {}
    for idx, (acc, ts, lab) in enumerate(zip(acc_ids, timestamps, labels)):
        if lab == 1:
            fraud_events.setdefault(acc, []).append((idx, float(ts)))
            
    for acc in fraud_events:
        fraud_events[acc] = np.array(fraud_events[acc], dtype=[('idx', 'i4'), ('ts', 'f8')])
        
    tmp_df = pd.DataFrame({'acc': acc_ids, 'ts': timestamps})
    ts_pos_k = tmp_df.groupby('acc', sort=False)['ts'].shift(k).values
    
    if delay_seconds == 0.0:
        cum_fraud = pd.Series(labels).groupby(acc_ids, sort=False).cumsum().astype('float32')
        ts_fraud_only = np.where(labels == 1, timestamps, np.nan)
        grp = pd.DataFrame({'acc': acc_ids, 'fr': cum_fraud, 'ts': ts_fraud_only}).groupby('acc', sort=False)
        fraud_counts = grp['fr'].shift(k).fillna(0.0).values.astype(np.float32)
        last_fraud_ts = grp['ts'].shift(k).groupby(acc_ids).ffill().values
        time_since_fraud = (timestamps - last_fraud_ts).astype(np.float32)
    else:
        for acc, events in fraud_events.items():
            acc_indices = np.where(acc_ids == acc)[0]
            if len(acc_indices) < k + 1:
                continue
            ev_ts = events['ts']
            for i in acc_indices[k:]:
                t_eval = timestamps[i]
                t_k = ts_pos_k[i]
                if np.isnan(t_k):
                    continue
                t_cutoff = min(t_k, t_eval - delay_seconds)
                c = np.searchsorted(ev_ts, t_cutoff, side='right')
                if c > 0:
                    fraud_counts[i] = float(c)
                    time_since_fraud[i] = float(t_eval - ev_ts[c - 1])
                    
    return fraud_counts, time_since_fraud
