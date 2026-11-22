"""Simulation of dual-channel mixture verification feedback."""
import numpy as np

def simulate_mixture_delay(q_fast_ratio, d_fast_sec=3600.0, d_slow_sec=2592000.0, random_seed=42):
    """Simulates mixed verification latency across channels."""
    rng = np.random.RandomState(random_seed)
    return {
        'fast_ratio': q_fast_ratio,
        'fast_sec': d_fast_sec,
        'slow_sec': d_slow_sec,
        'seed': random_seed
    }
