"""
code/cost_eval.py: Single Canonical Cost and Evaluation Module for IEEE-CIS Experiments.

This module is the SINGLE SOURCE OF TRUTH for operational cost schedules,
threshold tuning, and performance evaluation across all experimental scripts.

Cost Schedule:
- Quantile cutpoints derived from the training partition (70%):
  Cutpoints: [68.95, 125.0, 280.0, 554.0, 1651.0, 2734.99]
- 7 Escalating tiers: [10.0, 20.0, 30.0, 40.0, 60.0, 100.0, 200.0]
- Half-open interval definition: (prev_lim, lim]
  Implemented as np.searchsorted(CUTPOINTS, amounts, side='left')
- Operational Threshold Protocol:
  Operational threshold tau* is strictly tuned on the validation set (D_val).
  Test-set evaluations use the frozen validation-tuned tau*.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
)

# Canonical 7-Tier Quantile Brackets (Training Partition)
BASE_UNIT_USD = 10.0
ESCALATION = (1, 2, 3, 4, 6, 10, 20)
QUANTILE_CUTS = (0.50, 0.75, 0.90, 0.97, 0.995, 0.999)

# Exact IEEE-CIS 70% Train Amount Quantile Cutpoints
CUTPOINTS = np.array([68.95, 125.0, 280.0, 554.0, 1651.0, 2734.99], dtype=np.float64)
TIER_COSTS = np.array([10.0, 20.0, 30.0, 40.0, 60.0, 100.0, 200.0], dtype=np.float64)

BRACKETS = list(zip(list(CUTPOINTS) + [float('inf')], TIER_COSTS))

THRESH_GRID = np.linspace(0.01, 0.90, 100)


def fp_cost_vectorized(fp_amts):
    """Computes total operational false-positive cost using canonical 7-tier schedule.
    
    Equivalent bitwise to:
        costs = np.zeros(len(fp_amts))
        prev = -1.0
        for lim, c in BRACKETS:
            m = (fp_amts > prev) & (fp_amts <= lim)
            costs[m] = c
            prev = lim
    """
    if len(fp_amts) == 0:
        return 0.0
    idx = np.searchsorted(CUTPOINTS, fp_amts, side='left')
    return float(np.sum(TIER_COSTS[idx]))


def evaluate_cost(y_true, proba, amounts, tau):
    """Computes FN cost, FP cost, and Total Cost for a given decision threshold tau."""
    pred = (proba >= tau).astype(int)
    fn = float(np.sum(amounts[(y_true == 1) & (pred == 0)]))
    fp = fp_cost_vectorized(amounts[(y_true == 0) & (pred == 1)])
    return fn, fp, fn + fp


def tune_tau_validation(y_val, proba_val, amounts_val, thresh_grid=THRESH_GRID):
    """Finds optimal decision threshold tau* minimizing operational cost on validation set."""
    best_c = float('inf')
    best_tau = 0.5
    for tau in thresh_grid:
        pred = (proba_val >= tau).astype(int)
        fn = float(np.sum(amounts_val[(y_val == 1) & (pred == 0)]))
        fp = fp_cost_vectorized(amounts_val[(y_val == 0) & (pred == 1)])
        tot = fn + fp
        if tot < best_c:
            best_c = tot
            best_tau = tau
    return float(best_tau), float(best_c)


def evaluate_all_metrics(y_true, proba, amounts, tau):
    """Computes full set of classification and cost-sensitive performance metrics."""
    pred = (proba >= tau).astype(int)
    fn, fp, tot = evaluate_cost(y_true, proba, amounts, tau)

    # Fixed alert budget costs: top 1%, 2%, 5% alerts
    N = len(y_true)
    fixed_costs = {}
    for pct in [1.0, 2.0, 5.0]:
        k_alerts = int(np.round(N * (pct / 100.0)))
        cutoff_p = np.partition(proba, N - k_alerts)[N - k_alerts]
        pred_fixed = (proba >= cutoff_p).astype(int)
        fn_f = float(np.sum(amounts[(y_true == 1) & (pred_fixed == 0)]))
        fp_f = fp_cost_vectorized(amounts[(y_true == 0) & (pred_fixed == 1)])
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
        'total_cost': tot,
        **fixed_costs
    }
