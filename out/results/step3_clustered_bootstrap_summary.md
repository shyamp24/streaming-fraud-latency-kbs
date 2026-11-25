# Step 3: Account-Clustered Bootstrap Analysis (B=1,000)

**Methodology:** Accounts (unique `card1` entities, $N=7,080$) are resampled with replacement $B=1,000$ times. All transactions belonging to selected accounts are evaluated as a cluster, capturing intra-account transaction correlation and providing cluster-robust standard errors and empirical 95% confidence intervals.

---

## Clustered Bootstrap Estimation Results

| Comparison | Metric | Mean Difference ($\Delta$) | Cluster Std Err | 95% Cluster-Robust CI | Empirical $p$-value | Conclusion |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **D=0 vs Baseline** | ROC-AUC | **+0.0634** | 0.0023 | [+0.0591, +0.0679] | $p < 0.001$ | Statistically significant gain (oracle condition) |
| | PR-AUC | **+0.1871** | 0.0051 | [+0.1771, +0.1971] | $p < 0.001$ | Statistically significant gain |
| | Total Cost ($) | **-$122,440** | $10,393 | [$-142,652, $-103,281] | $p < 0.001$ | Significant operational savings |
| **D=30 vs Baseline** | ROC-AUC | **-0.0018** | 0.0010 | [-0.0037, +0.0002] | $p = 0.086$ | **No meaningful gain detected** (CI upper bound $\le 0.000$) |
| | PR-AUC | **-0.0115** | 0.0018 | [-0.0148, -0.0082] | $p = 0.000$ | Slight decrease |
| | Total Cost ($) | **+802** | $5,427 | [$-10,145, $10,985] | $p = 0.886$ | **No cost savings** (CI spans zero) |
| **D=30 vs D=0** | ROC-AUC | **-0.0652** | 0.0020 | [-0.0690, -0.0612] | $p < 0.001$ | Significant degradation under 30d latency |
| | PR-AUC | **-0.1986** | 0.0050 | [-0.2089, -0.1890] | $p < 0.001$ | Significant precision collapse |
| | Total Cost ($) | **+123,241** | $10,871 | [$103,217, $144,460] | $p < 0.001$ | Complete erasure of cost reduction |
| **No-Fraud-Hist vs Base** | ROC-AUC | **+0.0016** | 0.0009 | [-0.0002, +0.0035] | $p = 0.094$ | **No meaningful gain detected** |
| | Total Cost ($) | **-1,092** | $4,753 | [$-10,576, $7,648] | $p = 0.810$ | Equivalent cost |
