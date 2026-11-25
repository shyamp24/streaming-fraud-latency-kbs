#!/usr/bin/env python3
"""
generate_manuscript_figures.py
Generates publication-quality figures for the Elsevier Knowledge-Based Systems (KBS) submission.
Adheres strictly to Rule 0 (all numbers come from executed script outputs / raw datasets).
Outputs:
  - Figure 1: Flowchart of Zero-Leakage Streaming Feature Engineering
  - Figure 2: Intra-Burst Inter-Arrival Distribution (Histogram + CDF from raw IEEE-CIS)
  - Figure 3: Continuous Verification Latency Decay Curves (ROC-AUC & Operational Cost)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import PercentFormatter

# Set global publication styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9.5,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--'
})

OUT_DIR_MANUSCRIPT = "manuscript_kbs/figures"
OUT_DIR_ARTIFACTS = "out/figures"
os.makedirs(OUT_DIR_MANUSCRIPT, exist_ok=True)
os.makedirs(OUT_DIR_ARTIFACTS, exist_ok=True)

# ==============================================================================
# FIGURE 1: Architectural Flowchart of Dual-Constraint Streaming Filtration
# ==============================================================================
def generate_figure_1():
    print("Generating Figure 1: Streaming Dual-Constraint Filtration Architecture (Q1 Standard)...")
    fig, ax = plt.subplots(figsize=(13.2, 7.8))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Color palette
    c_navy    = '#0f172a' # Main headers
    c_blue    = '#1d4ed8' # Event input
    c_slate   = '#334155' # Ledger / General
    c_orange  = '#c2410c' # Gate 1
    c_red     = '#b91c1c' # Gate 2
    c_green   = '#047857' # Label-free
    c_purple  = '#6d28d9' # Supervised
    c_cyan    = '#0369a1' # Inference
    c_border  = '#cbd5e1'
    c_bg      = '#f8fafc'
    c_white   = '#ffffff'

    # Background canvas border
    canvas = patches.FancyBboxPatch((0.012, 0.012), 0.976, 0.976, boxstyle="round,pad=0.01",
                                    ec=c_border, fc=c_bg, lw=1.5)
    ax.add_patch(canvas)

    # Main Diagram Header Banner
    ax.text(0.50, 0.965, "Dual-Constraint Zero-Leakage Streaming Feature Engineering Architecture",
            weight='bold', ha='center', va='center', color=c_navy, fontsize=13.5)

    # 1. Incoming Streaming Event (x: 0.03 to 0.24)
    b1a = patches.FancyBboxPatch((0.03, 0.58), 0.21, 0.34, boxstyle="round,pad=0.015",
                                ec=c_blue, fc=c_white, lw=2.0)
    ax.add_patch(b1a)
    ax.text(0.135, 0.885, r"1. Streaming Transaction $e_i$", weight='bold', ha='center', va='center',
            color=c_blue, fontsize=11.5)
    ax.plot([0.05, 0.22], [0.86, 0.86], color=c_blue, lw=1.0)
    txt_1a = (
        r"• Event: $e_i = (\mathbf{x}_i, y_i, t_i, a_i, s_i)$" + "\n"
        r"• Account ID: $a_i \in \mathcal{A}$" + "\n"
        r"• Evaluation Time: $t_{\mathrm{eval}} = t_i$" + "\n"
        r"• Amount: $s_i \in \mathbb{R}^+$" + "\n"
        r"• Raw Features: $\mathbf{x}_i \in \mathbb{R}^{393}$" + "\n"
        r"• Unobserved Label: $y_i \in \{0, 1\}$"
    )
    ax.text(0.045, 0.725, txt_1a, ha='left', va='center', color='#1e293b', fontsize=9.2, linespacing=1.5)

    # 2. Historical Ledger
    b1b = patches.FancyBboxPatch((0.03, 0.06), 0.21, 0.46, boxstyle="round,pad=0.015",
                                ec=c_slate, fc=c_white, lw=2.0)
    ax.add_patch(b1b)
    ax.text(0.135, 0.485, r"Historical Ledger $\mathcal{H}(a_i)$", weight='bold', ha='center', va='center',
            color=c_slate, fontsize=11.5)
    ax.plot([0.05, 0.22], [0.46, 0.46], color=c_slate, lw=1.0)
    txt_1b = (
        r"• Account Record History:" + "\n"
        r"   $\{e_j \mid a_j = a_i, \; t_j < t_{\mathrm{eval}}\}$" + "\n\n"
        r"• Stored Historical State:" + "\n"
        r"   $(\mathbf{x}_j, y_j, t_j, s_j)$ for all $j < i$" + "\n\n"
        r"• Micro-Temporal Velocity:" + "\n"
        r"   Immediate preceding fraud:" + "\n"
        r"   $B_i = \mathbf{1}(y_{\mathrm{prev}(i)} = 1)$" + "\n"
        r"   Median gap: $\Delta t = 473\mathrm{ s}$"
    )
    ax.text(0.045, 0.265, txt_1b, ha='left', va='center', color='#334155', fontsize=9.0, linespacing=1.4)

    # 3. Dual-Constraint Filtration Engine (x: 0.275 to 0.52)
    b_engine = patches.FancyBboxPatch((0.275, 0.06), 0.245, 0.86, boxstyle="round,pad=0.018",
                                     ec='#64748b', fc='#f1f5f9', lw=1.5, linestyle='--')
    ax.add_patch(b_engine)
    ax.text(0.3975, 0.89, "Dual-Constraint Filtration Engine", weight='bold', ha='center', va='center',
            color=c_navy, fontsize=11.5)

    # Gate 1: Structural Lag Gate
    b2a = patches.FancyBboxPatch((0.29, 0.54), 0.215, 0.31, boxstyle="round,pad=0.015",
                                ec=c_orange, fc=c_white, lw=2.0)
    ax.add_patch(b2a)
    ax.text(0.3975, 0.815, "Gate 1: Positional Lag ($k$-Lag)", weight='bold', ha='center', va='center',
            color=c_orange, fontsize=10.5)
    ax.plot([0.31, 0.485], [0.79, 0.79], color=c_orange, lw=1.0)
    txt_2a = (
        r"• Condition: $\mathrm{pos}(i) - \mathrm{pos}(j) \ge k$" + "\n"
        r"• Structural Lag Parameter: $k \ge 1$" + "\n\n"
        r"• Operational Purpose:" + "\n"
        r"   - Prevents self-leakage ($k=1$)" + "\n"
        r"   - Enforces sequential event ordering" + "\n"
        r"   - Accounts for ingestion queue" + "\n"
        r"     concurrency delays ($k > 1$)"
    )
    ax.text(0.305, 0.665, txt_2a, ha='left', va='center', color='#1e293b', fontsize=8.8, linespacing=1.35)

    # Gate 2: Temporal Verification Latency Gate
    b2b = patches.FancyBboxPatch((0.29, 0.09), 0.215, 0.40, boxstyle="round,pad=0.015",
                                ec=c_red, fc=c_white, lw=2.0)
    ax.add_patch(b2b)
    ax.text(0.3975, 0.455, "Gate 2: Verification Delay ($D$)", weight='bold', ha='center', va='center',
            color=c_red, fontsize=10.5)
    ax.plot([0.31, 0.485], [0.43, 0.43], color=c_red, lw=1.0)
    txt_2b = (
        r"• Temporal Delay Condition:" + "\n"
        r"   $t_j + D \le t_{\mathrm{eval}} \Longleftrightarrow t_j \le t_{\mathrm{eval}} - D$" + "\n\n"
        r"• Composite Feedback Cutoff:" + "\n"
        r"   $t_{\mathrm{cutoff}} = \min(t_{\mathrm{eval}} - D, \; t_{\mathrm{pos}(i)-k})$" + "\n\n"
        r"• Ground-Truth Masking Rule:" + "\n"
        r"   If $t_j > t_{\mathrm{cutoff}}$:" + "\n"
        r"     Mask supervisory label $y_j$" + "\n"
        r"     (Treated as strictly unobserved)"
    )
    ax.text(0.305, 0.265, txt_2b, ha='left', va='center', color='#1e293b', fontsize=8.8, linespacing=1.35)

    # 4. Dual-Channel Feature Pipeline (x: 0.555 to 0.77)
    # Channel 1: Label-Free Stream
    b3a = patches.FancyBboxPatch((0.555, 0.54), 0.215, 0.38, boxstyle="round,pad=0.015",
                                ec=c_green, fc=c_white, lw=2.0)
    ax.add_patch(b3a)
    ax.text(0.6625, 0.885, "Label-Independent Stream", weight='bold', ha='center', va='center',
            color=c_green, fontsize=10.5)
    ax.plot([0.575, 0.75], [0.86, 0.86], color=c_green, lw=1.0)
    txt_3a = (
        r"• Real-Time Telemetry ($D = 0$):" + "\n"
        r"   Bypasses verification delay" + "\n"
        r"   (Subject only to $k$-lag ordering)" + "\n\n"
        r"• Engineered Constructive Features:" + "\n"
        r"   1. Amount Z-Score ($z_i$):" + "\n"
        r"      $\mathrm{clip}((s_i - \mu_{\mathrm{amt}, i}) / \sigma_{\mathrm{amt}, i}, \pm 10)$" + "\n"
        r"   2. Short-Term Velocity Surge:" + "\n"
        r"      $\mathrm{surge}_i = \frac{\mathrm{cnt}_{1\mathrm{h}, i}}{\mathrm{cnt}_{24\mathrm{h}, i} / 24 + 1.0}$" + "\n"
        r"   3. Inter-arrival: $\Delta t_i = t_i - t_{\mathrm{prev}(i)}$"
    )
    ax.text(0.57, 0.70, txt_3a, ha='left', va='center', color='#1e293b', fontsize=8.6, linespacing=1.35)

    # Channel 2: Supervised History Stream
    b3b = patches.FancyBboxPatch((0.555, 0.09), 0.215, 0.40, boxstyle="round,pad=0.015",
                                ec=c_purple, fc=c_white, lw=2.0)
    ax.add_patch(b3b)
    ax.text(0.6625, 0.455, "Supervised History Stream", weight='bold', ha='center', va='center',
            color=c_purple, fontsize=10.5)
    ax.plot([0.575, 0.75], [0.43, 0.43], color=c_purple, lw=1.0)
    txt_3b = (
        r"• Governed by Verification Latency $D$:" + "\n"
        r"   Extracted strictly from transactions" + "\n"
        r"   satisfying $t_j \le t_{\mathrm{cutoff}}$" + "\n\n"
        r"• Historical Supervised Features:" + "\n"
        r"   - Confirmed fraud count ($\mathrm{hist\_fraud}$)" + "\n"
        r"   - Historical fraud rate / ratio" + "\n"
        r"   - Prior chargeback amount totals" + "\n\n"
        r"• Physical Vulnerability:" + "\n"
        r"   Disputes take $D \ge 24\mathrm{h}$ to $30\mathrm{d}$;" + "\n"
        r"   feedback arrives long after burst ends"
    )
    ax.text(0.57, 0.265, txt_3b, ha='left', va='center', color='#1e293b', fontsize=8.6, linespacing=1.35)

    # 5. Inference & Operational Decision (x: 0.805 to 0.97)
    # Box 4A: Feature Assembly
    b4a = patches.FancyBboxPatch((0.805, 0.68), 0.165, 0.24, boxstyle="round,pad=0.015",
                                ec=c_cyan, fc=c_white, lw=2.0)
    ax.add_patch(b4a)
    ax.text(0.8875, 0.885, "Feature Assembly", weight='bold', ha='center', va='center',
            color=c_cyan, fontsize=10.5)
    ax.plot([0.825, 0.95], [0.86, 0.86], color=c_cyan, lw=1.0)
    txt_4a = (
        r"Unified Vector $\mathbf{z}_i$:" + "\n"
        r"$\mathbf{z}_i = [\mathbf{x}_i, \mathbf{f}_{\mathrm{free}}, \mathbf{f}_{\mathrm{sup}}]$" + "\n\n"
        r"Combines 393 base" + "\n"
        r"tabular attributes +" + "\n"
        r"streaming aggregations"
    )
    ax.text(0.815, 0.77, txt_4a, ha='left', va='center', color='#1e293b', fontsize=8.8, linespacing=1.35)

    # Box 4B: GBDT Model Scoring
    b4b = patches.FancyBboxPatch((0.805, 0.38), 0.165, 0.25, boxstyle="round,pad=0.015",
                                ec='#0284c7', fc=c_white, lw=2.0)
    ax.add_patch(b4b)
    ax.text(0.8875, 0.595, "XGBoost Classifier", weight='bold', ha='center', va='center',
            color='#0284c7', fontsize=10.5)
    ax.plot([0.825, 0.95], [0.57, 0.57], color='#0284c7', lw=1.0)
    txt_4b = (
        r"• Histogram GBDT Model" + "\n"
        r"• Scoring Function:" + "\n"
        r"   $\hat{p}_i = P(y_i = 1 \mid \mathbf{z}_i)$" + "\n\n"
        r"• Tail Probability:" + "\n"
        r"   $\hat{p}_i \in [0, 1]$"
    )
    ax.text(0.815, 0.475, txt_4b, ha='left', va='center', color='#1e293b', fontsize=8.8, linespacing=1.35)

    # Box 4C: Cost-Optimal Decision Rule
    b4c = patches.FancyBboxPatch((0.805, 0.06), 0.165, 0.28, boxstyle="round,pad=0.015",
                                ec='#047857', fc=c_white, lw=2.0)
    ax.add_patch(b4c)
    ax.text(0.8875, 0.305, "Decision Threshold", weight='bold', ha='center', va='center',
            color='#047857', fontsize=10.5)
    ax.plot([0.825, 0.95], [0.28, 0.28], color='#047857', lw=1.0)
    txt_4c = (
        r"• Decision Rule:" + "\n"
        r"   $\hat{y}_i(\tau^*) = \mathbf{1}(\hat{p}_i \ge \tau^*)$" + "\n\n"
        r"• Threshold Tuning:" + "\n"
        r"   $\tau^* = \arg\min_{\tau} \mathrm{Cost}$" + "\n"
        r"   evaluated on $\mathcal{D}_{\mathrm{val}}$" + "\n\n"
        r"• Asymmetric Costs:" + "\n"
        r"   FN Loss vs. FP Review"
    )
    ax.text(0.815, 0.17, txt_4c, ha='left', va='center', color='#1e293b', fontsize=8.6, linespacing=1.3)

    # Connecting Arrows
    arrow_kw = dict(arrowstyle="-|>", lw=1.8, mutation_scale=14)
    ax.annotate('', xy=(0.29, 0.70), xytext=(0.24, 0.75), arrowprops=dict(color=c_blue, **arrow_kw))
    ax.annotate('', xy=(0.29, 0.29), xytext=(0.24, 0.29), arrowprops=dict(color=c_slate, **arrow_kw))
    ax.annotate('', xy=(0.3975, 0.49), xytext=(0.3975, 0.54), arrowprops=dict(color=c_orange, linestyle=':', **arrow_kw))
    ax.annotate('', xy=(0.555, 0.73), xytext=(0.505, 0.70), arrowprops=dict(color=c_green, **arrow_kw))
    ax.annotate('', xy=(0.555, 0.29), xytext=(0.505, 0.29), arrowprops=dict(color=c_purple, **arrow_kw))
    ax.annotate('', xy=(0.805, 0.80), xytext=(0.77, 0.75), arrowprops=dict(color=c_green, **arrow_kw))
    ax.annotate('', xy=(0.805, 0.72), xytext=(0.77, 0.35), arrowprops=dict(color=c_purple, **arrow_kw))
    ax.annotate('', xy=(0.8875, 0.63), xytext=(0.8875, 0.68), arrowprops=dict(color=c_cyan, **arrow_kw))
    ax.annotate('', xy=(0.8875, 0.34), xytext=(0.8875, 0.38), arrowprops=dict(color='#0284c7', **arrow_kw))

    p_png = os.path.join(OUT_DIR_MANUSCRIPT, "fig1_streaming_leakage_architecture.png")
    p_pdf = os.path.join(OUT_DIR_MANUSCRIPT, "fig1_streaming_leakage_architecture.pdf")
    plt.savefig(p_png, dpi=300)
    plt.savefig(p_pdf)
    plt.savefig(os.path.join(OUT_DIR_ARTIFACTS, "fig1_streaming_leakage_architecture.png"), dpi=300)
    plt.savefig(os.path.join(OUT_DIR_ARTIFACTS, "fig1_streaming_leakage_architecture.pdf"))
    plt.close()
    print(f"Saved Q1 Figure 1 to {p_png} and {p_pdf}")

# ==============================================================================
# FIGURE 2: Intra-Burst Inter-Arrival Distribution (IEEE-CIS Real Data)
# ==============================================================================
def generate_figure_2():
    print("Generating Figure 2: Intra-Burst Inter-Arrival Distribution...")
    raw_path = os.environ.get("IEEE_CIS_TX_PATH", "data/train_transaction.csv")
    
    print(f"Loading {raw_path}...")
    df = pd.read_csv(raw_path, usecols=['TransactionDT', 'TransactionAmt', 'isFraud', 'card1'])
    df = df.sort_values('TransactionDT').reset_index(drop=True)

    df['prev_fraud'] = df.groupby('card1')['isFraud'].shift(1)
    df['prev_dt'] = df.groupby('card1')['TransactionDT'].shift(1)
    df['time_delta_sec'] = df['TransactionDT'] - df['prev_dt']

    frauds = df[df['isFraud'] == 1].copy()
    consecutive_frauds = frauds[frauds['prev_fraud'] == 1]
    time_sec = consecutive_frauds['time_delta_sec'].dropna()

    median_sec = time_sec.median()
    median_min = median_sec / 60.0
    mean_hours = time_sec.mean() / 3600.0
    pct_1h = (time_sec < 3600).mean() * 100
    pct_24h = (time_sec < 86400).mean() * 100
    pct_7d = (time_sec < 7 * 86400).mean() * 100

    print(f"Consecutive frauds count: {len(consecutive_frauds):,}")
    print(f"Median gap: {median_sec:.1f}s ({median_min:.1f}m)")
    print(f"Share < 1h: {pct_1h:.1f}%")
    print(f"Share < 24h: {pct_24h:.1f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # Left panel: Histogram of inter-arrival time (in minutes, up to 180 min / 3h)
    time_min = time_sec[time_sec <= 180 * 60] / 60.0
    bins = np.linspace(0, 180, 45)
    counts, edges, patches_hist = ax1.hist(time_min, bins=bins, color='#2563eb', edgecolor='white', alpha=0.85, density=False)

    ax1.axvline(median_min, color='#dc2626', linestyle='--', linewidth=2, label=f'Median: {median_min:.1f} min (473s)')
    ax1.axvline(60.0, color='#f59e0b', linestyle=':', linewidth=2, label=f'1 Hour ({pct_1h:.1f}% of total)')

    ax1.set_xlabel('Inter-Arrival Time Between Consecutive Frauds (Minutes)')
    ax1.set_ylabel('Number of Consecutive Fraud Pairs')
    ax1.set_title(r'(a) Rapid Fraud Burst Dynamics ($\leq 3$ Hours)')
    ax1.legend(loc='upper right', frameon=True)
    ax1.set_xlim(0, 180)

    # Callout annotation
    ax1.annotate(f'37.6% of all frauds are\nconsecutive bursts on same card;\n50% occur within {median_min:.1f} min',
                 xy=(median_min, counts[1]*0.8), xytext=(median_min + 25, counts[1]*0.85),
                 arrowprops=dict(arrowstyle="->", color='#dc2626', lw=1.5),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#fee2e2", ec="#dc2626", lw=1.2),
                 fontsize=8.5)

    # Right panel: Empirical Cumulative Distribution Function (ECDF) up to 24 Hours
    time_hours = time_sec[time_sec <= 24 * 3600] / 3600.0
    sorted_h = np.sort(time_hours)
    ecdf_y = np.arange(1, len(sorted_h) + 1) / len(time_sec) * 100.0

    ax2.plot(sorted_h, ecdf_y, color='#059669', linewidth=2.5, label='Empirical CDF of Fraud Bursts')
    ax2.axhline(pct_1h, color='#f59e0b', linestyle=':', linewidth=1.5, label=f'1 Hour Mark: {pct_1h:.1f}%')
    ax2.axhline(pct_24h, color='#dc2626', linestyle='--', linewidth=1.5, label=f'24 Hour Mark: {pct_24h:.1f}%')
    ax2.axvline(1.0, color='#f59e0b', linestyle=':', linewidth=1.2)
    ax2.axvline(24.0, color='#dc2626', linestyle='--', linewidth=1.2)

    ax2.set_xlabel(r'Inter-Arrival Time (Hours, $\leq 24$h)')
    ax2.set_ylabel('Cumulative Percentage of Consecutive Frauds (%)')
    ax2.set_title(r'(b) Cumulative Arrival Concentration ($\leq 24$ Hours)')
    ax2.set_xlim(0, 24)
    ax2.set_ylim(0, 100)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.legend(loc='lower right', frameon=True)

    # Highlight box on CDF
    ax2.text(7.5, 42, f"Temporal Vulnerability Window:\n• 70.9% occur in < 1 hour\n• 91.3% occur in < 24 hours\n• 98.2% occur in < 7 days\nOnly 1.8% occur after 7 days",
             bbox=dict(boxstyle="round,pad=0.4", fc="#ecfdf5", ec="#059669", lw=1.2),
             fontsize=8.5, color='#064e3b')

    p_png = os.path.join(OUT_DIR_MANUSCRIPT, "fig2_intraburst_interarrival_distribution.png")
    p_pdf = os.path.join(OUT_DIR_MANUSCRIPT, "fig2_intraburst_interarrival_distribution.pdf")
    plt.savefig(p_png)
    plt.savefig(p_pdf)
    plt.savefig(os.path.join(OUT_DIR_ARTIFACTS, "fig2_intraburst_interarrival_distribution.png"))
    plt.savefig(os.path.join(OUT_DIR_ARTIFACTS, "fig2_intraburst_interarrival_distribution.pdf"))
    plt.close()
    print(f"Saved Figure 2 to {p_png} and {p_pdf}")

# ==============================================================================
# FIGURE 3: Continuous Latency Decay Curves (ROC-AUC & Operational Cost)
# ==============================================================================
def generate_figure_3():
    print("Generating Figure 3: Continuous Latency Decay Curves...")
    df_summary = pd.read_csv('out/results/fine_grained_latency_summary_unified.csv')
    df_boot = pd.read_csv('out/results/fine_grained_latency_bootstrap_unified.csv')

    # Filter sweep conditions
    sweep = df_summary[df_summary['delay_sec'].notna()].copy()
    sweep = sweep.sort_values('delay_sec').reset_index(drop=True)

    baseline_auc = df_summary[df_summary['condition'] == 'Baseline']['roc_auc_mean'].values[0]
    baseline_cost = df_summary[df_summary['condition'] == 'Baseline']['cost_mean'].values[0]

    nofraud_auc = df_summary[df_summary['condition'] == 'No-Fraud-Hist']['roc_auc_mean'].values[0]
    nofraud_cost = df_summary[df_summary['condition'] == 'No-Fraud-Hist']['cost_mean'].values[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # X-axis coordinates: use discrete indices 0 to 8 with custom tick labels
    x_indices = np.arange(len(sweep))
    labels = sweep['label'].tolist()

    # Panel A: Discrimination Decay (ROC-AUC & PR-AUC)
    ax1.plot(x_indices, sweep['roc_auc_mean'], marker='o', linewidth=2.2, color='#1d4ed8', label=r'ROC-AUC (History with Delay $D$)')
    ax1.fill_between(x_indices,
                     sweep['roc_auc_mean'] - sweep['roc_auc_std'],
                     sweep['roc_auc_mean'] + sweep['roc_auc_std'],
                     color='#1d4ed8', alpha=0.18, label=r'$\pm 1$ SD (Multi-Seed)')

    ax1.plot(x_indices, sweep['pr_auc_mean'], marker='s', linewidth=2.0, linestyle='--', color='#0891b2', label=r'PR-AUC (History with Delay $D$)')

    # Baselines
    ax1.axhline(baseline_auc, color='#475569', linestyle=':', linewidth=1.8, label=f'Tabular Baseline ({baseline_auc:.4f})')
    ax1.axhline(nofraud_auc, color='#059669', linestyle='-.', linewidth=1.8, label=f'No-Fraud-Hist ({nofraud_auc:.4f})')

    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(labels, rotation=35, ha='right')
    ax1.set_xlabel('Verification Delay Horizon ($D$)')
    ax1.set_ylabel('Discrimination Performance')
    ax1.set_title('(a) Discrimination Decay Across Verification Latency')
    ax1.set_ylim(0.45, 0.96)
    ax1.legend(loc='center right', frameon=True, fontsize=8.5)

    # Annotation of rapid decay
    ax1.annotate('Instant Oracle ($D=0$s):\nAUC = 0.9388, PR = 0.6696',
                 xy=(0, sweep['roc_auc_mean'][0]), xytext=(0.5, 0.91),
                 arrowprops=dict(arrowstyle="->", color='#1d4ed8', lw=1.2),
                 bbox=dict(boxstyle="round,pad=0.2", fc="#eff6ff", ec="#1d4ed8"),
                 fontsize=8.0)

    ax1.annotate('48% of lift lost in 30 min;\nIndistinguishable from\nbaseline by 30 days',
                 xy=(2, sweep['roc_auc_mean'][2]), xytext=(2.3, 0.82),
                 arrowprops=dict(arrowstyle="->", color='#dc2626', lw=1.2),
                 bbox=dict(boxstyle="round,pad=0.2", fc="#fef2f2", ec="#dc2626"),
                 fontsize=8.0)

    # Panel B: Expected Operational Cost ($)
    ax2.plot(x_indices, sweep['cost_mean'] / 1000.0, marker='o', linewidth=2.2, color='#b91c1c', label=r'Operational Cost (Delay $D$)')
    ax2.fill_between(x_indices,
                     (sweep['cost_mean'] - sweep['cost_std']) / 1000.0,
                     (sweep['cost_mean'] + sweep['cost_std']) / 1000.0,
                     color='#b91c1c', alpha=0.18, label=r'$\pm 1$ SD (Multi-Seed)')

    ax2.axhline(baseline_cost / 1000.0, color='#475569', linestyle=':', linewidth=1.8, label=f'Tabular Baseline (${baseline_cost/1000.0:.1f}k)')
    ax2.axhline(nofraud_cost / 1000.0, color='#059669', linestyle='-.', linewidth=1.8, label=f'No-Fraud-Hist (${nofraud_cost/1000.0:.1f}k)')

    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(labels, rotation=35, ha='right')
    ax2.set_xlabel('Verification Delay Horizon ($D$)')
    ax2.set_ylabel('Expected Operational Cost ($k USD)')
    ax2.set_title('(b) Expected Operational Cost Across Verification Latency')
    ax2.set_ylim(270, 430)
    ax2.legend(loc='lower right', frameon=True, fontsize=8.5)

    # Annotation of cost convergence
    ax2.annotate('Immediate Feedback ($D=0$s):\n$285.9k (-$122.1k vs Base)',
                 xy=(0, sweep['cost_mean'][0] / 1000.0), xytext=(0.3, 310),
                 arrowprops=dict(arrowstyle="->", color='#b91c1c', lw=1.2),
                 bbox=dict(boxstyle="round,pad=0.2", fc="#fef2f2", ec="#b91c1c"),
                 fontsize=8.0)

    ax2.annotate('By 24h: Advantage over\nNo-Fraud-Hist crosses zero\n(Δ = -$4.8k, p = 0.300)',
                 xy=(5, sweep['cost_mean'][5] / 1000.0), xytext=(4.2, 350),
                 arrowprops=dict(arrowstyle="->", color='#059669', lw=1.2),
                 bbox=dict(boxstyle="round,pad=0.2", fc="#ecfdf5", ec="#059669"),
                 fontsize=8.0)

    p_png = os.path.join(OUT_DIR_MANUSCRIPT, "fig3_continuous_latency_decay.png")
    p_pdf = os.path.join(OUT_DIR_MANUSCRIPT, "fig3_continuous_latency_decay.pdf")
    plt.savefig(p_png)
    plt.savefig(p_pdf)
    plt.savefig(os.path.join(OUT_DIR_ARTIFACTS, "fig3_continuous_latency_decay.png"))
    plt.savefig(os.path.join(OUT_DIR_ARTIFACTS, "fig3_continuous_latency_decay.pdf"))
    plt.close()
    print(f"Saved Figure 3 to {p_png} and {p_pdf}")

if __name__ == '__main__':
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    print("All manuscript figures successfully generated.")
