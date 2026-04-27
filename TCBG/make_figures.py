"""
make_figures.py — NeurIPS-quality figures and LaTeX tables for TCBG paper.

Generates:
  figures/fig_main_results.pdf          Main accuracy vs T3Former
  figures/fig_ablation.pdf              Ablation grouped bar chart
  figures/fig_sensitivity_pipeline.pdf  T_grid / K_grid / min_persistence
  figures/fig_sensitivity_arch.pdf      gin_layers/hidden/dropout/lr/wd/ls
  figures/fig_convergence.pdf           Train/val loss + accuracy curves
  figures/fig_complexity.pdf            Runtime + GPU memory
  figures/tables.tex                    All LaTeX tables

Usage:
  python make_figures.py
"""

import json, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
FIGDIR  = ROOT / 'results' / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

# ── Global rcParams — NeurIPS style ──────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset':   'dejavuserif',
    'font.size':          9,
    'axes.titlesize':     9,
    'axes.labelsize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    7.5,
    'legend.framealpha':  0.6,          # semi-transparent
    'legend.facecolor':   'white',
    'legend.edgecolor':   '0.3',
    'legend.frameon':     True,
    'legend.fancybox':    False,
    'axes.linewidth':     0.8,
    'axes.edgecolor':     '#333333',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.color':         '#e0e0e0',
    'grid.linestyle':     ':',
    'grid.linewidth':     0.5,
    'grid.alpha':         0.5,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'xtick.major.size':   3,
    'ytick.major.size':   3,
    'figure.dpi':         300,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.03,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
})

EDGE  = 'black'
EDGEW = 0.6

# ── Color Palette ─────────────────────────────────────────────────────────────
# Carefully curated — high contrast, visually stunning, proven scientific combos
PALETTE = {
    'indigo':    '#3730A3',
    'sky':       '#0EA5E9',
    'teal':      '#0D9488',
    'emerald':   '#059669',
    'amber':     '#D97706',
    'rose':      '#E11D48',
    'violet':    '#7C3AED',
    'orange':    '#EA580C',
    'slate':     '#475569',
    'pink':      '#DB2777',
    'cyan':      '#0891B2',
    'lime':      '#65A30D',
}

# Dataset colors — consistent across ALL figures
DS_COLORS = {
    'infectious': '#E41A1C',  # red
    'dblp':       '#377EB8',  # blue
    'tumblr':     '#4DAF4A',  # green
    'mit':        '#FF7F00',  # orange
    'highschool': '#984EA3',  # purple
    'facebook':   '#FFDD33',  # yellow
}
DS_LABELS = {
    'infectious': 'Infectious',
    'dblp':       'DBLP',
    'tumblr':     'Tumblr',
    'mit':        'MIT',
    'highschool': 'Highschool',
    'facebook':   'Facebook',
}
DS_ORDER = ['infectious', 'dblp', 'tumblr', 'mit', 'highschool', 'facebook']

TCBG_COLOR    = '#1D4ED8'   # deep blue — hero color
T3F_COLOR     = '#DC2626'   # deep red — competitor
TCBG_EDGE     = '#1e3a8a'
T3F_EDGE      = '#7f1d1d'

# ── Load data ─────────────────────────────────────────────────────────────────
def load(path): return json.load(open(ROOT / path))

def safe_load(path):
    p = ROOT / path
    return json.load(open(p)) if p.exists() else None

GS    = {ds: safe_load(f'results_gs/{ds}_gs_result.json') for ds in DS_ORDER}
GS    = {k: v for k, v in GS.items() if v is not None}
ABL   = load('results/ablation/ablation_summary.json')
SENS  = load('results/sensitivity/sensitivity_summary.json')
CONV  = {ds: safe_load(f'results/convergence/convergence_{ds}.json') for ds in DS_ORDER}
CONV  = {k: v for k, v in CONV.items() if v is not None}
CMPLX = load('results/complexity/complexity_results.json')
RT    = load('results/complexity/runtime_results.json')

T3FORMER = {'infectious':68.50,'dblp':60.90,'tumblr':63.20,'mit':73.16,'highschool':67.20,'facebook':0.0}

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Main Results: TCBG vs T3Former
# ═══════════════════════════════════════════════════════════════════════════════
def fig_main_results():
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    ds_avail = [ds for ds in DS_ORDER if ds in GS]
    x        = np.arange(len(ds_avail))
    w        = 0.52
    means    = [GS[ds]['mean_acc']*100 for ds in ds_avail]
    stds     = [GS[ds]['std_acc']*100  for ds in ds_avail]
    colors   = [DS_COLORS[ds] for ds in ds_avail]

    bars = ax.bar(x, means, w, color=colors, edgecolor=EDGE,
                  linewidth=EDGEW, zorder=3)

    ax.errorbar(x, means, yerr=stds, fmt='none',
                color='#1a1a1a', capsize=3, capthick=1, linewidth=1, zorder=4)

    # Value labels on top of bars
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.8,
                f'{m:.1f}%', ha='center', va='bottom',
                fontsize=6.5, fontweight='bold', color='#1a1a1a')

    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABELS[d] for d in ds_avail], fontsize=8.5)
    ax.set_ylabel('Accuracy (%)', fontsize=9)
    ax.set_ylim(45, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0f}'))
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig_main_results.pdf')
    plt.close()
    print('  [OK] fig_main_results.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Ablation Study (key 8 variants, grouped by dataset)
# ═══════════════════════════════════════════════════════════════════════════════
ABL_KEY_VARIANTS = [
    ('full',           'Full TCBG',         '#1D4ED8'),
    ('no_global',      'w/o Global Feat.',  '#0EA5E9'),
    ('no_jk',          'w/o JK-Max',        '#0D9488'),
    ('no_curvature',   'w/o Curvature',     '#7C3AED'),
    ('no_global_no_jk','w/o Global+JK',     '#6366F1'),
    ('h0_only',        'H\u2080 only',       '#D97706'),
    ('h1_only',        'H\u2081 only',       '#EA580C'),
    ('k_grid_3',       'K=3 (near 1D)',     '#E11D48'),
]

def fig_ablation():
    ds_avail = [ds for ds in DS_ORDER if ds in ABL]
    n_ds  = len(ds_avail)
    n_var = len(ABL_KEY_VARIANTS)
    w     = 0.80 / n_var
    x     = np.arange(n_ds)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    for vi, (vkey, vlabel, vcol) in enumerate(ABL_KEY_VARIANTS):
        vals = []
        for ds in ds_avail:
            v = ABL.get(ds, {}).get(vkey, {})
            vals.append(v.get('mean', 0) * 100)
        offset = (vi - n_var/2 + 0.5) * w
        ax.bar(x + offset, vals, w,
               color=vcol, edgecolor=EDGE,
               linewidth=EDGEW, zorder=3, label=vlabel)

    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABELS[d] for d in ds_avail], fontsize=8.5)
    ax.set_ylabel('Accuracy (%)', fontsize=9)
    ax.set_ylim(40, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0f}'))
    ax.legend(loc='upper right', ncol=2, fontsize=6.5,
              columnspacing=0.8, handlelength=1.2,
              framealpha=0.6, facecolor='white', edgecolor='0.3')
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig_ablation.pdf')
    plt.close()
    print('  [OK] fig_ablation.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Sensitivity: Pipeline params (T_grid, K_grid, min_persistence)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_sensitivity_pipeline():
    params = [
        ('T_grid',          'T (Time Resolution)',         DS_ORDER, None),
        ('K_grid',          'K (Curvature Resolution)',    DS_ORDER, None),
        ('min_persistence', 'min_persistence',             DS_ORDER, None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))

    for ax, (pname, plabel, datasets, _) in zip(axes, params):
        pdata = SENS.get(pname, {})
        for ds in datasets:
            if ds not in pdata: continue
            d      = pdata[ds]
            vals   = d['values']
            means  = [m*100 for m in d['means']]
            stds   = [s*100 for s in d['stds']]
            defval = d['default_val']
            col    = DS_COLORS[ds]

            ax.plot(vals, means, color=col, linewidth=1.8,
                    marker='o', markersize=4, markeredgecolor='white',
                    markeredgewidth=0.8, zorder=3, label=DS_LABELS[ds])
            ax.fill_between(vals,
                            [m-s for m,s in zip(means,stds)],
                            [m+s for m,s in zip(means,stds)],
                            color=col, alpha=0.12, zorder=2)
            # Mark default
            di = vals.index(defval) if defval in vals else -1
            if di >= 0:
                ax.axvline(defval, color='#475569', linewidth=0.8,
                           linestyle=':', alpha=0.7, zorder=1)

        ax.set_xlabel(plabel, fontsize=8.5)
        ax.set_ylabel('Accuracy (%)' if ax == axes[0] else '', fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0f}'))
        ax.legend(fontsize=7, frameon=True)
        ax.set_axisbelow(True)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIGDIR / 'fig_sensitivity_pipeline.pdf')
    plt.close()
    print('  [OK] fig_sensitivity_pipeline.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Sensitivity: Architecture + Training params (2×3 grid)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_sensitivity_arch():
    params = [
        ('gin_layers',      'GIN Layers'),
        ('gin_hidden',      'Hidden Dim'),
        ('dropout',         'Dropout'),
        ('lr',              'Learning Rate'),
        ('weight_decay',    'Weight Decay'),
        ('label_smoothing', 'Label Smoothing'),
    ]
    # 2×3 grid for 6 params; legend placed as figure-level legend below
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8))
    axes_flat = axes.flatten()

    for ax, (pname, plabel) in zip(axes_flat, params):
        pdata = SENS.get(pname, {})
        plotted = False
        for ds in DS_ORDER:
            if ds not in pdata: continue
            d     = pdata[ds]
            vals  = d['values']
            means = [m*100 for m in d['means']]
            stds  = [s*100 for s in d['stds']]
            col   = DS_COLORS[ds]

            ax.plot(range(len(vals)), means, color=col, linewidth=1.6,
                    marker='o', markersize=3.5, markeredgecolor='white',
                    markeredgewidth=0.7, zorder=3, label=DS_LABELS[ds])
            ax.fill_between(range(len(vals)),
                            [m-s for m,s in zip(means,stds)],
                            [m+s for m,s in zip(means,stds)],
                            color=col, alpha=0.10, zorder=2)
            plotted = True

        if plotted:
            ax.set_xticks(range(len(vals)))
            xlabels = [str(v) if not isinstance(v, float) or v >= 0.001
                       else f'{v:.0e}' for v in vals]
            ax.set_xticklabels(xlabels, fontsize=6.5, rotation=20)
        ax.set_xlabel(plabel, fontsize=8)
        ax.set_ylabel('Accuracy (%)' if ax in [axes_flat[0], axes_flat[3]] else '', fontsize=8.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0f}'))
        ax.set_axisbelow(True)

    # Shared figure-level legend below all subplots
    handles = [mpatches.Patch(color=DS_COLORS[ds], label=DS_LABELS[ds])
               for ds in DS_ORDER]
    fig.legend(handles=handles, fontsize=7.5, frameon=True,
               loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.01),
               columnspacing=1.0, handlelength=1.2)

    fig.tight_layout(w_pad=1.2, h_pad=1.5)
    fig.subplots_adjust(bottom=0.10)
    fig.savefig(FIGDIR / 'fig_sensitivity_arch.pdf')
    plt.close()
    print('  [OK] fig_sensitivity_arch.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Convergence curves (train loss + val acc, 5 datasets)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_convergence():
    ds_avail = [ds for ds in DS_ORDER if ds in CONV]
    if not ds_avail:
        print('  [skip] fig_convergence (no data)'); return
    n = len(ds_avail)
    fig, axes = plt.subplots(2, n, figsize=(1.4*n + 0.5, 3.8), squeeze=False)
    loss_axes = axes[0]
    acc_axes  = axes[1]

    for i, ds in enumerate(ds_avail):
        d      = CONV[ds]
        epochs = d['epochs']
        col    = DS_COLORS[ds]

        # Loss
        ax = loss_axes[i]
        tl_m = d['train_loss_mean']
        tl_s = d['train_loss_std']
        vl_m = d['val_loss_mean']
        vl_s = d['val_loss_std']

        ax.plot(epochs, tl_m, color=col,       linewidth=1.5, label='Train', zorder=3)
        ax.plot(epochs, vl_m, color=col,       linewidth=1.5, linestyle='--',
                label='Val', zorder=3, alpha=0.8)
        ax.fill_between(epochs,
                        [max(0, m-s) for m,s in zip(tl_m,tl_s)],
                        [m+s for m,s in zip(tl_m,tl_s)],
                        color=col, alpha=0.10, zorder=2)
        ax.fill_between(epochs,
                        [max(0, m-s) for m,s in zip(vl_m,vl_s)],
                        [m+s for m,s in zip(vl_m,vl_s)],
                        color=col, alpha=0.10, zorder=2)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Epoch', fontsize=7.5)
        if i == 0: ax.set_ylabel('Loss', fontsize=8.5)
        ax.set_title(DS_LABELS[ds], fontsize=8, pad=3, color=col, fontweight='bold')
        ax.set_axisbelow(True)
        if i == 0:
            ax.legend(fontsize=6.5, frameon=True, loc='upper right')

        # Accuracy
        ax2 = acc_axes[i]
        ta_m = d['val_acc_mean']
        ta_s = d['val_acc_std']
        te_m = d['test_acc_at_best_mean']
        te_s = d['test_acc_at_best_std']

        ax2.plot(epochs, [v*100 for v in ta_m], color=col,
                 linewidth=1.5, label='Val Acc', zorder=3)
        ax2.plot(epochs, [v*100 for v in te_m], color=col,
                 linewidth=1.5, linestyle='--', label='Test Acc',
                 zorder=3, alpha=0.8)
        ax2.fill_between(epochs,
                         [(m-s)*100 for m,s in zip(te_m,te_s)],
                         [(m+s)*100 for m,s in zip(te_m,te_s)],
                         color=col, alpha=0.10, zorder=2)
        ax2.set_xlabel('Epoch', fontsize=7.5)
        if i == 0: ax2.set_ylabel('Accuracy (%)', fontsize=8.5)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0f}'))
        ax2.set_axisbelow(True)
        if i == 0:
            ax2.legend(fontsize=6.5, frameon=True, loc='lower right')

    fig.tight_layout(w_pad=0.8, h_pad=1.0)
    fig.savefig(FIGDIR / 'fig_convergence.pdf')
    plt.close()
    print('  [OK] fig_convergence.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Complexity: Training time + GPU memory (side by side)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_complexity():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))

    datasets = [ds for ds in DS_ORDER if ds in RT]
    colors   = [DS_COLORS[ds] for ds in datasets]
    labels   = [DS_LABELS[ds] for ds in datasets]
    x        = np.arange(len(datasets))
    w        = 0.55

    # Panel 1: Training time per fold (seconds)
    ax = axes[0]
    vals = [RT[ds]['train_per_fold_s'] for ds in datasets]
    bars = ax.bar(x, vals, w, color=colors, edgecolor=EDGE, linewidth=EDGEW, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f'{v:.1f}s', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5, rotation=15, ha='right')
    ax.set_ylabel('Train Time / Fold (s)', fontsize=8.5)
    ax.set_axisbelow(True)

    # Panel 2: Inference time (ms/graph)
    ax2 = axes[1]
    vals2 = [RT[ds]['inference_ms_graph'] for ds in datasets]
    bars2 = ax2.bar(x, vals2, w, color=colors, edgecolor=EDGE, linewidth=EDGEW, zorder=3)
    for bar, v in zip(bars2, vals2):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=7.5, rotation=15, ha='right')
    ax2.set_ylabel('Inference (ms/graph)', fontsize=8.5)
    ax2.set_axisbelow(True)

    # Panel 3: Peak GPU memory (MB)
    ax3 = axes[2]
    vals3 = [RT[ds]['peak_gpu_mb'] for ds in datasets]
    bars3 = ax3.bar(x, vals3, w, color=colors, edgecolor=EDGE, linewidth=EDGEW, zorder=3)
    for bar, v in zip(bars3, vals3):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                 f'{v:.0f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')
    ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=7.5, rotation=15, ha='right')
    ax3.set_ylabel('Peak GPU Memory (MB)', fontsize=8.5)
    ax3.set_axisbelow(True)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIGDIR / 'fig_complexity.pdf')
    plt.close()
    print('  [OK] fig_complexity.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Sensitivity diverging bar (delta from default, key params)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_sensitivity_delta():
    """Show change in accuracy relative to default value for key arch params."""
    params_show = ['gin_hidden', 'dropout', 'lr', 'label_smoothing']
    param_labels = {
        'gin_hidden':      'Hidden Dim',
        'dropout':         'Dropout',
        'lr':              'Learning Rate',
        'label_smoothing': 'Label Smoothing',
    }
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.8), sharey=False)

    for ax, pname in zip(axes, params_show):
        pdata = SENS.get(pname, {})
        for ds in DS_ORDER:
            if ds not in pdata: continue
            d      = pdata[ds]
            vals   = d['values']
            means  = [m*100 for m in d['means']]
            defval = d['default_val']
            di     = vals.index(defval) if defval in vals else -1
            if di < 0: continue
            baseline = means[di]
            deltas   = [m - baseline for m in means]
            col      = DS_COLORS[ds]

            ax.plot(range(len(vals)), deltas, color=col, linewidth=1.5,
                    marker='o', markersize=3.5, markeredgecolor='white',
                    markeredgewidth=0.7, zorder=3, label=DS_LABELS[ds])

        ax.axhline(0, color='#475569', linewidth=0.9, linestyle='--', alpha=0.8, zorder=1)
        ax.set_xticks(range(len(vals)))
        xlabels = [str(v) if not isinstance(v,float) or v >= 0.001
                   else f'{v:.0e}' for v in vals]
        ax.set_xticklabels(xlabels, fontsize=6.5, rotation=25, ha='right')
        ax.set_xlabel(param_labels[pname], fontsize=8)
        ax.set_ylabel(r'$\Delta$ Accuracy (%)' if ax == axes[0] else '', fontsize=8.5)
        ax.set_axisbelow(True)

    handles = [mpatches.Patch(color=DS_COLORS[ds], label=DS_LABELS[ds]) for ds in DS_ORDER]
    axes[-1].legend(handles=handles, fontsize=6.5, frameon=True,
                    loc='upper right', bbox_to_anchor=(1.0, 1.02))

    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIGDIR / 'fig_sensitivity_delta.pdf')
    plt.close()
    print('  [OK] fig_sensitivity_delta.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# TABLES — LaTeX
# ═══════════════════════════════════════════════════════════════════════════════
def make_tables():
    lines = []
    lines.append('% ===== AUTO-GENERATED LATEX TABLES =====\n')

    # ── Table 1: Ablation ──────────────────────────────────────────────────────
    ds_headers = ' & '.join([f'\\textbf{{{DS_LABELS[d]}}}' for d in DS_ORDER])
    lines.append('% --- Ablation Study ---')
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Ablation study results (mean accuracy \\%, 5 seeds $\\times$ 5-fold CV).}')
    lines.append('\\label{tab:ablation}')
    lines.append('\\small')
    lines.append('\\setlength{\\tabcolsep}{5pt}')
    lines.append('\\begin{tabular}{l' + 'c'*len(DS_ORDER) + '}')
    lines.append('\\toprule')
    lines.append(f'\\textbf{{Variant}} & {ds_headers} \\\\')
    lines.append('\\midrule')

    VARIANT_GROUPS = [
        ('Core Components', ['full','no_global','no_jk','no_curvature','no_global_no_jk']),
        ('Curvature Resolution', ['k_grid_3','k_grid_10']),
        ('Homology Dimension', ['h0_only','h1_only']),
        ('GIN Depth', ['gin_1layer','gin_2layers','gin_4layers']),
        ('Training Protocol', ['no_label_smooth','no_scheduler']),
    ]

    # Get labels from first dataset
    first_ds = DS_ORDER[0]
    vlabels  = {vk: ABL[first_ds][vk]['label'] for vk in ABL[first_ds]}

    for grp_name, vkeys in VARIANT_GROUPS:
        lines.append(f'\\multicolumn{{{len(DS_ORDER)+1}}}{{l}}{{\\textit{{{grp_name}}}}} \\\\')
        for vk in vkeys:
            row_vals = []
            for ds in DS_ORDER:
                v = ABL.get(ds, {}).get(vk, {})
                m = v.get('mean', 0) * 100
                s = v.get('std',  0) * 100
                # Bold if it's the full model
                cell = f'{m:.1f}$_{{\\pm{s:.1f}}}$'
                if vk == 'full':
                    cell = f'\\textbf{{{m:.1f}}}$_{{\\pm{s:.1f}}}$'
                row_vals.append(cell)
            vlab = vlabels.get(vk, vk).replace('κ', '$\\kappa$').replace('±','$\\pm$')
            lines.append(f'\\quad {vlab} & ' + ' & '.join(row_vals) + ' \\\\')
        lines.append('\\midrule')

    lines[-1] = '\\bottomrule'  # replace last midrule
    lines.append('\\end{tabular}')
    lines.append('\\end{table}\n')

    # ── Table 2: Sensitivity (key params, DBLP + Infectious) ──────────────────
    lines.append('% --- Sensitivity Analysis (Key Parameters) ---')
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Sensitivity analysis. Accuracy (\\%) on DBLP and Infectious as each parameter varies. $\\star$ = default value used in main experiments.}')
    lines.append('\\label{tab:sensitivity}')
    lines.append('\\small')
    lines.append('\\setlength{\\tabcolsep}{4pt}')

    sens_params = [
        ('T_grid',          'T (time res.)',           ['dblp','infectious']),
        ('K_grid',          'K (curvature res.)',      ['dblp','infectious']),
        ('min_persistence', 'min\\_pers.',             ['dblp','infectious']),
        ('gin_layers',      'GIN layers',              ['dblp','infectious']),
        ('gin_hidden',      'Hidden dim',              ['dblp','infectious']),
        ('dropout',         'Dropout',                 ['dblp','infectious']),
        ('lr',              'LR',                      ['dblp','infectious']),
    ]
    # Use dblp + infectious columns
    col_ds = ['dblp', 'infectious']
    col_h  = ' & '.join([f'\\textbf{{{DS_LABELS[d]}}}' for d in col_ds])

    # Find max column width needed
    max_vals = max(len(SENS[p]['dblp']['values']) for p,_,_ in sens_params if 'dblp' in SENS.get(p,{}))
    col_fmt  = 'l' + 'c'*max_vals*2 + 'c'  # rough

    lines.append(f'\\begin{{tabular}}{{llrrrrrrrr}}')
    lines.append('\\toprule')
    lines.append('\\textbf{Parameter} & \\textbf{Dataset} & \\multicolumn{7}{c}{\\textbf{Values}} \\\\')
    lines.append('\\midrule')

    for pname, plabel, ds_list in sens_params:
        pdata = SENS.get(pname, {})
        first = True
        for ds in ds_list:
            if ds not in pdata: continue
            d      = pdata[ds]
            vals   = d['values']
            means  = [m*100 for m in d['means']]
            defval = d['default_val']
            cells  = []
            for v, m in zip(vals, means):
                marker = '$^\\star$' if v == defval else ''
                cells.append(f'{m:.1f}{marker}')
            row_label = plabel if first else ''
            ds_label  = DS_LABELS[ds]
            # Pad to 7 cells
            while len(cells) < 7: cells.append('—')
            lines.append(f'{row_label} & {ds_label} & ' + ' & '.join(cells[:7]) + ' \\\\')
            first = False
        lines.append('\\midrule')

    lines[-1] = '\\bottomrule'
    lines.append('\\end{tabular}')
    lines.append('\\end{table}\n')

    # ── Table 3: Runtime ──────────────────────────────────────────────────────
    lines.append('% --- Runtime and Complexity ---')
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Computational complexity of TCBG across datasets.}')
    lines.append('\\label{tab:runtime}')
    lines.append('\\small')
    lines.append('\\setlength{\\tabcolsep}{5pt}')
    lines.append('\\begin{tabular}{lrrrrr}')
    lines.append('\\toprule')
    lines.append('\\textbf{Dataset} & \\textbf{\\#Graphs} & \\textbf{\\#Params} & '
                 '\\textbf{Train/fold (s)} & \\textbf{Infer (ms/g)} & \\textbf{GPU (MB)} \\\\')
    lines.append('\\midrule')
    for ds in DS_ORDER:
        if ds not in RT: continue
        r = RT[ds]
        lines.append(f'{DS_LABELS[ds]} & {r["n_graphs"]} & {r["n_params"]:,} & '
                     f'{r["train_per_fold_s"]:.2f} & {r["inference_ms_graph"]:.3f} & '
                     f'{r["peak_gpu_mb"]:.1f} \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}\n')

    # Write
    out = FIGDIR / 'tables.tex'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print('  [OK] tables.tex')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — t-SNE of graph-level representations (no model needed)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_tsne():
    import torch
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    CACHE_ROOT = ROOT / 'data' / 'cache'

    fig, axes = plt.subplots(1, 5, figsize=(7.0, 2.4))
    for ax, ds in zip(axes, DS_ORDER):
        cf = CACHE_ROOT / f'{ds}_nc2_T30_K20_mp0.05_mb20_nf8_gf21_epssocial.pt'
        if not cf.exists():
            ax.axis('off'); continue
        saved = torch.load(cf, weights_only=False)
        data_list, labels = saved['data_list'], saved['labels']

        # Graph-level repr: mean-pool node feats + global feats
        feats = []
        for g in data_list:
            mn = g.x.mean(0).numpy()          # 8-dim
            gf = g.gf.squeeze(0).numpy()      # 21-dim
            feats.append(np.concatenate([mn, gf]))
        X = StandardScaler().fit_transform(np.stack(feats))
        y = np.array(labels)

        perp = min(30, len(y) // 5)
        Z = TSNE(n_components=2, perplexity=perp, random_state=42,
                 max_iter=1000).fit_transform(X)

        for cls, marker, alpha in [(0,'o',0.7),(1,'^',0.7)]:
            idx = y == cls
            ax.scatter(Z[idx,0], Z[idx,1], s=8, alpha=alpha,
                       color=DS_COLORS[ds], marker=marker,
                       edgecolors='none', zorder=3,
                       label=f'Class {cls}')
        ax.set_title(DS_LABELS[ds], fontsize=8, color=DS_COLORS[ds],
                     fontweight='bold', pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    # Shared legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0],marker='o',color='#555',linestyle='None',
                      markersize=5,label='Class 0'),
               Line2D([0],[0],marker='^',color='#555',linestyle='None',
                      markersize=5,label='Class 1')]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               fontsize=7.5, bbox_to_anchor=(0.5,-0.04), frameon=True)
    fig.tight_layout(w_pad=0.5)
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(FIGDIR / 'fig_tsne.pdf')
    plt.close()
    print('  [OK] fig_tsne.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Curvature Evolution (Forman-Ricci κ over time, 2 example graphs)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_curvature_example():
    import sys
    sys.path.insert(0, str(ROOT))
    import networkx as nx
    from src.curvature import compute_forman_ricci, auto_epsilon
    from data.social_loader import load_social_dataset

    graphs, nodes, labels = load_social_dataset(
        'infectious', root=str(ROOT / 'data' / 'raw'), verbose=False)
    labels = np.array(labels)

    # Pick one graph per class with reasonable size
    idx0 = int(np.where(labels == 0)[0][2])
    idx1 = int(np.where(labels == 1)[0][2])
    examples = [(idx0, 0), (idx1, 1)]

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2))
    time_fracs = [0.25, 0.5, 0.9]

    for row, (gi, cls) in enumerate(examples):
        edges = graphs[gi]
        if not edges: continue
        eps = auto_epsilon(edges)
        curv = compute_forman_ricci(edges, eps)

        times = sorted(set(t for u,v,t,k in curv))
        if not times: continue

        for col, frac in enumerate(time_fracs):
            ax = axes[row][col]
            t_snap = times[min(int(frac * len(times)), len(times)-1)]
            snap_edges = [(u,v,k) for u,v,t,k in curv if t <= t_snap]

            G = nx.Graph()
            edge_curv = {}
            for u,v,k in snap_edges:
                G.add_edge(u, v)
                edge_curv[(min(u,v), max(u,v))] = k

            if len(G.nodes()) == 0:
                ax.axis('off'); continue

            pos = nx.spring_layout(G, seed=42, k=1.5)
            kvals = np.array([edge_curv.get((min(u,v),max(u,v)),0)
                              for u,v in G.edges()])

            if len(kvals) > 0:
                vmin, vmax = kvals.min(), kvals.max()
                if vmax <= vmin: vmax = vmin + 0.1
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.cm.RdYlGn
                edge_colors = [cmap(norm(k)) for k in kvals]
            else:
                edge_colors = ['gray']

            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=25,
                                   node_color=DS_COLORS['infectious'],
                                   alpha=0.85)
            nx.draw_networkx_edges(G, pos, ax=ax,
                                   edge_color=edge_colors,
                                   width=1.5, alpha=0.9)
            ax.set_title(f't={t_snap:.0f}', fontsize=7.5, pad=2)
            ax.axis('off')

            if col == 0:
                ax.set_ylabel(f'Class {cls}', fontsize=8,
                              labelpad=4, rotation=90)
                ax.yaxis.set_label_position('left')
                ax.set_frame_on(True)
                for sp in ax.spines.values(): sp.set_visible(False)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn,
                                norm=plt.Normalize(vmin=-2, vmax=2))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, orientation='vertical',
                      fraction=0.02, pad=0.02, shrink=0.7)
    cb.set_label('Forman-Ricci $\\kappa$', fontsize=8)
    cb.ax.tick_params(labelsize=7)

    fig.suptitle('', fontsize=1)
    fig.tight_layout(w_pad=0.3, h_pad=0.8)
    fig.savefig(FIGDIR / 'fig_curvature_example.pdf')
    plt.close()
    print('  [OK] fig_curvature_example.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 10 — Bifiltration Grid Heatmap (T×K grid for example graphs)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_bifiltration_grid():
    import sys
    sys.path.insert(0, str(ROOT))
    from src.curvature import compute_forman_ricci, auto_epsilon
    from data.social_loader import load_social_dataset

    graphs, nodes, labels = load_social_dataset(
        'infectious', root=str(ROOT / 'data' / 'raw'), verbose=False)
    labels = np.array(labels)

    # Pick the graphs with the most edges per class for a rich visualization
    sizes = [len(g) for g in graphs]
    idx0 = int(max(np.where(labels == 0)[0], key=lambda i: sizes[i]))
    idx1 = int(max(np.where(labels == 1)[0], key=lambda i: sizes[i]))

    T, K = 30, 20
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    for ax, gi, cls in zip(axes, [idx0, idx1], [0, 1]):
        edges = graphs[gi]
        curv  = compute_forman_ricci(edges, auto_epsilon(edges))
        if not curv: ax.axis('off'); continue

        # Bin raw (t, kappa) values into T×K grid
        ts     = np.array([t for u,v,t,k in curv], dtype=float)
        ks     = np.array([k for u,v,t,k in curv], dtype=float)
        t_bins = np.linspace(ts.min(), ts.max() + 1e-9, T + 1)
        k_bins = np.linspace(ks.min(), ks.max() + 1e-9, K + 1)
        grid, _, _ = np.histogram2d(ts, ks, bins=[t_bins, k_bins])

        im = ax.imshow(grid.T, aspect='auto', origin='lower',
                       cmap='YlOrRd', interpolation='bilinear')
        ax.set_xlabel('Time Step T', fontsize=8.5)
        ax.set_ylabel(r'Curvature Level $\kappa$' if ax == axes[0] else '', fontsize=8.5)
        # Axis tick labels = actual t and κ values
        ax.set_xticks([0, T//2 - 0.5, T - 1])
        ax.set_xticklabels([f'{ts.min():.0f}',
                            f'{(ts.min()+ts.max())/2:.0f}',
                            f'{ts.max():.0f}'], fontsize=7)
        ax.set_yticks([0, K//2 - 0.5, K - 1])
        ax.set_yticklabels([f'{ks.min():.1f}',
                            f'{(ks.min()+ks.max())/2:.1f}',
                            f'{ks.max():.1f}'], fontsize=7)
        ax.set_title(f'Class {cls} — Infectious ({len(edges)} edges)',
                     fontsize=8, color=DS_COLORS['infectious'],
                     fontweight='bold', pad=3)
        plt.colorbar(im, ax=ax, label='Edge count', shrink=0.85,
                     pad=0.03).ax.tick_params(labelsize=6.5)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIGDIR / 'fig_bifiltration_grid.pdf')
    plt.close()
    print('  [OK] fig_bifiltration_grid.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 11 — Persistence Barcodes (H0 + H1, 2 example graphs, 2 classes)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_persistence_barcodes():
    """Aggregate persistence diagram (birth vs death) over 25 graphs per class."""
    import sys
    sys.path.insert(0, str(ROOT))
    import gudhi
    from data.social_loader import load_social_dataset

    graphs, nodes, labels = load_social_dataset(
        'infectious', root=str(ROOT / 'data' / 'raw'), verbose=False)
    labels = np.array(labels)

    def get_dgm(edges):
        st = gudhi.SimplexTree()
        unique_t = sorted(set(t for u,v,t in edges))
        t_map = {t: i for i, t in enumerate(unique_t)}
        max_fi = float(len(unique_t))
        seen = {}
        for u, v, t in sorted(edges, key=lambda x: x[2]):
            fi = float(t_map[t])
            if u not in seen: st.insert([u], filtration=fi); seen[u] = fi
            if v not in seen: st.insert([v], filtration=fi); seen[v] = fi
            st.insert([u, v], filtration=fi)
        st.compute_persistence()
        dgm = {0: [], 1: []}
        for hd, (b, d) in st.persistence():
            if hd in dgm:
                dgm[hd].append((b, min(d, max_fi + 1)))
        return dgm, max_fi

    hd_cols = {0: DS_COLORS['infectious'], 1: DS_COLORS['dblp']}
    hd_labs = {0: 'H\u2080', 1: 'H\u2081'}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    for ax, cls in zip(axes, [0, 1]):
        idxs = np.where(labels == cls)[0]
        np.random.seed(42)
        sample = np.random.choice(idxs, size=min(25, len(idxs)), replace=False)

        all_pts = {0: [], 1: []}
        all_max = []
        for gi in sample:
            try:
                dgm, mf = get_dgm(graphs[gi])
                all_max.append(mf)
                for hd in [0, 1]:
                    all_pts[hd].extend(dgm[hd])
            except Exception:
                pass

        global_max = max(all_max) + 1 if all_max else 20
        ax.plot([0, global_max], [0, global_max], color='#888',
                linewidth=0.8, linestyle='--', zorder=1, alpha=0.6)

        for hd in [0, 1]:
            pts = all_pts[hd]
            if not pts: continue
            b_arr = np.array([p[0] for p in pts])
            d_arr = np.array([p[1] for p in pts])
            inf_mask = d_arr >= global_max
            fin_mask = ~inf_mask
            if fin_mask.any():
                ax.scatter(b_arr[fin_mask], d_arr[fin_mask], s=14,
                           color=hd_cols[hd], alpha=0.5, edgecolors='none',
                           zorder=3, label=hd_labs[hd])
            if inf_mask.any():
                ax.scatter(b_arr[inf_mask], np.full(inf_mask.sum(), global_max),
                           s=20, color=hd_cols[hd], alpha=0.85, marker='^',
                           edgecolors='none', zorder=4,
                           label=f'{hd_labs[hd]} (ess.)')

        ax.set_xlim(-0.5, global_max + 0.5)
        ax.set_ylim(-0.5, global_max + 1.5)
        ax.set_xlabel('Birth', fontsize=8.5)
        ax.set_ylabel('Death' if ax == axes[0] else '', fontsize=8.5)
        ax.set_title(f'Class {cls} — Infectious (n=25)',
                     fontsize=8.5, color=DS_COLORS['infectious'],
                     fontweight='bold', pad=3)
        ax.legend(fontsize=6.5, frameon=True, loc='lower right', markerscale=1.4)
        ax.set_axisbelow(True)

    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIGDIR / 'fig_persistence_barcodes.pdf')
    plt.close()
    print('  [OK] fig_persistence_barcodes.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 12 — Global Feature Heatmap (21 features × 5 datasets, mean per class)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_global_feature_heatmap():
    import torch

    CACHE_ROOT = ROOT / 'data' / 'cache'
    FEAT_NAMES = [
        'log(|E|)', 'log(pairs)', 'log(N)', 'mean κ', 'std κ',
        'frac κ>0', 'frac κ<0', 'burstiness', 'temp. range',
        'edge density', 'deg. entropy', 'repeat frac', 'κ volatility',
        'Fiedler', 'spec. gap', 'spec. entropy', 'mean spec.',
        'std spec.', 'temp. activity', 'activity var.', 'graph energy'
    ]

    fig, axes = plt.subplots(1, 5, figsize=(7.0, 3.2), sharey=True)

    for ax, ds in zip(axes, DS_ORDER):
        cf = CACHE_ROOT / f'{ds}_nc2_T30_K20_mp0.05_mb20_nf8_gf21_epssocial.pt'
        if not cf.exists(): ax.axis('off'); continue

        saved = torch.load(cf, weights_only=False)
        data_list, labels = saved['data_list'], saved['labels']
        labels = np.array(labels)

        n_feat = data_list[0].gf.shape[-1]
        names  = FEAT_NAMES[:n_feat]

        gfs = np.stack([g.gf.squeeze(0).numpy() for g in data_list])
        classes = np.unique(labels)

        # Normalize each feature to [0,1] for visual clarity
        mn = gfs.min(0); mx = gfs.max(0)
        rng = np.where(mx > mn, mx - mn, 1.0)
        gfs_n = (gfs - mn) / rng

        # Stack mean per class: shape (n_classes, n_feat)
        mat = np.stack([gfs_n[labels == c].mean(0) for c in classes])

        im = ax.imshow(mat, aspect='auto', cmap='RdYlBu_r',
                       vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(DS_LABELS[ds], fontsize=7.5,
                     color=DS_COLORS[ds], fontweight='bold', pad=3)
        ax.set_xticks(range(n_feat))
        ax.set_xticklabels(names, fontsize=4.5, rotation=90, ha='center')
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels([f'C{c}' for c in classes],
                           fontsize=7, rotation=0)

    # Shared colorbar
    fig.colorbar(im, ax=axes, orientation='vertical',
                 fraction=0.015, pad=0.02, label='Normalized mean value',
                 shrink=0.8).ax.tick_params(labelsize=6.5)

    fig.tight_layout(w_pad=0.3)
    fig.savefig(FIGDIR / 'fig_global_feature_heatmap.pdf')
    plt.close()
    print('  [OK] fig_global_feature_heatmap.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 13 — Node Feature Distributions (8 features, DBLP by class)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_node_feature_dist():
    import torch
    from scipy.stats import gaussian_kde

    CACHE_ROOT = ROOT / 'data' / 'cache'
    FEAT_NAMES = ['birth', 'death', r'$\kappa_{norm}$', 'persistence',
                  'midpoint', r'$\log$ pers.', 'birth ratio',
                  r'$\kappa \times$ pers.']

    cf = CACHE_ROOT / 'dblp_nc2_T30_K20_mp0.05_mb20_nf8_gf21_epssocial.pt'
    saved = torch.load(cf, weights_only=False)
    data_list, labels = saved['data_list'], saved['labels']
    labels = np.array(labels)

    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.8))
    axes_flat = axes.flatten()
    class_colors = ['#3730A3', '#E11D48']

    for fi, ax in enumerate(axes_flat):
        for cls, col in enumerate(class_colors):
            vals = np.concatenate([
                g.x[:, fi].numpy() for g, lbl in zip(data_list, labels)
                if lbl == cls
            ])
            # Remove extreme outliers for KDE
            p1, p99 = np.percentile(vals, 1), np.percentile(vals, 99)
            vals_c = vals[(vals >= p1) & (vals <= p99)]
            if len(vals_c) < 5: continue

            try:
                kde = gaussian_kde(vals_c, bw_method=0.3)
                xs  = np.linspace(vals_c.min(), vals_c.max(), 200)
                ax.plot(xs, kde(xs), color=col, linewidth=1.6,
                        label=f'Class {cls}', alpha=0.9)
                ax.fill_between(xs, kde(xs), alpha=0.12, color=col)
            except Exception:
                pass

        ax.set_xlabel(FEAT_NAMES[fi], fontsize=7.5)
        ax.set_ylabel('Density' if fi % 4 == 0 else '', fontsize=7.5)
        ax.set_yticks([])
        ax.set_axisbelow(True)

    axes_flat[0].legend(fontsize=7, frameon=True, loc='upper right')
    fig.suptitle('DBLP Node Feature Distributions by Class',
                 fontsize=8.5, y=1.01)
    fig.tight_layout(w_pad=1.0, h_pad=1.5)
    fig.savefig(FIGDIR / 'fig_node_feature_dist.pdf')
    plt.close()
    print('  [OK] fig_node_feature_dist.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 14 — Pipeline Architecture Diagram
# ═══════════════════════════════════════════════════════════════════════════════
def fig_pipeline_diagram():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from matplotlib.patches import ArrowStyle

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    ax.axis('off')

    # Pipeline stages
    stages = [
        (0.55, 'Temporal\nGraph\n$\\mathcal{G}=(V,E,\\tau)$', '#1D4ED8', 'white'),
        (2.35, 'Forman-Ricci\nCurvature\n$\\kappa(e,t)$',      '#0D9488', 'white'),
        (4.15, '$(t,\\kappa)$\nBifiltration\n$T{\\times}K$ grid', '#7C3AED', 'white'),
        (5.95, 'Graphcode\n(1D PH via\nGUDHI)',                '#D97706', 'white'),
        (7.75, 'JK-Max\nGIN\nClassifier',                      '#E11D48', 'white'),
    ]

    box_w, box_h = 1.5, 1.8
    y_center = 1.5

    for x_center, label, color, tcolor in stages:
        bbox = FancyBboxPatch(
            (x_center - box_w/2, y_center - box_h/2), box_w, box_h,
            boxstyle='round,pad=0.08', linewidth=1.2,
            edgecolor='white', facecolor=color, zorder=3, alpha=0.93)
        ax.add_patch(bbox)
        ax.text(x_center, y_center, label, ha='center', va='center',
                fontsize=7.2, color=tcolor, fontweight='bold',
                zorder=4, linespacing=1.4)

    # Arrows between stages
    arrow_style = dict(arrowstyle='->', color='#334155',
                       lw=1.5, mutation_scale=14)
    gap = 0.12
    for i in range(len(stages) - 1):
        x_start = stages[i][0]   + box_w/2 + gap
        x_end   = stages[i+1][0] - box_w/2 - gap
        ax.annotate('', xy=(x_end, y_center), xytext=(x_start, y_center),
                    arrowprops=arrow_style, zorder=5)

    # Node/edge feature annotations below boxes
    annotations = [
        (0.55, '$|E|$ temporal edges'),
        (2.35, '$\\kappa{=}4{-}d_u{-}d_v{+}3\\Delta$'),
        (4.15, '$T{=}30,\\ K{=}20$'),
        (5.95, '8-dim node\n21-dim global'),
        (7.75, '3-layer JK-Max\nLabel smooth $0.1$'),
    ]
    for x_center, note in annotations:
        ax.text(x_center, y_center - box_h/2 - 0.18, note,
                ha='center', va='top', fontsize=5.8,
                color='#475569', style='italic', zorder=4)

    # Output label
    ax.text(9.5, y_center, 'Class\npred.',
            ha='center', va='center', fontsize=7.5,
            color='#15803d', fontweight='bold')
    ax.annotate('', xy=(9.15, y_center),
                xytext=(stages[-1][0] + box_w/2 + gap, y_center),
                arrowprops=dict(arrowstyle='->', color='#15803d',
                                lw=1.5, mutation_scale=14), zorder=5)

    fig.tight_layout(pad=0.3)
    fig.savefig(FIGDIR / 'fig_pipeline_diagram.pdf')
    plt.close()
    print('  [OK] fig_pipeline_diagram.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE — Dataset Statistics
# ═══════════════════════════════════════════════════════════════════════════════
def make_dataset_stats_table():
    import torch
    CACHE_ROOT = ROOT / 'data' / 'cache'

    DS_INFO = {
        'infectious': {'full': 'Infectious', 'domain': 'Contact'},
        'dblp':       {'full': 'DBLP',       'domain': 'Co-author'},
        'tumblr':     {'full': 'Tumblr',      'domain': 'Social'},
        'mit':        {'full': 'MIT',         'domain': 'Proximity'},
        'highschool': {'full': 'Highschool',  'domain': 'Contact'},
        'facebook':   {'full': 'Facebook',    'domain': 'Social'},
    }

    lines = []
    lines.append('% --- Dataset Statistics ---')
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Dataset statistics. $|\\mathcal{G}|$: graphs, '
                 '$\\bar{N}$: mean nodes, $\\bar{E}$: mean temporal edges, '
                 '$T$: time span, $C$: classes.}')
    lines.append('\\label{tab:datasets}')
    lines.append('\\small')
    lines.append('\\setlength{\\tabcolsep}{5pt}')
    lines.append('\\begin{tabular}{llrrrrr}')
    lines.append('\\toprule')
    lines.append('\\textbf{Dataset} & \\textbf{Domain} & '
                 '$|\\mathcal{G}|$ & $\\bar{N}$ & $\\bar{E}$ & $T$ & $C$ \\\\')
    lines.append('\\midrule')

    sys_path_added = False
    for ds in DS_ORDER:
        cf = CACHE_ROOT / f'{ds}_nc2_T30_K20_mp0.05_mb20_nf8_gf21_epssocial.pt'
        saved    = torch.load(cf, weights_only=False)
        dl, labs = saved['data_list'], saved['labels']
        n_graphs = len(dl)
        n_classes = len(set(labs))
        mean_nodes = np.mean([g.num_nodes for g in dl])

        # Get raw edge info from raw data
        if not sys_path_added:
            import sys; sys.path.insert(0, str(ROOT))
            sys_path_added = True
        from data.social_loader import load_social_dataset
        raw_g, _, _ = load_social_dataset(ds, root=str(ROOT/'data'/'raw'),
                                           verbose=False)
        mean_edges = np.mean([len(g) for g in raw_g])
        time_spans = [max(t for u,v,t in g) - min(t for u,v,t in g)
                      for g in raw_g if g]
        mean_T = np.mean(time_spans)

        info = DS_INFO[ds]
        lines.append(f'{info["full"]} & {info["domain"]} & '
                     f'{n_graphs} & {mean_nodes:.0f} & {mean_edges:.0f} & '
                     f'{mean_T:.0f} & {n_classes} \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    out = FIGDIR / 'table_dataset_stats.tex'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print('  [OK] table_dataset_stats.tex')


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE — Statistical Significance (Wilcoxon signed-rank test)
# ═══════════════════════════════════════════════════════════════════════════════
def make_significance_table():
    from scipy.stats import wilcoxon
    import json

    ABL_ROOT = ROOT / 'results' / 'ablation'
    GS_ROOT  = ROOT / 'results_gs'

    # Compare all ablation variants vs full using seed_means (paired Wilcoxon)
    COMPARE_VARIANTS = [
        ('no_curvature',    'w/o Curvature'),
        ('no_global',       'w/o Global Feat.'),
        ('no_jk',           'w/o JK-Max'),
        ('no_global_no_jk', 'w/o Global+JK'),
    ]

    lines = []
    lines.append('% --- Statistical Significance ---')
    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append('\\caption{Statistical significance: Wilcoxon signed-rank test '
                 '(one-sided, Full TCBG vs.\\ ablation variants, 5 seeds). '
                 '$p{<}0.05$: $^*$,\\ $p{<}0.01$: $^{**}$,\\ $p{<}0.001$: $^{***}$,\\ n.s.: not significant.}')
    lines.append('\\label{tab:significance}')
    lines.append('\\small')
    lines.append('\\setlength{\\tabcolsep}{4pt}')
    lines.append('\\begin{tabular}{ll' + 'r'*len(DS_ORDER) + '}')
    lines.append('\\toprule')
    ds_header = ' & '.join(f'\\textbf{{{DS_LABELS[d]}}}' for d in DS_ORDER)
    lines.append(f'\\textbf{{Variant}} & \\textbf{{Metric}} & {ds_header} \\\\')
    lines.append('\\midrule')

    for vkey, vlabel in COMPARE_VARIANTS:
        delta_row = [vlabel, '$\\Delta$\\%']
        pval_row  = ['',      '$p$-value']
        sig_row   = ['',      'Sig.']

        for ds in DS_ORDER:
            f = ABL_ROOT / f'ablation_{ds}.json'
            if not f.exists():
                delta_row.append('—'); pval_row.append('—'); sig_row.append('—')
                continue
            d    = json.load(open(f))
            vars_ = d.get('variants', d)
            full = np.array(vars_.get('full', {}).get('seed_means', []))
            base = np.array(vars_.get(vkey,  {}).get('seed_means', []))
            if len(full) == 0 or len(base) == 0:
                delta_row.append('—'); pval_row.append('—'); sig_row.append('—')
                continue

            n = min(len(full), len(base))
            fa, ba = full[:n]*100, base[:n]*100
            delta = fa.mean() - ba.mean()
            try:
                _, p = wilcoxon(fa, ba, alternative='greater')
            except Exception:
                p = 1.0

            sign = '+' if delta >= 0 else ''
            sig  = ('^{***}' if p < 0.001 else '^{**}' if p < 0.01
                    else '^{*}' if p < 0.05 else '\\text{n.s.}')
            p_str = f'{p:.3f}' if p >= 0.001 else f'{p:.1e}'
            delta_row.append(f'{sign}{delta:.1f}')
            pval_row.append(p_str)
            sig_row.append(f'${sig}$')

        lines.append(' & '.join(delta_row) + ' \\\\')
        lines.append(' & '.join(pval_row)  + ' \\\\')
        lines.append(' & '.join(sig_row)   + ' \\\\')
        lines.append('\\midrule')

    lines[-1] = '\\bottomrule'
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    out = FIGDIR / 'table_significance.tex'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print('  [OK] table_significance.tex')


if __name__ == '__main__':
    print(f'Saving figures to: {FIGDIR}\n')
    fig_main_results()
    fig_ablation()
    fig_sensitivity_pipeline()
    fig_sensitivity_arch()
    fig_convergence()
    fig_complexity()
    fig_sensitivity_delta()
    make_tables()
    print('\n--- Additional visualizations ---')
    fig_tsne()
    fig_curvature_example()
    fig_bifiltration_grid()
    fig_persistence_barcodes()
    fig_global_feature_heatmap()
    fig_node_feature_dist()
    print('\n--- Supplementary tables + diagrams ---')
    fig_pipeline_diagram()
    make_dataset_stats_table()
    make_significance_table()
    print(f'\nAll done. Files in {FIGDIR}')
