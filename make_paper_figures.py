"""NeurIPS-style paper figures. Vector PDF output, serif font.

Style conventions:
  - single-column width = 5.5 in  (NeurIPS 2024/25 single-column layout)
  - tick labels 7pt, axis labels 8.5pt, legend 7pt
  - contrastive palette: ColorBrewer Set1 / Paul Tol bright
  - every filled element has black edgecolor lw=0.6
  - legends: transparent bg (framealpha=0), thin frame, empty corner
  - no plot titles (captions go in the LaTeX source)
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── Global style ──────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Nimbus Roman', 'Times New Roman', 'Times'],
    'mathtext.fontset': 'dejavuserif',
    'axes.titlesize': 9,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'legend.frameon': True,
    'legend.framealpha': 0.6,       # semi-transparent white background
    'legend.edgecolor': '0.3',
    'legend.fancybox': False,
    'legend.borderpad': 0.35,
    'legend.handletextpad': 0.5,
    'legend.columnspacing': 0.9,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.35,
    'grid.linestyle': ':',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Contrastive, vivid palette — ColorBrewer Set1 / Paul Tol variants
PAL = {
    'red':    '#E41A1C',
    'blue':   '#377EB8',
    'green':  '#4DAF4A',
    'purple': '#984EA3',
    'orange': '#FF7F00',
    'yellow': '#FFDD33',
    'brown':  '#A65628',
    'pink':   '#F781BF',
    'grey':   '#999999',
    'cyan':   '#17BECF',
}
EDGE = 'black'
EDGEW = 0.6

BASE = Path('/nas/home/jahin/TCBG_fair_eval/results')
FIG_DIR = BASE / 'figures'
FIG_DIR.mkdir(exist_ok=True)

SOCIAL = ['infectious', 'dblp', 'tumblr', 'mit', 'highschool', 'facebook']
LABELS = ['Infect.', 'DBLP', 'Tumblr', 'MIT', 'Highsch.', 'Facebook']
DS_COLOR = {
    'infectious':  PAL['red'],
    'dblp':        PAL['blue'],
    'tumblr':      PAL['green'],
    'mit':         PAL['orange'],
    'highschool':  PAL['purple'],
}

METHOD_COLOR = {
    'TempGNTK':        PAL['grey'],
    'T3Former paper':  PAL['yellow'],
    'T3Former fair':   PAL['blue'],
    'TCBG':            PAL['red'],
}


def _save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(FIG_DIR / f'{name}.{ext}',
                    dpi=300 if ext == 'png' else None)
    plt.close(fig)
    print(f'  → {FIG_DIR/name}.pdf')


# ── Fig 1: seed-variance box plot (reviewer-facing) ──────────────
def fig_seed_variance():
    fig, ax = plt.subplots(figsize=(5.5, 2.4))
    method_colors = [PAL['grey'], PAL['blue'], PAL['red']]
    method_names  = ['TempGNTK', 'T3Former', 'TCBG']

    group_w = 0.22
    inter_group = 1.0
    intra_gap = 0.04
    positions_all, data_all, box_colors = [], [], []

    for gi, ds in enumerate(SOCIAL):
        k = ds + '_nc2'
        tcbg = np.array(json.load(open(BASE / f'tcbg_fair/{k}_fair.json'))['seed_accs']) * 100
        tmp  = np.array(json.load(open(BASE / f'tempgntk_fair/{ds}_ct1_fair.json'))['seed_means']) * 100
        t3f_f = BASE / f't3former_fair/{ds}_ct1_fair.json'
        t3f = (np.array(json.load(open(t3f_f))['seed_means']) * 100
               if t3f_f.exists() else np.array([np.nan]*10))
        trio = [tmp, t3f, tcbg]
        center = gi * inter_group
        offsets = [-(group_w + intra_gap), 0, (group_w + intra_gap)]
        for mi, d in enumerate(trio):
            positions_all.append(center + offsets[mi])
            data_all.append(d)
            box_colors.append(method_colors[mi])

    bp = ax.boxplot(data_all, positions=positions_all, widths=group_w,
                    patch_artist=True,
                    boxprops=dict(linewidth=0.6, edgecolor=EDGE),
                    whiskerprops=dict(linewidth=0.6, color=EDGE),
                    capprops=dict(linewidth=0.6, color=EDGE),
                    medianprops=dict(color=EDGE, linewidth=1.2),
                    flierprops=dict(marker='o', ms=2, markerfacecolor='none',
                                    markeredgecolor=EDGE, markeredgewidth=0.5))
    for patch, c in zip(bp['boxes'], box_colors):
        patch.set_facecolor(c); patch.set_alpha(0.75)

    rng = np.random.default_rng(0)
    for pos, d in zip(positions_all, data_all):
        if not np.isnan(d).all():
            xs = rng.normal(pos, 0.015, size=len(d))
            ax.scatter(xs, d, color=EDGE, s=3.5, zorder=3, alpha=0.55, linewidths=0)

    ax.set_xticks([gi * inter_group for gi in range(len(SOCIAL))])
    ax.set_xticklabels(LABELS)
    ax.set_ylabel('accuracy (%)')
    ax.grid(axis='y')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(-0.6, (len(SOCIAL) - 1) * inter_group + 0.6)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, edgecolor=EDGE, linewidth=0.6, alpha=0.75, label=n)
               for n, c in zip(method_names, method_colors)]
    leg = ax.legend(handles=handles, loc='upper right',
                    bbox_to_anchor=(0.995, 0.99), ncol=3,
                    framealpha=0.6, facecolor='white', edgecolor='0.3', handlelength=1.2,
                    handleheight=0.9)
    fig.tight_layout()
    _save(fig, 'fig_seed_variance')


# ── Fig 2: ablation heatmap ──────────────────────────────────────
def fig_ablation():
    rows = ['full', 'no_global', 'no_jk', 'no_curvature', 'no_global_no_jk',
            'k_grid_3', 'k_grid_10', 'h0_only', 'h1_only',
            'gin_1layer', 'gin_2layers', 'gin_4layers',
            'no_label_smooth', 'no_scheduler']
    labels = ['Full', r'w/o $\mathbf{x}^{\text{glob}}$', 'w/o JK',
              r'$K_{\text{grid}}=1$', r'w/o $\mathbf{x}^{\text{glob}}\!+$JK',
              r'$K_{\text{grid}}=3$', r'$K_{\text{grid}}=10$',
              r'$H_0$ only', r'$H_1$ only',
              'GIN-1', 'GIN-2', 'GIN-4',
              'no label-sm.', 'no sched.']
    mat = np.full((len(rows), len(SOCIAL)), np.nan)
    for j, ds in enumerate(SOCIAL):
        fp = BASE / f'tcbg_ablation/ablation_{ds}.json'
        if not fp.exists():
            continue
        d = json.load(open(fp))
        full_m = d['variants']['full']['mean']
        for i, r in enumerate(rows):
            if r in d['variants']:
                mat[i, j] = (d['variants'][r]['mean'] - full_m) * 100

    fig, ax = plt.subplots(figsize=(4.4, 5.0))
    v = max(np.nanmax(np.abs(mat)), 1.0)
    im = ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-v, vmax=v)
    for i in range(mat.shape[0] + 1):
        ax.axhline(i - 0.5, color=EDGE, lw=0.3)
    for j in range(mat.shape[1] + 1):
        ax.axvline(j - 0.5, color=EDGE, lw=0.3)
    ax.set_xticks(range(len(SOCIAL)))
    ax.set_xticklabels(LABELS, rotation=20, ha='right')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                col = 'white' if abs(mat[i, j]) > v * 0.55 else 'black'
                ax.text(j, i, f'{mat[i,j]:+.1f}', ha='center', va='center',
                        fontsize=6.5, color=col)
    cbar = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03,
                        label=r'$\Delta$ accuracy (pp)')
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.6)
    ax.tick_params(length=0)
    fig.tight_layout()
    _save(fig, 'fig_ablation_heatmap')


# ── Fig 3: sensitivity curves ────────────────────────────────────
def fig_sensitivity():
    params_in_order = ['T_grid', 'K_grid', 'min_persistence', 'gin_layers',
                       'gin_hidden', 'dropout', 'lr', 'weight_decay',
                       'label_smoothing']
    param_labels = {
        'T_grid': r'$T$', 'K_grid': r'$K$',
        'min_persistence': r'min persistence',
        'gin_layers': 'GIN layers', 'gin_hidden': 'GIN hidden',
        'dropout': 'dropout', 'lr': 'learning rate',
        'weight_decay': 'weight decay', 'label_smoothing': r'$\lambda_{\text{sm}}$',
    }
    avail = [p for p in params_in_order
             if (BASE / f'tcbg_sensitivity/sensitivity_{p}.json').exists()]
    if not avail:
        print('  (no sensitivity files yet)'); return

    cols = 3
    rows = (len(avail) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5, 1.75*rows), squeeze=False)

    markers = ['o', 's', '^', 'D', 'v']
    for k, p in enumerate(avail):
        ax = axes[k//cols][k%cols]
        d = json.load(open(BASE / f'tcbg_sensitivity/sensitivity_{p}.json'))
        log_x = p in ('lr', 'weight_decay')
        for mi, ds in enumerate(d['datasets']):
            means = np.asarray(d['datasets'][ds]['means']) * 100
            stds  = np.asarray(d['datasets'][ds]['stds']) * 100
            xs = d['values']
            col = DS_COLOR.get(ds, PAL['grey'])
            mk = markers[mi % len(markers)]
            try:
                xs_num = [float(x) for x in xs]
                ax.errorbar(xs_num, means, yerr=stds, marker=mk,
                            label=ds, color=col, capsize=2,
                            lw=1.1, ms=4.0, elinewidth=0.7,
                            markerfacecolor=col, markeredgecolor=EDGE,
                            markeredgewidth=0.5)
                if log_x:
                    ax.set_xscale('log')
            except Exception:
                ax.plot(range(len(xs)), means, marker=mk,
                        label=ds, color=col, lw=1.1, ms=4.0,
                        markerfacecolor=col, markeredgecolor=EDGE,
                        markeredgewidth=0.5)
                ax.set_xticks(range(len(xs)))
                ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel(param_labels.get(p, p))
        ax.grid(axis='both')
        ax.spines[['top', 'right']].set_visible(False)
        if k % cols == 0:
            ax.set_ylabel('accuracy (%)')
        ax.tick_params(length=2.5)

    # unified bottom-centered legend
    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, ncol=min(5, len(lbls)),
               loc='lower center', bbox_to_anchor=(0.5, -0.02),
               framealpha=0.6, facecolor='white', edgecolor='0.3', handlelength=1.4,
               handletextpad=0.5, columnspacing=1.0)

    for k in range(len(avail), rows*cols):
        axes[k//cols][k%cols].axis('off')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08 if rows > 1 else 0.20)
    _save(fig, 'fig_sensitivity')


# ── Fig 4: runtime ────────────────────────────────────────────────
def fig_runtime():
    fp = BASE / 'tcbg_runtime/runtime_results.json'
    if not fp.exists():
        print('  (no runtime file)'); return
    d = json.load(open(fp))
    datasets = [ds for ds in SOCIAL if ds in d]
    preprocess = [d[ds]['preprocess_total_s'] for ds in datasets]
    train = [d[ds]['train_total_s'] for ds in datasets]
    infer_us = [d[ds]['inference_ms_graph'] * 1000 for ds in datasets]  # us

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.2))
    x = np.arange(len(datasets))
    w = 0.65
    ax1.bar(x, preprocess, w, label='Graphcode preproc.',
            color=PAL['blue'], edgecolor=EDGE, linewidth=EDGEW)
    ax1.bar(x, train, w, bottom=preprocess, label='Training (5-fold)',
            color=PAL['orange'], edgecolor=EDGE, linewidth=EDGEW)
    ax1.set_xticks(x); ax1.set_xticklabels([LABELS[SOCIAL.index(ds)] for ds in datasets], rotation=20, ha='right')
    ax1.set_ylabel('wall-clock seconds')
    top1 = max(np.array(preprocess) + np.array(train))
    ax1.set_ylim(0, top1 * 1.3)
    ax1.legend(loc='upper left', framealpha=0.6, facecolor='white', edgecolor='0.3',
               handlelength=1.2, handletextpad=0.5)
    ax1.grid(axis='y')
    ax1.spines[['top', 'right']].set_visible(False)

    ax2.bar(x, infer_us, w, color=PAL['green'],
            edgecolor=EDGE, linewidth=EDGEW)
    ax2.set_xticks(x); ax2.set_xticklabels([LABELS[SOCIAL.index(ds)] for ds in datasets], rotation=20, ha='right')
    ax2.set_ylabel(r'inference $\mu$s / graph')
    top2 = max(infer_us)
    ax2.set_ylim(0, top2 * 1.25)
    ax2.grid(axis='y')
    ax2.spines[['top', 'right']].set_visible(False)
    for i, v in enumerate(infer_us):
        ax2.text(i, v + top2*0.02, f'{v:.0f}',
                 ha='center', va='bottom', fontsize=6.5)
    fig.tight_layout(w_pad=1.4)
    _save(fig, 'fig_runtime')


# ── Fig 5: social comparison (mean ± std, fair eval only) ────────
def fig_paper_vs_fair():
    tempgntk = json.load(open(BASE / 'tempgntk_fair/summary.json'))
    t3f_m, t3f_s, tcbg_m, tcbg_s, tmp_m, tmp_s = [], [], [], [], [], []
    for ds in SOCIAL:
        r3 = json.load(open(BASE / f't3former_fair/{ds}_ct1_fair.json'))
        rt = json.load(open(BASE / f'tcbg_fair/{ds}_nc2_fair.json'))
        t3f_m.append(r3['mean_acc'] * 100);  t3f_s.append(r3['std_acc'] * 100)
        tcbg_m.append(rt['mean_acc'] * 100); tcbg_s.append(rt['std_acc'] * 100)
        tmp_m.append(tempgntk[f'{ds}_ct1']['mean'] * 100)
        tmp_s.append(tempgntk[f'{ds}_ct1']['std']  * 100)

    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    x = np.arange(len(SOCIAL)); w = 0.26
    bar_kw = dict(edgecolor=EDGE, linewidth=EDGEW, error_kw=dict(
        elinewidth=0.6, capthick=0.6, capsize=2.5, ecolor=EDGE))
    ax.bar(x - w, tmp_m,  w, yerr=tmp_s,  label='TempGNTK', color=PAL['grey'], **bar_kw)
    ax.bar(x,     t3f_m,  w, yerr=t3f_s,  label='T3Former', color=PAL['blue'], **bar_kw)
    ax.bar(x + w, tcbg_m, w, yerr=tcbg_s, label='TCBG (ours)', color=PAL['red'], **bar_kw)

    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_ylabel('accuracy (%)')
    ax.set_ylim(40, 100)
    ax.legend(loc='upper left', ncol=3, framealpha=0.6, facecolor='white', edgecolor='0.3',
              handlelength=1.2, handletextpad=0.4, columnspacing=0.9,
              borderaxespad=0.3)
    ax.grid(axis='y')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    _save(fig, 'fig_social_comparison')


if __name__ == '__main__':
    for fn in [fig_seed_variance, fig_ablation, fig_sensitivity, fig_runtime, fig_paper_vs_fair]:
        try:
            fn()
        except Exception as e:
            print(f'  [warn] {fn.__name__}: {e}')