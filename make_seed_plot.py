"""Figure: per-seed accuracy distribution showing baselines are seed-sensitive
but TCBG is stable. Intended for NeurIPS paper appendix."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path('/nas/home/jahin/TCBG_fair_eval/results')
SOCIAL = ['infectious_ct1', 'dblp_ct1', 'tumblr_ct1', 'mit_ct1', 'highschool_ct1']
LABELS = ['Infectious', 'DBLP', 'Tumblr', 'MIT', 'Highschool']


def main():
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), sharey=False)
    for ax, ds, lab in zip(axes, SOCIAL, LABELS):
        k = ds.replace('_ct1', '') + '_nc2'
        tcbg = np.array(json.load(open(BASE / f'tcbg_fair/{k}_fair.json'))['seed_accs']) * 100
        tmp = np.array(json.load(open(BASE / f'tempgntk_fair/{ds}_fair.json'))['seed_means']) * 100
        t3f_file = BASE / f't3former_fair/{ds}_fair.json'
        t3f = (np.array(json.load(open(t3f_file))['seed_means']) * 100
               if t3f_file.exists() else None)

        data = [tmp, t3f if t3f is not None else np.array([np.nan]*10), tcbg]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True,
                        tick_labels=['TempGNTK', 'T3Former', 'TCBG'])
        colors = ['#d9d9d9', '#b3cde3', '#fbb4ae']
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
        for idx, d in enumerate(data):
            if not np.isnan(d).all():
                ax.scatter(np.random.normal(idx + 1, 0.04, size=len(d)),
                           d, color='k', s=8, zorder=3, alpha=0.7)
        ax.set_title(lab, fontsize=11)
        ax.set_ylabel('accuracy (%)' if ax is axes[0] else '')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=15, labelsize=9)
    fig.suptitle('Per-seed accuracy: 10 seeds × 5-fold CV, identical protocol',
                 y=1.03, fontsize=12)
    fig.tight_layout()
    out = BASE / 'tcbg_fair' / 'seed_variance.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    fig.savefig(str(out).replace('.png', '.pdf'), bbox_inches='tight')
    print(f'saved {out}')


if __name__ == '__main__':
    main()