"""Statistical significance + effect sizes for TCBG vs baselines (social only)."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path('/nas/home/jahin/TCBG_fair_eval/results')
SOCIAL = ['infectious_ct1', 'dblp_ct1', 'tumblr_ct1', 'mit_ct1', 'highschool_ct1', 'facebook_ct1']


def welch(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    d = (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else float('nan')
    return t, p, d


def load_seeds(fp, key):
    if not Path(fp).exists():
        return None
    d = json.load(open(fp))
    return np.asarray(d.get(key, []))


def main():
    print(f'{"Dataset":15s}  {"TCBG vs TempGNTK":>35s}  {"TCBG vs T3Former-fair":>40s}')
    print('-' * 100)
    social_tcbg, social_tempgntk, social_t3f = [], [], []

    for ds in SOCIAL:
        # TCBG seed-level fold means
        k = ds.replace('_ct1', '') + '_nc2'
        tcbg = json.load(open(BASE / f'tcbg_fair/{k}_fair.json'))
        tcbg_seeds = np.array(tcbg['seed_accs']) * 100

        # TempGNTK per-seed means (10 seeds from its protocol)
        tmp = json.load(open(BASE / f'tempgntk_fair/{ds}_fair.json'))
        tmp_seeds = np.array(tmp['seed_means']) * 100

        # T3Former per-seed means (StratifiedKFold(5) × 10 seeds)
        t3f_file = BASE / f't3former_fair/{ds}_fair.json'
        if t3f_file.exists():
            t3f = json.load(open(t3f_file))
            t3f_seeds = np.array(t3f['seed_means']) * 100
        else:
            t3f_seeds = None

        t1, p1, d1 = welch(tcbg_seeds, tmp_seeds)
        if t3f_seeds is not None:
            t2, p2, d2 = welch(tcbg_seeds, t3f_seeds)
            s2 = f't={t2:+.2f}  p={p2:.1e}  d={d2:+.2f}'
        else:
            s2 = '(no T3F seeds)'

        print(f'{ds:15s}  t={t1:+.2f}  p={p1:.1e}  d={d1:+.2f}  |  {s2}')

        social_tcbg.append(tcbg_seeds.mean())
        social_tempgntk.append(tmp_seeds.mean())
        if t3f_seeds is not None:
            social_t3f.append(t3f_seeds.mean())

    print()
    print(f'Mean social TCBG       = {np.mean(social_tcbg):.2f}')
    print(f'Mean social TempGNTK   = {np.mean(social_tempgntk):.2f}')
    print(f'Mean social T3Former   = {np.mean(social_t3f):.2f}')
    print(f'Gap TCBG vs TempGNTK   = {np.mean(social_tcbg) - np.mean(social_tempgntk):+.2f} pp')
    print(f'Gap TCBG vs T3F-fair   = {np.mean(social_tcbg) - np.mean(social_t3f):+.2f} pp')

    # Win rate
    wins_tmp = sum(1 for t, m in zip(social_tcbg, social_tempgntk) if t > m)
    wins_t3 = sum(1 for t, m in zip(social_tcbg, social_t3f) if t > m)
    n = len(social_tcbg)
    print(f'TCBG wins {wins_tmp}/{n} vs TempGNTK, {wins_t3}/{n} vs T3Former-fair')


if __name__ == '__main__':
    main()