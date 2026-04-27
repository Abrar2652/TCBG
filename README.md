# TCBG: Temporal Curvature Bifiltration Graphcodes

Temporal graph classification via 2-parameter persistence over (time, Forman-Ricci curvature), vectorized as a Graphcode and classified with a GIN.

This repo contains TCBG plus a unified fair-evaluation harness against two NeurIPS 2024 baselines — **T3Former** and **TempGNTK** — under one protocol: StratifiedKFold(5) × 10 seeds.


## Install

```bash
pip install torch torch_geometric scikit-learn numpy
pip install gudhi          # accurate persistence
pip install neurograph     # brain datasets only
pip install -r TCBG/requirements.txt
```

## Quick start

```bash
# TCBG only
python run_tcbg_fair.py --device cuda

# Baselines (clones TempGNTK, runs both fair evals, prints comparison)
python run_all.py --device cuda

# Skip a baseline
python run_all.py --device cuda --skip_t3former
python run_all.py --device cuda --skip_tempgntk
```

Per-domain entry points:

```bash
python run_t3former_fair.py            --device cuda 
python run_tempgntk_fair.py            --device cuda
```


## Repo layout

```
run_all.py                    one-shot harness for all baselines
run_tcbg_fair.py              TCBG across datasets
run_tempgntk_fair.py          TempGNTK fair eval
run_t3former_fair.py          T3Former social
compute_stats.py              Welch's t / Cohen's d
make_paper_figures.py         Figs 1–5 (PDF + PNG)
make_seed_plot.py             seed-variance box plots

TCBG/                         model code (curvature, bifiltration, Graphcode, GIN)
Baselines/                    Source codes of all baseline models

data/                         TU-format datasets
results/                      (symlink)
logs/                         per-run stdout/stderr
```

## Reproducing the headline numbers

```bash
# 1. TCBG on all datasets (10 seeds × 5-fold = 50 runs/dataset)
python run_tcbg_fair.py --device cuda --seeds 10

# 2. Baselines
python run_all.py --device cuda

# 3. Stats + figures
python compute_stats.py
python make_paper_figures.py
```



