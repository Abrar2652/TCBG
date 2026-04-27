# \# TCBG: Temporal Curvature Bifiltration Graphcodes

# 

# Temporal graph classification via 2-parameter persistence over (time, Forman-Ricci curvature), vectorized as a Graphcode and classified with a GIN.

# 

# This repo contains TCBG plus a unified fair-evaluation harness against two NeurIPS 2024 baselines — \*\*T3Former\*\* and \*\*TempGNTK\*\* — under one protocol: StratifiedKFold(5) × 10 seeds.

# 

# 

# \## Why fair re-evaluation

# 

# The published baselines use looser protocols:

# 

# \- \*\*T3Former\*\*: single `seed=42`, non-stratified KFold

# \- \*\*TempGNTK\*\*: fixed 70/30 split, single run

# \- \*\*TCBG (ours)\*\*: StratifiedKFold(5) × 10 seeds = 50 runs/dataset

# 

# We re-ran each baseline's own published code under the TCBG protocol so the comparison is apples-to-apples.

# 

# \## Install

# 

# ```bash

# pip install torch torch\_geometric scikit-learn numpy

# pip install gudhi          # accurate persistence

# pip install neurograph     # brain datasets only

# pip install -r TCBG/requirements.txt

# ```

# 

# \## Quick start

# 

# ```bash

# \# TCBG only

# python run\_tcbg\_fair.py --device cuda

# 

# \# Baselines (clones TempGNTK, runs both fair evals, prints comparison)

# python run\_all.py --device cuda

# 

# \# Skip a baseline

# python run\_all.py --device cuda --skip\_t3former

# python run\_all.py --device cuda --skip\_tempgntk

# ```

# 

# Per-domain entry points:

# 

# ```bash

# python run\_t3former\_fair.py            --device cuda 

# python run\_tempgntk\_fair.py            --device cuda

# ```

# 

# 

# \## Repo layout

# 

# ```

# run\_all.py                    one-shot harness for all baselines

# run\_tcbg\_fair.py              TCBG across datasets

# run\_tempgntk\_fair.py          TempGNTK fair eval

# run\_t3former\_fair.py          T3Former social

# compute\_stats.py              Welch's t / Cohen's d

# make\_paper\_figures.py         Figs 1–5 (PDF + PNG)

# make\_seed\_plot.py             seed-variance box plots

# 

# TCBG/                         model code (curvature, bifiltration, Graphcode, GIN)

# Baselines/                    Source codes of all baseline models

# 

# data/                         TU-format datasets

# results/                      (symlink)

# logs/                         per-run stdout/stderr

# ```

# 

# \## Reproducing the headline numbers

# 

# ```bash

# \# 1. TCBG on all datasets (10 seeds × 5-fold = 50 runs/dataset)

# python run\_tcbg\_fair.py --device cuda --seeds 10

# 

# \# 2. Baselines

# python run\_all.py --device cuda

# 

# \# 3. Stats + figures

# python compute\_stats.py

# python make\_paper\_figures.py

# ```

# 

# 

# 



