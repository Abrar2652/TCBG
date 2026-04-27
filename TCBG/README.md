# TCBG: Temporal Curvature Bifiltration Graphcodes

**Temporal Curvature Bifiltration Graphcodes for Temporal Graph Classification**
IEEE DSAA 2026 — Submission deadline: May 30, 2026

---

## Overview

TCBG classifies temporal graphs by constructing a 2-parameter
bifiltration over (time, Forman-Ricci curvature) axes, vectorizing
the resulting persistence module as a Graphcode, and classifying
with a Graph Isomorphism Network (GIN).

Pipeline:
```
temporal graph
  → Forman-Ricci curvature per edge
  → (time, curvature) bifiltration grid
  → Graphcode (nodes = persistence bars, edges = cross-level connections)
  → GIN classifier
  → graph label
```

---

## Installation

```bash
pip install -r requirements.txt
```

For GUDHI (recommended for accurate persistence computation):
```bash
pip install gudhi
```

For NeuroGraph brain datasets:
```bash
pip install neurograph
```

---

## Quick Start

```bash
# Social network (5-fold CV)
python experiments/train.py --dataset infectious --device cpu

# Brain network (fixed 70/10/20 split)
python experiments/train.py --dataset dynhcp_task --num_classes 7 --device cuda

# Traffic (binary + multi-class)
python experiments/train.py --dataset pems04 --num_classes 2
python experiments/train.py --dataset pems04 --num_classes 3
```

---

## File Structure

```
TCBG/
├── src/
│   ├── curvature.py        Forman-Ricci curvature computation
│   ├── bifiltration.py     (time, curvature) bifiltration grid
│   ├── graphcode.py        Graphcode construction
│   ├── gin_classifier.py   GIN model
│   ├── pipeline.py         End-to-end pipeline
│   └── stability.py        Perturbation utilities
├── data/
│   ├── social_loader.py    TUDataset social graphs
│   ├── brain_loader.py     NeuroGraph DynHCP
│   ├── traffic_loader.py   PEMS + label construction
│   └── utils.py
├── experiments/
│   ├── train.py            Main training script
│   ├── ablation.py         Ablation study runner
│   ├── sensitivity.py      Grid resolution sensitivity
│   ├── runtime.py          Runtime benchmarks
│   └── visualize.py        t-SNE + Graphcode visualization
├── configs/
│   ├── default.yaml
│   ├── social.yaml
│   ├── brain.yaml
│   └── traffic.yaml
└── scripts/
    ├── run_social.sh
    ├── run_brain.sh
    ├── run_traffic.sh
    └── run_all.sh
```

---

## Datasets

| Category | Datasets | Protocol |
|----------|----------|----------|
| Social   | Infectious, DBLP, Tumblr, MIT, Highschool | 5-fold CV |
| Brain    | DynHCP-Task, DynHCP-Gender, DynHCP-Age | 70/10/20 split |
| Traffic  | PEMS04, PEMS08, PEMSBAY (binary + 3-class) | 5-fold CV |

Download social datasets: auto-downloaded via `torch_geometric.datasets.TUDataset`

Download brain datasets: `pip install neurograph`

Download traffic datasets:
- PEMS04/08: https://github.com/guoshnBJTU/ASTGCN-r-pytorch
- PEMSBAY: https://github.com/liyaguang/DCRNN

Place under `data/raw/PEMS04/`, `data/raw/PEMS08/`, `data/raw/PEMSBAY/`.

---

## Ablation Studies

```bash
# Run all ablation variants on Infectious
python experiments/ablation.py --dataset infectious --all_variants

# Single variant
python experiments/ablation.py --dataset infectious --variant time_only
```

Available variants: `full`, `time_only`, `curvature_only`, `pi_mlp`,
`degree_axis`, `edgeweight_axis`, `random_axis`, `h0_only`, `h1_only`

---

## Scores to Beat (T3Former)

| Dataset | T3Former Acc |
|---------|-------------|
| Infectious | 68.50 ± 6.30 |
| DBLP | 60.90 ± 0.70 |
| Tumblr | 63.20 ± 3.20 |
| MIT | 73.16 ± 4.13 |
| Highschool | 67.20 ± 3.20 |
| DynHCP-Task | 90.76 |
| DynHCP-Gender | 75.79 |
| DynHCP-Age | 58.73 |
| PEMS04 Binary | 96.76 ± 1.92 |
| PEMS04 Multi | 92.66 ± 1.93 |
| PEMS08 Binary | 95.16 ± 1.50 |
| PEMS08 Multi | 89.65 ± 2.02 |
| PEMSBAY Binary | 96.68 ± 1.30 |
| PEMSBAY Multi | 92.35 ± 1.52 |
