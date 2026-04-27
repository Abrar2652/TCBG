#!/bin/bash
# Run TCBG on all social network datasets

DEVICE=${1:-cpu}
DATA_ROOT=${2:-./data/raw}

for DATASET in infectious dblp tumblr mit highschool; do
  echo "=== Running $DATASET ==="
  python experiments/train.py \
    --dataset $DATASET \
    --num_classes 2 \
    --device $DEVICE \
    --data_root $DATA_ROOT \
    --result_dir ./results/social
done
