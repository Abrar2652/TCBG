#!/bin/bash
# Run TCBG on all traffic datasets (binary AND multi-class)

DEVICE=${1:-cpu}
DATA_ROOT=${2:-./data/raw}

for DATASET in pems04 pems08 pemsbay; do
  for CLASSES in 2 3; do
    echo "=== Running $DATASET ($CLASSES-class) ==="
    python experiments/train.py \
      --dataset $DATASET \
      --num_classes $CLASSES \
      --device $DEVICE \
      --data_root $DATA_ROOT \
      --result_dir ./results/traffic
  done
done
