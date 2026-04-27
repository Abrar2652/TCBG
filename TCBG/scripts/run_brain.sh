#!/bin/bash
# Run TCBG on all brain network datasets

DEVICE=${1:-cpu}
DATA_ROOT=${2:-./data/raw}

for DATASET in dynhcp_task dynhcp_gender dynhcp_age; do
  CLASSES=2
  if [ "$DATASET" = "dynhcp_task" ]; then CLASSES=7; fi
  if [ "$DATASET" = "dynhcp_age" ];  then CLASSES=3; fi

  echo "=== Running $DATASET ($CLASSES classes) ==="
  python experiments/train.py \
    --dataset $DATASET \
    --num_classes $CLASSES \
    --device $DEVICE \
    --data_root $DATA_ROOT \
    --result_dir ./results/brain
done
