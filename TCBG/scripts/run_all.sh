#!/bin/bash
# Run full TCBG benchmark across all 14 tasks

DEVICE=${1:-cpu}
DATA_ROOT=${2:-./data/raw}

bash scripts/run_social.sh  $DEVICE $DATA_ROOT
bash scripts/run_brain.sh   $DEVICE $DATA_ROOT
bash scripts/run_traffic.sh $DEVICE $DATA_ROOT

echo ""
echo "All experiments done. Results in ./results/"
