#!/bin/bash
cd /nas/home/jahin/TCBG_fair_eval
LOG=/nas/ckgfs/jaunts/jahin/tcbg_data/gen_brain_dos_loop.log
MAX=50
for i in $(seq 1 $MAX); do
  echo "=== attempt $i $(date) ===" >> $LOG
  python3 -u gen_brain_dos.py \
    --out_dir T3Former-3311/neuro_fe \
    --data_root /nas/ckgfs/jaunts/jahin/tcbg_data/brain \
    --ckpt_dir /nas/ckgfs/jaunts/jahin/tcbg_data/ckpt \
    --datasets DynHCPGender DynHCPActivity \
    --ckpt_every 40 >> $LOG 2>&1
  ec=$?
  echo "=== exit=$ec $(date) ===" >> $LOG
  if [ $ec -eq 0 ]; then
    echo "=== all done ===" >> $LOG
    break
  fi
  sleep 5
done
