#!/bin/bash
cd /workspace
nohup python3 same_item_alignment/scripts/s3_run_solver.py \
  --solver_id solver_3_smolvlm2_2b \
  --items_parquet same_item_alignment/data/eedi948_item_manifest.parquet \
  --out same_item_alignment/data/raw_predictions/solver3_full.csv \
  > /workspace/logs_solver3.txt 2>&1 &
echo "STARTED_PID_$!"
