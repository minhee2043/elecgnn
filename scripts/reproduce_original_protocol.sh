#!/usr/bin/env bash
# Reproduce the original-protocol runs (0617_ioncon_GAT, 0507_ioncon_GCN, 0508_LCE_GAT/GCN):
# random folds shared across seeds, early stopping on the test fold, dipole
# pretraining, no graph-level scalars, sigma in mS/cm before log1p.
# Expect agreement within about one fold standard deviation, not exact numbers:
# the cleaned data, batch order and library versions differ.
set -euo pipefail
DATA_DIR=${DATA_DIR:-data}
OUT=${OUT:-results/original_protocol}
COMMON=(--split original --early-stop outer --scalars none
        --pretrain dipole --pretrain-data "$DATA_DIR/HOMO_LUMO_data.xlsx")

for model in gat gcn; do
  elecgnn train --task ioncon --data "$DATA_DIR/unified_md.csv" --model "$model" --transform log1p \
    "${COMMON[@]}" --out "$OUT/ioncon_$model"
  elecgnn train --task lce --data "$DATA_DIR/Ref_LCEdata.xlsx" --model "$model" \
    "${COMMON[@]}" --out "$OUT/lce_$model"
done
