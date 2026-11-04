#!/usr/bin/env bash
# Does adding the graph-level scalars (solvation free energy, dielectric constant) help?
# Runs GAT and GCN with each scalar mode on identical folds, then paired tests vs "none".
#
# Usage: bash scripts/run_scalar_ablation.sh [ioncon|lce] [fold_col]
#   ioncon defaults to the leave-solvent-out folds; lce uses the original split.
set -euo pipefail

TASK=${1:-ioncon}
FOLD_COL=${2:-fold_solvent}
DATA_DIR=${DATA_DIR:-data}
OUT=${OUT:-results/scalar_ablation/${TASK}_${FOLD_COL}}

if [[ "$TASK" == "ioncon" ]]; then
  DATA="$DATA_DIR/unified_md.csv"
  SPLIT=(--split file --folds-file "$DATA_DIR/folds_inhouse.csv" --fold-col "$FOLD_COL")
else
  DATA="$DATA_DIR/Ref_LCEdata.xlsx"
  SPLIT=(--split original)
fi

for model in gat gcn; do
  for scalars in none gsol dielec both; do
    elecgnn train --task "$TASK" --data "$DATA" --scalar-db "$DATA_DIR/solvent_db_1_clean.csv" \
      --model "$model" --scalars "$scalars" "${SPLIT[@]}" --out "$OUT/${model}_${scalars}"
  done
  for scalars in gsol dielec both; do
    echo "== $model: $scalars vs none (MAE) =="
    elecgnn compare "$OUT/${model}_${scalars}" "$OUT/${model}_none" --metric MAE
  done
done
