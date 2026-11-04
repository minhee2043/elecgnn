#!/usr/bin/env bash
# Two-step OEDB study (RQ3/RQ5): OEDB-predicted electrolyte-level properties as inputs.
#
#   bash scripts/run_oedb_features.sh predictor          # 1. train the OEDB predictor once
#   bash scripts/run_oedb_features.sh ioncon fold_solvent
#   bash scripts/run_oedb_features.sh ioncon fold_ethers
#   bash scripts/run_oedb_features.sh lce
#
# Each task run trains the GAT with every feature set on identical folds and prints
# paired tests against the no-feature baseline.
set -euo pipefail

STEP=${1:?usage: run_oedb_features.sh predictor|ioncon|lce [fold_col]}
FOLD_COL=${2:-fold_solvent}
DATA_DIR=${DATA_DIR:-data}
FEAT_DIR=${FEAT_DIR:-$DATA_DIR/oedb_features}
SETS=(transport structure physics all eta sigma)

if [[ "$STEP" == "predictor" ]]; then
  elecgnn oedb-features --unified "$DATA_DIR/unified_md.csv" --lce "$DATA_DIR/Ref_LCEdata.xlsx" \
    --n-models 5 --out "$FEAT_DIR"
  # Zero-shot check: does the OEDB predictor alone rank the in-house systems (363 K)?
  python - "$DATA_DIR" "$FEAT_DIR" <<'PY'
import sys
import pandas as pd
from scipy.stats import spearmanr
d, f = sys.argv[1:]
u = pd.read_csv(f"{d}/unified_md.csv").query("source == 'inhouse'")
p = pd.read_csv(f"{f}/features_ioncon.csv")
m = u.merge(p, on="row_id")
print("zero-shot Spearman, OEDB-predicted log sigma vs in-house sigma:")
print(f"  all        n={len(m):3d}  rho={spearmanr(m.log_sigma, m.sigma_mS_cm)[0]:.2f}")
for c, g in m.groupby("solvent_class"):
    if len(g) >= 5:
        print(f"  {c:22s} n={len(g):3d}  rho={spearmanr(g.log_sigma, g.sigma_mS_cm)[0]:.2f}")
PY
  exit 0
fi

OUT=${OUT:-results/oedb_features/${STEP}_${FOLD_COL}}
if [[ "$STEP" == "ioncon" ]]; then
  BASE=(--task ioncon --data "$DATA_DIR/unified_md.csv" --split file
        --folds-file "$DATA_DIR/folds_inhouse.csv" --fold-col "$FOLD_COL")
  [[ "$FOLD_COL" == "fold_ethers" ]] && BASE+=(--test-folds 0)
  FEAT="$FEAT_DIR/features_ioncon.csv"
else
  OUT=${OUT/_${FOLD_COL}/}
  BASE=(--task lce --data "$DATA_DIR/Ref_LCEdata.xlsx" --split original)
  FEAT="$FEAT_DIR/features_lce.csv"
fi

elecgnn train "${BASE[@]}" --model gat --out "$OUT/none"
for fs in "${SETS[@]}"; do
  elecgnn train "${BASE[@]}" --model gat --extra-features "$FEAT" --feature-set "$fs" --out "$OUT/$fs"
done
for fs in "${SETS[@]}"; do
  echo "== $fs vs none (MAE) =="
  elecgnn compare "$OUT/$fs" "$OUT/none" --metric MAE
done
