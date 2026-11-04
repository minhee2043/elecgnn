#!/bin/bash
# Run one light command (analysis, figures, interpret on CPU) through SLURM without a GPU.
#   sbatch scripts/slurm/cpu_job.sbatch python scripts/make_figures.py --results results --data data --out figures
#SBATCH --job-name=elecgnn-cpu
#SBATCH --partition=CHANGE_ME
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out

set +eu   # strict mode only after conda activation (its hooks use unset variables)
ENV=${ELECGNN_ENV:-elecgnn}   # conda environment name or path; export ELECGNN_ENV to override
eval "$(conda shell.bash hook)"
conda activate "$ENV"
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
srun "$@"
