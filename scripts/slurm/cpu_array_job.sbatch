#!/bin/bash
# Run every line of a command file as one SLURM array task (CPU only).
#   sbatch --array=1-$(grep -cv '^\s*\(#\|$\)' jobs.txt)%4 scripts/slurm/array_job.sbatch jobs.txt
# "%4" caps how many tasks run at once. Blank lines and lines starting with # are skipped.
#SBATCH --job-name=elecgnn-cpuarray
#SBATCH --partition=CHANGE_ME
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out

set +eu   # strict mode only after conda activation (its hooks use unset variables)
ENV=${ELECGNN_ENV:-elecgnn}   # conda environment name or path; export ELECGNN_ENV to override
eval "$(conda shell.bash hook)"
conda activate "$ENV"
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
CMD=$(grep -v '^\s*\(#\|$\)' "$1" | sed -n "${SLURM_ARRAY_TASK_ID}p")
echo "task ${SLURM_ARRAY_TASK_ID}  host $(hostname)  gpu ${CUDA_VISIBLE_DEVICES:-none}  cmd: $CMD"
srun bash -c "$CMD"
