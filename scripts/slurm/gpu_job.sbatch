#!/bin/bash
# Run one elecgnn command on one GPU through SLURM.
#   sbatch scripts/slurm/gpu_job.sbatch elecgnn train --task ioncon ... --out results/x
# Edit the partition (and limits) for your cluster: `sinfo -o "%P %G %l"` lists partitions, GPUs, time limits.
#SBATCH --job-name=elecgnn
#SBATCH --partition=CHANGE_ME
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

set +eu   # strict mode only after conda activation (its hooks use unset variables)
ENV=${ELECGNN_ENV:-elecgnn}   # conda environment name or path; export ELECGNN_ENV to override
eval "$(conda shell.bash hook)"
conda activate "$ENV"
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}   # SLURM sets CUDA_VISIBLE_DEVICES itself
echo "host $(hostname)  gpu ${CUDA_VISIBLE_DEVICES:-none}  cmd: $*"
srun "$@"
